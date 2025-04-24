import math
import numpy as np
from typing import Optional, Tuple, Union
import os
import torch
from torch import nn
import torch.utils.checkpoint
import torch.nn.functional as F
from cuml.cluster import KMeans
from torch.utils.dlpack import to_dlpack
import cupy as cp
from cupy import fromDlpack

import types

from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    apply_rotary_pos_emb,
    repeat_kv,
)


def local_heavy_hitter_mask(attn_weights, token_budget, chunk_size):
    # attn_weights (BS, head, query, keys)

    # expend attn_weights to be divisible by chunk_size
    seq_length = attn_weights.shape[-1]
    padding_length = chunk_size - ((seq_length - 1) % chunk_size + 1)
    attn_weights = torch.cat(
        [
            attn_weights,
            torch.ones(
                (
                    attn_weights.shape[0],
                    attn_weights.shape[1],
                    attn_weights.shape[2],
                    padding_length,
                ),
                device=attn_weights.device,
            )
            * torch.tensor(torch.finfo(attn_weights.dtype).min),
        ],
        dim=-1,
    )

    # chunk attn_weights into chunk_size tokens
    chunk_attn_weights = attn_weights.reshape(
        attn_weights.shape[0],
        attn_weights.shape[1],
        attn_weights.shape[2],
        attn_weights.shape[3] // chunk_size,
        chunk_size,
    ).amax(dim=-1)

    _, topk = chunk_attn_weights.topk(
        k=min(max(3, token_budget // chunk_size), chunk_attn_weights.size(-1)), dim=-1
    )

    # (BS, head, query, chunks)
    topk_return = topk
    # repeat topk chunk_size times and recover the original indexes (* chunk_size + arange(chunk_size))
    topk = topk.unsqueeze(-1).repeat(
        1, 1, 1, 1, chunk_size
    ) * chunk_size + torch.arange(chunk_size, device=topk.device)
    topk = topk.reshape(topk.shape[0], topk.shape[1], topk.shape[2], -1)
    mask_bottom = torch.zeros_like(attn_weights, dtype=torch.bool)
    mask_bottom.scatter_(-1, topk, True)

    # remove the padding
    mask_bottom = mask_bottom[:, :, :, :seq_length]

    return mask_bottom, topk_return


def cluster_heavy_hitter_mask(
    query_states: torch.Tensor,         # [B, H, q_len, D]
    centroids: torch.Tensor,            # [B, H, n_clusters, D]
    labels: torch.Tensor,               # [B, H, L], the cluster id for each token
    reorder_map: torch.Tensor,          # [B, H, L], maps the clustered order back to the original sequence
    B: int,                             # KV budget (excluding tokens with label==-1)
    L_orig: int,                        # original KV sequence length
) -> torch.Tensor:
    """
    Return mask_bottom_original: [B, H, q_len, L_orig] as a bool mask.

    Optimization points:
      1) First, unify tokens with label = -1, keep them, and subtract their count from the budget B;
      2) For the remaining tokens (label >= 0), perform a single torch.sort(labels_bh),
         find each cluster's range [start_idx, end_idx) in the sorted order,
         to avoid doing a boolean index (labels_bh == c_id_val) for each cluster individually.
      3) Then use sorted_cluster_idx to determine the priority of clusters and allocate the budget in order.
    """
    device = query_states.device
    bsz, num_heads, q_len, d = query_states.shape
    # centroids shape [B, H, n_clusters, D]
    _, _, n_clusters, _ = centroids.shape

    # ------------------------------------------------------
    # (1) Compute each cluster's score => scores: [B, H, n_clusters]
    # ------------------------------------------------------
    centroids = centroids.to(query_states.dtype)

    scores_4d = torch.einsum("bhqd,bhcd->bhqc", query_states, centroids) / math.sqrt(d)

    scores = scores_4d.squeeze(2)

    sorted_scores, sorted_cluster_idx = torch.sort(scores, dim=-1, descending=True)

    # ------------------------------------------------------
    # (2) Initialize the final mask: [B, H, q_len, L_orig]
    # ------------------------------------------------------
    mask_bottom_original = torch.zeros(
        (bsz, num_heads, q_len, L_orig),
        dtype=torch.bool,
        device=device,
    )

    # ------------------------------------------------------
    # (3) Main loop: process (b_i, h_i) one by one
    # ------------------------------------------------------
    for b_i in range(bsz):
        for h_i in range(num_heads):
            labels_bh = labels[b_i, h_i]      # [L]
            pos_bh    = reorder_map[b_i, h_i] # [L]

            # =========== (3.1) First deal with tokens where label=-1 ===========
            unclustered_mask = (labels_bh == -1)
            unclustered_positions = pos_bh[unclustered_mask]
            num_unclustered = unclustered_positions.numel()

            # Keep all tokens with label=-1
            if num_unclustered > 0:
                mask_bottom_original[b_i, h_i, :, unclustered_positions] = True

            # Subtract these from the budget
            budget_remaining = B - num_unclustered
            if budget_remaining < 0:
                budget_remaining = 0

            # =========== (3.2) Sort tokens with label>=0 (ascending) ===========
            # Goal: let tokens from the same cluster group together, so we can get
            # them in O(1) time from their range later
            # Filter out label==-1 first, to reduce unnecessary overhead
            non_neg_mask = (labels_bh >= 0)
            if not torch.any(non_neg_mask):
                continue

            # Get tokens with label>=0
            labels_non_neg = labels_bh[non_neg_mask]  
            pos_non_neg    = pos_bh[non_neg_mask]    

            # Sort labels_non_neg => ascending
            sort_labels, sort_idx = torch.sort(labels_non_neg)  # ascending by label
            # The positions in the original sequence must also be rearranged consistently
            pos_sorted = pos_non_neg[sort_idx]

            # =========== (3.3) Scan sort_labels to find each cluster's interval [start, end) ===========
            # cluster_positions_index[c] = (start_index, end_index)
            # indicates sort_labels[start_index:end_index] are all label=c
            cluster_positions_index = [None] * n_clusters  # Python list

            # Use a "difference" method to determine the boundaries of each interval at once
            # First find where the label value changes
            # shape=[K], these are positions where sort_labels[i] != sort_labels[i+1]
            diff_mask = (sort_labels[1:] != sort_labels[:-1])
            diff_idx  = torch.where(diff_mask)[0]  # [K], ascending

            # boundaries => [0] + (diff_idx+1) + [len(sort_labels)]
            # boundaries[i] ~ boundaries[i+1] is one block of identical labels
            boundaries = torch.cat([
                diff_idx.new_zeros((1,)),         
                diff_idx + 1,                     
                diff_idx.new_tensor([sort_labels.size(0)])  
            ], dim=0)

            # Traverse all intervals
            # boundaries[i], boundaries[i+1] => a block of the same label
            for seg_i in range(len(boundaries) - 1):
                start_i = boundaries[seg_i].item()
                end_i   = boundaries[seg_i + 1].item()
                c_label = sort_labels[start_i].item()  # the label value of this interval
                # Ensure c_label is in [0, n_clusters)
                if 0 <= c_label < n_clusters:
                    cluster_positions_index[c_label] = (start_i, end_i)

            # =========== (3.4) Pick clusters in descending order of sorted_cluster_idx ===========
            c_ids_ordered = sorted_cluster_idx[b_i, h_i]  # shape=[n_clusters], descending by scores

            for c_id in c_ids_ordered:
                c_id_val = c_id.item()
                if cluster_positions_index[c_id_val] is None:
                    continue

                start_i, end_i = cluster_positions_index[c_id_val]
                # sort_labels[start_i:end_i] are all label=c_id_val
                # the corresponding original sequence positions => pos_sorted[start_i:end_i]
                cluster_positions = pos_sorted[start_i:end_i]
                csize = cluster_positions.size(0)

                if csize <= budget_remaining:
                    # keep the entire cluster
                    mask_bottom_original[b_i, h_i, :, cluster_positions] = True
                    budget_remaining -= csize
                else:
                    # only keep part of the cluster
                    part_positions = cluster_positions[:budget_remaining]
                    mask_bottom_original[b_i, h_i, :, part_positions] = True
                    budget_remaining = 0

                if budget_remaining == 0:
                    break 

    return mask_bottom_original


def cluster_keys_with_cuml_kmeans(
    keys: torch.Tensor,      # shape: [L, D]
    values: torch.Tensor,    # shape: [L, D]
    n_clusters: int,
) -> tuple:
    """
    Use cuML KMeans to cluster the keys, then reorder them according to the cluster labels.
    Returns the rearranged keys, rearranged values, reorder_map, labels in new order, and centroids_cp.
    """
    device = keys.device
    
    data_normalized = F.normalize(keys, p=2, dim=-1)

    dlpack_tensor = to_dlpack(data_normalized)
    data_cp = cp.fromDlpack(dlpack_tensor)  # [L, D] on GPU
    dev_id = data_cp.device.id   

    with cp.cuda.Device(dev_id):
        kmeans = KMeans(
            n_clusters=n_clusters,
            max_iter=15,
            init='k-means++',
            random_state=0,
        )
        kmeans.fit(data_cp)

    labels_cp = kmeans.labels_
    centroids_cp = kmeans.cluster_centers_

    labels_t = torch.tensor(labels_cp, dtype=torch.long, device=device)

    sorted_idx = torch.argsort(labels_t, dim=0)
    rearranged_keys   = keys[sorted_idx]
    rearranged_values = values[sorted_idx]
    labels_new_order  = labels_t[sorted_idx]
    reorder_map       = sorted_idx

    return rearranged_keys, rearranged_values, reorder_map, labels_new_order, centroids_cp


def cluster_on_every_head(
    key_states: torch.Tensor,     # [1, H, L, D]
    value_states: torch.Tensor,   # [1, H, L, D]
    chunk_size: int,
):
    """
    Cluster on every head for batch_size=1. chunk_size is indirectly used to determine n_clusters.
    Returns rearranged keys, values, reorder_map, labels, and centroids for each head.
    """
    device = key_states.device
    # bsz=1
    bsz, num_heads, seq_len, head_dim = key_states.shape
    assert bsz == 1, "This function only supports batch_size=1"

    # Use chunk_size to indirectly decide the number of clusters
    n_clusters = (seq_len + chunk_size - 1) // chunk_size

    # Use lists to store per-head results
    rearranged_keys_heads = []
    rearranged_values_heads = []
    reorder_maps_heads = []
    labels_heads = []
    centroids_heads = []

    for h_idx in range(num_heads):
        rearranged_k, rearranged_v, reorder_map, labels_t, centroids_cp = cluster_keys_with_cuml_kmeans(
            keys=key_states[0, h_idx],      # shape: [L, D]
            values=value_states[0, h_idx],  # shape: [L, D]
            n_clusters=n_clusters,
        )
        rearranged_keys_heads.append(rearranged_k)
        rearranged_values_heads.append(rearranged_v)
        reorder_maps_heads.append(reorder_map)

        centroids_t = torch.tensor(centroids_cp, device=device)

        labels_heads.append(labels_t)
        centroids_heads.append(centroids_t)

    # stack => [H, L, D] / [H, L], etc.
    rearranged_keys_tensor  = torch.stack(rearranged_keys_heads,   dim=0)   # [H, L, D]
    rearranged_values_tensor= torch.stack(rearranged_values_heads, dim=0)   # [H, L, D]
    reorder_map_tensor      = torch.stack(reorder_maps_heads,      dim=0)   # [H, L]
    labels_tensor           = torch.stack(labels_heads,            dim=0)   # [H, L]
    centroids_tensor        = torch.stack(centroids_heads,         dim=0)   # [H, n_clusters, D]

    # Add a batch dimension size=1
    rearranged_keys_tensor  = rearranged_keys_tensor.unsqueeze(0)    # => [1, H, L, D]
    rearranged_values_tensor= rearranged_values_tensor.unsqueeze(0)  # => [1, H, L, D]
    reorder_map_tensor      = reorder_map_tensor.unsqueeze(0)        # => [1, H, L]
    labels_tensor           = labels_tensor.unsqueeze(0)             # => [1, H, L]
    centroids_tensor        = centroids_tensor.unsqueeze(0)          # => [1, H, n_clusters, D]

    return (
        rearranged_keys_tensor,        # [1, H, L, D]
        rearranged_values_tensor,      # [1, H, L, D]
        reorder_map_tensor,            # [1, H, L]
        labels_tensor,                 # [1, H, L]
        centroids_tensor,              # [1, H, n_clusters, D]
    )


def forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor, ...]] = None,
    output_attentions: bool = False,
    use_cache: bool = True,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, ...]]]:
    """
    Custom forward function with on-the-fly clustering for partial decoding.
    """
    bsz, q_len, _ = hidden_states.size()
    # 0) If it is prefill or layer_id<2, use the original logic
    if q_len > 1 or self.layer_id < 2:
        return self.flash_forward(
            hidden_states,
            attention_mask,
            position_ids,
            past_key_value,
            output_attentions,
            use_cache,
            **kwargs,
        )

    # ------------------------------------------------
    # 1) Parse our custom past_key_value structure
    # ------------------------------------------------
    if past_key_value is not None:
        if len(past_key_value) == 2:
            old_k, old_v = past_key_value
            original_k, original_v = past_key_value
            old_clustered_k = None
            old_clustered_v = None
            old_reorder_map = None
            decode_step = 0
            new_token_count = 0
            old_labels = None
            cluster_centers = None
        elif len(past_key_value) == 11:
            original_k, original_v, old_k, old_v, old_clustered_k, old_clustered_v, old_reorder_map, decode_step, new_token_count, old_labels, cluster_centers = past_key_value
        else:
            raise ValueError("Unsupported past_key_value format!")
    else:
        old_k, old_v = None, None
        original_k, original_v = None, None
        old_clustered_k, old_clustered_v = None, None
        old_reorder_map = None
        decode_step, new_token_count = 0, 0
        old_labels = None
        cluster_centers = None

    # ------------------------------------------------
    # 2) Standard Q/K/V
    # ------------------------------------------------
    query_states = (
        self.q_proj(hidden_states)
        .view(bsz, q_len, self.num_heads, self.head_dim)
        .transpose(1, 2)
    )
    key_states = (
        self.k_proj(hidden_states)
        .view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        .transpose(1, 2)
    )
    value_states = (
        self.v_proj(hidden_states)
        .view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        .transpose(1, 2)
    )

    # Rotary
    kv_seq_len = key_states.shape[-2]

    if original_k is not None:
        kv_seq_len += original_k.shape[-2]
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    if original_k is not None:
        key_states_temp = torch.cat([original_k, key_states], dim=2)
        value_states_temp = torch.cat([original_v, value_states], dim=2)
    else:
        key_states_temp = key_states
        value_states_temp = value_states

    if old_k is not None:
        key_states = torch.cat([old_k, key_states], dim=2)
        value_states = torch.cat([old_v, value_states], dim=2)

    # Cache new "original KV" (no clustering)
    new_past_k = key_states
    new_past_v = value_states

    # ------------------------------------------------
    # 3) decode_step & new_token_count
    # ------------------------------------------------
    decode_step += 1
    new_token_count += 1
    # ------------------------------------------------
    # 4) Clustering: first decode => cluster the entire sequence; otherwise cluster every 128 tokens
    # ------------------------------------------------
    # old_clustered_k / old_clustered_v: already clustered
    # key_states / value_states: currently unclustered (BS, nHeads, seqLen, headDim)
    if decode_step == 1:
        # First decode => cluster the entire KV
        old_clustered_k, old_clustered_v, old_reorder_map, old_labels, cluster_centers = cluster_on_every_head(key_states, value_states, self.chunk_size)

        new_past_k = None
        new_past_v = None
        key_states = key_states[:, :, :0, :]  # empty tensor
        value_states = value_states[:, :, :0, :]
        new_token_count = 0
    else:
        # Subsequent decoding
        if new_token_count >= 128:
            # Only cluster new tokens
            new_clustered_k, new_clustered_v, new_reorder_map, new_labels, new_cluster_centers = cluster_on_every_head(key_states, value_states, self.chunk_size)

            if old_reorder_map is not None and old_reorder_map.numel() > 0:
                offset_map = old_reorder_map.max().item() + 1
            else:
                offset_map = 0

            new_reorder_map = new_reorder_map + offset_map

            if old_labels is not None and old_labels.numel() > 0:
                offset_label = old_labels.max().item() + 1
            else:
                offset_label = 0

            new_labels = new_labels + offset_label

            # Concatenate
            if old_clustered_k is not None and old_clustered_v is not None:
                updated_k = torch.cat([old_clustered_k, new_clustered_k], dim=2)
                updated_v = torch.cat([old_clustered_v, new_clustered_v], dim=2)
                updated_map = torch.cat([old_reorder_map, new_reorder_map], dim=2)  # [B, H, L_old+L_new]
                updated_label = torch.cat([old_labels, new_labels], dim=2)
            else:
                updated_k = new_clustered_k
                updated_v = new_clustered_v
                updated_map = new_reorder_map
                updated_label = new_labels

            old_clustered_k = updated_k
            old_clustered_v = updated_v
            old_reorder_map = updated_map
            new_past_k = None
            new_past_v = None
            key_states = key_states[:, :, :0, :]
            value_states = value_states[:, :, :0, :]
            old_labels = updated_label
            cluster_centers = torch.cat([cluster_centers, new_cluster_centers], dim=2)
            # Reset after clustering
            new_token_count = 0

    if old_clustered_k is not None and old_clustered_v is not None:
        L_new = key_states.shape[2]
        if old_reorder_map.numel() > 0:
            current_offset = old_reorder_map.max().item() + 1
        else:
            current_offset = 0
            
        new_reorder_map_identity = torch.arange(
            current_offset,
            current_offset + L_new,
            device=old_reorder_map.device
        ).unsqueeze(0).unsqueeze(0)   # shape [1,1,L_new]
        # Let it broadcast to [B, H, L_new]
        bsz = key_states.shape[0]
        num_heads = key_states.shape[1]
        new_reorder_map_identity = new_reorder_map_identity.expand(bsz, num_heads, L_new)
        final_map = torch.cat([old_reorder_map, new_reorder_map_identity], dim=2)
        key_states = torch.cat([old_clustered_k, key_states], dim=2)
        value_states = torch.cat([old_clustered_v, value_states], dim=2)

        # -------- Assign label=-1 to these new tokens --------
        if old_labels is not None:
            new_label_segment = torch.full(
                (bsz, num_heads, L_new),
                fill_value=-1,
                dtype=old_labels.dtype,
                device=old_labels.device,
            )
            final_labels = torch.cat([old_labels, new_label_segment], dim=2)
        # -----------------------------------------------------

    # Step 5: cluster-based filtering

    if old_clustered_k is not None and old_clustered_v is not None:
        # key_states_temp, value_states_temp are "original order" K/V

        mask_bottom_original = cluster_heavy_hitter_mask(
            query_states=query_states,                 # [B, H, q_len, D]
            centroids=cluster_centers,                 # [B, H, n_clusters, D] 
            labels=final_labels,                       # [B, H, L]
            reorder_map=final_map,                     # [B, H, L]
            B=self.token_budget,                       # KV budget
            L_orig=key_states_temp.shape[-2],
        )

        # 在层 forward 内部，如果需要保存 npy，就读取 config 里的保存路径
        save_folder = getattr(self.config, "quest_save_folder", None)
        # ------------------------------ 
        # 输出需要的文件（仅当 layer_id 为 8, 16, 24 时） 
        # ------------------------------
        if self.layer_id in [4, 8, 12, 16, 20, 24, 28] and save_folder is not None:
            # 创建 main folder
            os.makedirs(save_folder, exist_ok=True)
            
            # 分别创建 masks 和 labels 子文件夹
            masks_subfolder = os.path.join(save_folder, "masks")
            labels_subfolder = os.path.join(save_folder, "labels")
            os.makedirs(masks_subfolder, exist_ok=True)
            os.makedirs(labels_subfolder, exist_ok=True)

            # 1) 保存 mask_bottom_original 到 masks 子文件夹
            mask_path = os.path.join(
                masks_subfolder,
                f"mask_bottom_original_layer{self.layer_id}_step_{decode_step}.npy"
            )
            np.save(mask_path, mask_bottom_original.cpu().numpy())

            # 2) 保存原顺序 cluster labels 到 labels 子文件夹
            if final_map is not None and final_labels is not None:
                b, h, L = final_map.shape  # 假设 b = 1
                # 初始化一个值全部为 -1 的张量
                cluster_labels_in_original = torch.full(
                    (h, key_states_temp.shape[-2]), 
                    -1,  # 未聚类标记
                    dtype=final_labels.dtype,
                    device=final_labels.device
                )
                # 使用向量化操作，将 final_labels 中的数据根据 final_map 指定的位置散列到 cluster_labels_in_original 中
                # 这里 final_map[0] 的 shape 为 (h, L)，与 final_labels[0] 的 shape (h, L) 对应
                cluster_labels_in_original.scatter_(1, final_map[0], final_labels[0])
                
                labels_path = os.path.join(
                    labels_subfolder,
                    f"cluster_labels_layer{self.layer_id}_step_{decode_step}.npy"
                )
                np.save(labels_path, cluster_labels_in_original.cpu().numpy())


    attn_weights_orig = torch.matmul(query_states, key_states_temp.transpose(2, 3)) / math.sqrt(self.head_dim)
    # if self.layer_id in [4, 12, 20, 28] and q_len == 1:
    #     save_dir = "/home/fanz2/SPARC/evaluation/saved_keys/"
    #     os.makedirs(save_dir, exist_ok=True)

    #     file_path = os.path.join(save_dir, f"keys_layer{self.layer_id}_step{decode_step}.pt")
    #     torch.save(key_states_temp.cpu(), file_path)
    #     print(f"[INFO] Saved full keys at decode step {decode_step} -> {file_path}")
    
    # Below code for recall rate calculation is commented out in the original snippet.
    # You can uncomment it if needed for debugging.

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #           计算 "recall rate"
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # if self.token_budget > 0 and attn_weights_orig is not None and self.layer_id == 16:
    #     # 对所有 (b, h, q) 计算前 B 大 tokens 的下标，并统计其中有多少被 mask_bottom_original 选中
    #     B = self.token_budget 
    #     recall_sum = 0.0
    #     count = 0

    #     for b in range(bsz):
    #         for h in range(num_heads):
    #             for q in range(q_len):
    #                 # attn_weights_orig[b, h, q, :] 形状为 [L_orig]
    #                 weights_1d = attn_weights_orig[b, h, q, :]
    #                 # 取 top-B 的索引
    #                 topB = weights_1d.topk(B, largest=True)
    #                 topB_indices = topB.indices  # shape: [B]

    #                 # 取对应的 mask_bottom_original[b, h, q, :] => bool [L_orig]
    #                 chosen_mask = mask_bottom_original[b, h, q, :]

    #                 # 计算交集数量
    #                 intersection_count = chosen_mask[topB_indices].sum().item()

    #                 # 当前 q 的 recall = intersection / B
    #                 recall_sum += intersection_count / B
    #                 count += 1

    #     recall_rate = recall_sum / count
    #     print(f"[DEBUG] Decode step {decode_step} recall_rate = {recall_rate:.4f}")
        
        # # 根据不同的 token_budget 写入不同文件
        # output_file_budget = f"recall_rate_budget_{self.token_budget}.txt"
        # with open(output_file_budget, "a") as f:
        #     f.write(f"Decode step {decode_step}: recall_rate = {recall_rate:.4f}\n")
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # if attention_mask is not None:
    #     attn_weights_orig = attn_weights_orig + attention_mask
    #     attn_weights_orig = torch.max(
    #         attn_weights_orig,
    #         torch.tensor(torch.finfo(attn_weights_orig.dtype).min, device=attn_weights_orig.device),
    #     )
    mask_bottom_original = torch.tril(mask_bottom_original, diagonal=position_ids[0][0].item())

    attn_weights_orig[~mask_bottom_original] = torch.tensor(
        torch.finfo(attn_weights_orig.dtype).min,
        device=attn_weights_orig.device,
        dtype=attn_weights_orig.dtype
    )

    attn_weights_orig = nn.functional.softmax(attn_weights_orig, dim=-1, dtype=torch.float32).to(query_states.dtype)

    attn_output = torch.matmul(attn_weights_orig, value_states_temp)
    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` shape mismatch. Expected {(bsz, self.num_heads, q_len, self.head_dim)}, got {attn_output.size()}."
        )

    attn_output = attn_output.transpose(1, 2)
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights_orig = None
    # ------------------------------------------------
    # 6) Pack new information back into past_key_value and return
    # ------------------------------------------------
    updated_past_key_value = None
    if use_cache:
        updated_past_key_value = (
            key_states_temp,
            value_states_temp,
            new_past_k,              # original K (no clustering)
            new_past_v,              # original V
            old_clustered_k,         # clustered K
            old_clustered_v,         # clustered V
            old_reorder_map,
            decode_step,
            new_token_count,
            old_labels,
            cluster_centers
        )

    return attn_output, attn_weights_orig, updated_past_key_value



def forward_yarn(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    is_padded_inputs: Optional[bool] = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, h_size = hidden_states.size()

    # Prefill stage utilizes flash attention
    if q_len > 1 or self.layer_id < 2:
        return self.flash_forward(
            hidden_states,
            attention_mask,
            position_ids,
            past_key_value,
            output_attentions,
            use_cache,
            is_padded_inputs,
        )

    has_layer_past = past_key_value is not None

    if has_layer_past:
        past_kv = past_key_value[0]
        past_len = past_key_value[1]
    else:
        past_len = 0

    if self.config.pretraining_tp > 1:
        key_value_slicing = (
            self.num_key_value_heads * self.head_dim
        ) // self.config.pretraining_tp
        query_slices = self.q_proj.weight.split(
            (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
        )
        key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
        value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

        q = [
            F.linear(hidden_states, query_slices[i])
            for i in range(self.config.pretraining_tp)
        ]
        q = torch.cat(q, dim=-1)

        k = [
            F.linear(hidden_states, key_slices[i])
            for i in range(self.config.pretraining_tp)
        ]
        k = torch.cat(k, dim=-1)

        v = [
            F.linear(hidden_states, value_slices[i])
            for i in range(self.config.pretraining_tp)
        ]
        v = torch.cat(v, dim=-1)

    else:
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

    q = q.view(bsz, q_len, self.num_heads, self.head_dim)
    k = k.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
    v = v.view(bsz, q_len, self.num_key_value_heads, self.head_dim)

    q, k = self.rotary_emb(q, k, past_len)

    @torch.jit.script
    def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        """
        This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
        num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
        """
        batch, slen, _, num_key_value_heads, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, :, :, None, :].expand(
            batch, slen, 2, num_key_value_heads, n_rep, head_dim
        )
        return hidden_states.reshape(
            batch, slen, 2, num_key_value_heads * n_rep, head_dim
        )

    kv = torch.stack([k, v], 2)
    kv = repeat_kv(kv, self.num_key_value_groups)

    # Cache QKV values
    if has_layer_past:
        new_len = past_len + q.size(1)
        if new_len > past_kv.size(1):
            past_kv = torch.cat(
                [
                    past_kv,
                    torch.empty(
                        bsz,
                        256,
                        2,
                        kv.size(3),
                        kv.size(4),
                        dtype=kv.dtype,
                        device=kv.device,
                    ),
                ],
                1,
            )
        past_kv[:, past_len:new_len] = kv
        kv = past_kv[:, :new_len]
    else:
        past_kv = kv

    k, v = kv.split(1, dim=2)
    k = k.squeeze(2)
    v = v.squeeze(2)

    past_key_value = (past_kv, past_len + q.size(1)) if use_cache else None

    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    kv_seq_len = k.shape[-2]

    attn_weights = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)

    sign = (q > 0) + (~(q > 0)) * -1
    max_key = k * sign
    postive_query = q * sign

    # expend max_key to be divisible by chunk_size
    seq_length = max_key.shape[-2]
    padding_length = self.chunk_size - ((seq_length - 1) % self.chunk_size + 1)
    max_key = torch.cat(
        [
            max_key,
            torch.ones(
                (max_key.shape[0], max_key.shape[1], padding_length, max_key.shape[3]),
                device=max_key.device,
            )
            * torch.tensor(torch.finfo(max_key.dtype).min),
        ],
        dim=-2,
    )

    # chunk max_key into chunk_size tokens
    chunk_max_key = max_key.reshape(
        max_key.shape[0],
        max_key.shape[1],
        max_key.shape[2] // self.chunk_size,
        self.chunk_size,
        max_key.shape[3],
    ).amax(dim=-2)

    # duplicate chunk_max_key chunk_size times
    chunk_max_key = chunk_max_key.unsqueeze(-2).repeat(1, 1, 1, self.chunk_size, 1)
    # reshape chunk_max_key to the original shape
    chunk_max_key = chunk_max_key.reshape(
        chunk_max_key.shape[0], chunk_max_key.shape[1], -1, chunk_max_key.shape[-1]
    )[:, :, :seq_length, :]

    quantized_weight = torch.matmul(
        postive_query.float(),
        chunk_max_key.transpose(2, 3),
    )

    if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
        raise ValueError(
            f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
            f" {attn_weights.size()}"
        )

    assert q_len == 1, "Prefill stage utilizes flash attention."

    token_budget = min(kv_seq_len, self.token_budget)

    attn_weights_for_selection = quantized_weight
    # attn_weights_for_selection = attn_weights

    if token_budget > 0:
        mask_bottom = local_heavy_hitter_mask(
            attn_weights_for_selection, token_budget, self.chunk_size
        )  # Default: No padding applied to input
    else:
        mask_bottom = torch.zeros_like(attn_weights_for_selection, dtype=torch.bool)

    # Attention mask for multi-stage Q&A, todo
    mask_bottom = torch.tril(mask_bottom, diagonal=k.shape[-2] - q.shape[-2])
    attn_weights[~mask_bottom] = torch.tensor(torch.finfo(attn_weights.dtype).min)

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        q.dtype
    )
    attn_output = torch.matmul(attn_weights, v)

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    if self.config.pretraining_tp > 1:
        attn_output = attn_output.split(
            self.hidden_size // self.config.pretraining_tp, dim=2
        )
        o_proj_slices = self.o_proj.weight.split(
            self.hidden_size // self.config.pretraining_tp, dim=1
        )
        attn_output = sum(
            [
                F.linear(attn_output[i], o_proj_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
        )
    else:
        attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


global layer_id
layer_id = 32


def enable_quest_attention_eval(model, args):
    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            enable_quest_attention_eval(
                module,
                args,
            )

        global layer_id
        if isinstance(module, LlamaAttention):
            # For longchat model
            layer_id -= 1
            model._modules[name].layer_id = layer_id
            model._modules[name].flash_forward = model._modules[name].forward
            model._modules[name].forward = types.MethodType(
                forward, model._modules[name]
            )

            model._modules[name].token_budget = args.token_budget
            model._modules[name].chunk_size = args.chunk_size
        elif module.__class__.__name__ == "LlamaAttention":
            # For yarn model
            layer_id -= 1
            model._modules[name].layer_id = layer_id
            model._modules[name].flash_forward = model._modules[name].forward
            model._modules[name].forward = types.MethodType(
                forward_yarn, model._modules[name]
            )

            model._modules[name].token_budget = args.token_budget
            model._modules[name].chunk_size = args.chunk_size
