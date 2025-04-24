import torch
from tqdm import tqdm
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.nn import CrossEntropyLoss

import argparse
from argparse import ArgumentParser
device = "cuda"

parser = argparse.ArgumentParser()
parser.add_argument("--model_name_or_path", type=str)
parser.add_argument("--fixed-length", type=int)
parser.add_argument("--max_tokens", type=int, default=8192)
parser.add_argument("--min_tokens", type=int, default=256)
parser.add_argument("--tokens-step", type=int)
parser.add_argument("--length-step", type=int, default=128)
parser.add_argument("--iterations", type=int, default=20)
parser.add_argument("--output_dir", type=str)
parser.add_argument("--start_idx", type=int, default=0, help="Starting index of the token")
parser.add_argument("--num_eval_tokens", type=int, default=None)
parser.add_argument("--quest", action="store_true", help="Enable quest attention")
parser.add_argument("--token_budget", type=int, default=1024)
parser.add_argument("--chunk_size", type=int, default=16)


def load(model_name_or_path):
    print(f"Loading model from {model_name_or_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.pad_token_id = 0

    model.eval()

    return model, tokenizer


args = parser.parse_args()

# data = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
data = load_dataset("emozilla/pg19-test", split="test")

model, tokenizer = load(args.model_name_or_path)

nlls = []
ppls = []
lengths = []
loss_fn = CrossEntropyLoss(reduction="none")
past_key_values = None


if args.quest:
    print("Enable quest attention")
    from evaluation.quest_attention_origin import (
        enable_quest_attention_eval,
    )

    enable_quest_attention_eval(model, args)

os.makedirs(args.output_dir, exist_ok=True)
f = open(f"{args.output_dir}/log_PG19_full.txt", "w")

num_eval_tokens = 0
for text in data["text"]:
    encodings = tokenizer(text, return_tensors="pt")
    seq_len = encodings.input_ids.size(1)
    print(f"seq_len: {seq_len}")
    
    # 检查文本长度，跳过过短的文本
    if seq_len <= args.min_tokens:
        print(f"Skipping text with seq_len {seq_len} (less than min tokens {args.min_tokens})")
        continue
    
    start_idx = args.start_idx

    if start_idx >= seq_len:
        print(f"Start index {start_idx} exceeds sequence length {seq_len}.")
        break
    
    context_input_ids = encodings.input_ids[:, :start_idx + 1].to(device)

    # 将上下文输入模型以获得 past_key_values
    with torch.no_grad():
        outputs = model(
            context_input_ids,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values  # 保存上下文的past_key_values

    # 从 start_idx 开始逐步递增输入序列的长度，并添加进度条
    pbar = tqdm(range(start_idx, seq_len - 1), desc="Processing tokens")
    for idx in pbar:
        input_ids = encodings.input_ids[:, idx : idx + 1].to(device)
        with torch.no_grad():
            outputs = model(
                input_ids,
                past_key_values=past_key_values,  # 使用上下文的past_key_values
                use_cache=True,
            )
            logits = outputs.logits.view(-1, model.config.vocab_size)
            past_key_values = outputs.past_key_values
            label = encodings.input_ids[:, idx + 1 : idx + 2].to(logits.device).view(-1)
            neg_log_likelihood = loss_fn(logits, label)

        nlls.append(neg_log_likelihood)
        mean_nll = torch.stack(nlls).mean()  # 累积到当前的 NLL 平均值
        overall_ppl = torch.exp(mean_nll).item()
        
        # 更新进度条显示每一步的 nll 和 ppl
        pbar.set_description(f"nll: {neg_log_likelihood.item():.2f}, ppl: {overall_ppl:.2f}")
        
        # 输出每步的困惑度到 log.txt
        print(overall_ppl, file=f, flush=True)

        num_eval_tokens += 1
        if args.num_eval_tokens is not None and num_eval_tokens >= args.num_eval_tokens:
            break

    # 在处理完符合条件的序列后立即退出文本的循环
    print(f"Processed a sequence with start_idx {start_idx}. Stopping further processing for this text.")
    break  # 可选：如果只需要处理一个序列，处理完一段 text 后退出

f.close()


ppl = torch.exp(torch.stack(nlls).mean())
print(ppl.item())
with open(f"{args.output_dir}/ppl_PG19_full.txt", "w") as f:
    f.write(f"{ppl.item()}\n")
