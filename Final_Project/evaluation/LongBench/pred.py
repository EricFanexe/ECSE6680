import os
from datasets import load_dataset
import torch
import json
from transformers import (
    AutoTokenizer,
    LlamaTokenizer,
    LlamaForCausalLM,
    AutoModelForCausalLM,
)
from tqdm import tqdm
import numpy as np
import random
import argparse
from evaluation.flash_attn_monkey_patch import (
    replace_llama_attn_with_flash_attn,
    replace_mistral_attn_with_flash_attn,
)
from evaluation.quest_attention import enable_quest_attention_eval


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=[
            "llama2-7b-chat-4k",
            "longchat-v1.5-7b-32k",
            "xgen-7b-8k",
            "internlm-7b-8k",
            "chatglm2-6b",
            "chatglm2-6b-32k",
            "chatglm3-6b-32k",
            "vicuna-v1.5-7b-16k",
            "Mistral-7B-v0.2-hf",
        ],
    )
    parser.add_argument("--e", action="store_true", help="Evaluate on LongBench-E")
    parser.add_argument("--task", type=str, help="task name", required=True)
    parser.add_argument("--token_budget", type=int, default=None)
    parser.add_argument("--chunk_size", type=int, default=None)
    parser.add_argument("--quest", action="store_true", help="Enable Quest Attention")
    parser.add_argument("--start_index", type=int, default=0, help="Start index of input sequence")
    return parser.parse_args(args)


# Customized building prompt for chat models
def build_chat(tokenizer, prompt, model_name):
    if "chatglm3" in model_name:
        prompt = tokenizer.build_chat_input(prompt)
    elif "chatglm" in model_name:
        prompt = tokenizer.build_prompt(prompt)
    elif "longchat" in model_name or "vicuna" in model_name:
        from fastchat.model import get_conversation_template

        conv = get_conversation_template("vicuna")
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    elif "llama2" in model_name:
        prompt = f"[INST]{prompt}[/INST]"
    elif "xgen" in model_name:
        header = (
            "A chat between a curious human and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n"
        )
        prompt = header + f" ### Human: {prompt}\n###"
    elif "internlm" in model_name:
        prompt = f"<|User|>:{prompt}<eoh>\n<|Bot|>:"
    return prompt


def post_process(response, model_name):
    if "xgen" in model_name:
        response = response.strip().replace("Assistant:", "")
    elif "internlm" in model_name:
        response = response.split("<eoa>")[0]
    return response


def get_pred_single(
    model,
    tokenizer,
    json_obj,
    max_length,
    max_gen,
    prompt_format,
    dataset,
    device,
    model_name,
):
    """
    对单条数据进行推理并返回结果。
    """
    prompt = prompt_format.format(**json_obj)

    # truncate to fit max_length
    tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
    if "chatglm3" in model_name:
        tokenized_prompt = tokenizer(
            prompt, truncation=False, return_tensors="pt", add_special_tokens=False
        ).input_ids[0]

    if len(tokenized_prompt) > max_length:
        half = int(max_length / 2)
        prompt = (
            tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)
            + tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        )

    # 对部分数据集做对话式包装
    if dataset not in [
        "trec",
        "triviaqa",
        "samsum",
        "lsht",
        "lcc",
        "repobench-p",
    ]:
        prompt = build_chat(tokenizer, prompt, model_name)

    # 寻找 question 的位置（如果需要）
    if dataset in ["qasper", "hotpotqa"]:
        q_pos = prompt.rfind("Question:")
    elif dataset in ["multifieldqa_en", "gov_report"]:
        q_pos = prompt.rfind("Now,")
    elif dataset in ["triviaqa"]:
        q_pos = prompt.rfind("Answer the question")
    elif dataset in ["narrativeqa"]:
        q_pos = prompt.rfind("Do not provide")
    else:
        q_pos = -1

    # 只取后100个字符中找 question 标记，避免意外截断
    q_pos = max(len(prompt) - 100, q_pos)

    if q_pos != None and q_pos != -1:
        question = prompt[q_pos:]
        prompt = prompt[:q_pos]
    else:
        question = ""

    # 构造输入
    if "chatglm3" in model_name:
        # chatglm3 的拼接方式可能需要根据官方文档定制
        input_ids = tokenizer(prompt, truncation=False, return_tensors="pt")["input_ids"].to(device)
        # 额外 question 的拼接
        question_ids = tokenizer(question, truncation=False, return_tensors="pt")["input_ids"].to(device)
    else:
        input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
        q_input = tokenizer(question, truncation=False, return_tensors="pt").to(device)
        # 对 question 去掉开头的 token
        if q_input.input_ids.shape[-1] > 1:
            q_input.input_ids = q_input.input_ids[:, 1:]

    with torch.no_grad():
        # 先前向 prompt
        if "chatglm3" in model_name:
            output = model(
                input_ids=input_ids,
                past_key_values=None,
                use_cache=True,
            )
            past_key_values = output.past_key_values
            # 前向 question
            if question_ids.shape[-1] > 0:
                output = model(
                    input_ids=question_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                past_key_values = output.past_key_values
        else:
            output = model(
                input_ids=input.input_ids,
                past_key_values=None,
                use_cache=True,
            )
            past_key_values = output.past_key_values
            # question
            for input_id in q_input.input_ids[0]:
                output = model(
                    input_ids=input_id.unsqueeze(0).unsqueeze(0),
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                past_key_values = output.past_key_values

        # 从最后一个位置开始生成
        pred_token_idx = output.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
        generated_content = [pred_token_idx.item()]

        for _ in range(max_gen - 1):
            outputs = model(
                input_ids=pred_token_idx,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
            token_id = pred_token_idx.item()
            generated_content.append(token_id)
            # 如果生成了 EOS，就停止
            if token_id == tokenizer.eos_token_id:
                break

    pred = tokenizer.decode(generated_content, skip_special_tokens=True)
    pred = post_process(pred, model_name)

    return {
        "pred": pred,
        "answers": json_obj["answers"],
        "all_classes": json_obj["all_classes"],
        "length": json_obj["length"],
    }


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def load_model_and_tokenizer(path, model_name, device):
    """
    根据 model_name 加载相应的模型与 tokenizer。
    """
    if "chatglm" in model_name or "internlm" in model_name or "xgen" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            path, trust_remote_code=True, torch_dtype=torch.bfloat16
        ).to(device)
    elif "llama2" in model_name:
        # 替换 Llama 注意力为 Flash Attention
        tokenizer = LlamaTokenizer.from_pretrained(path)
        model = LlamaForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16).to(
            device
        )
    elif "longchat" in model_name or "vicuna" in model_name:
        replace_llama_attn_with_flash_attn()
        model = AutoModelForCausalLM.from_pretrained(
            path, trust_remote_code=True, torch_dtype=torch.float16, device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(
            path, trust_remote_code=True, use_fast=False
        )
    elif "Mistral" in model_name:
        replace_mistral_attn_with_flash_attn()
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto"
        )
    model = model.eval()

    # 启用 Quest Attention (如果需要)
    if args.quest:
        save_folder = f"/home/fanz2/SPARC/evaluation/LongBench/{args.task}-{args.token_budget}"
        # 为了确保文件夹存在，可以先创建一下
        os.makedirs(save_folder, exist_ok=True)

        # 把保存路径记录到模型 config 里
        model.config.quest_save_folder = save_folder

        enable_quest_attention_eval(model, args)

    return model, tokenizer


if __name__ == "__main__":
    seed_everything(42)
    args = parse_args()

    model2path = json.load(open("config/model2path.json", "r"))
    model2maxlen = json.load(open("config/model2maxlen.json", "r"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = args.model

    # 初始化模型
    model, tokenizer = load_model_and_tokenizer(model2path[model_name], model_name, device)
    max_length = model2maxlen[model_name]

    # 判断要跑哪些数据集
    if args.e:
        datasets = [
            "qasper",
            "multifieldqa_en",
            "hotpotqa",
            "2wikimqa",
            "gov_report",
            "multi_news",
            "trec",
            "triviaqa",
            "samsum",
            "passage_count",
            "passage_retrieval_en",
            "lcc",
            "repobench-p",
        ]
    else:
        datasets = [args.task]

    # 读入提示模板和最大生成长度
    dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))

    # 建立输出目录
    if not os.path.exists("pred_50_test"):
        os.makedirs("pred_50_test")
    if not os.path.exists("pred_e_50_test"):
        os.makedirs("pred_e_50_test")

    for dataset in datasets:
        # 选择数据
        if args.e:
            data = load_dataset("THUDM/LongBench", f"{dataset}_e", split="test")
            data = data.select(range(1))
            if args.start_index < len(data):
                data = data.select(range(args.start_index, len(data)))

            if not os.path.exists(f"pred_e_50_test/{model_name}"):
                os.makedirs(f"pred_e_50_test/{model_name}")
            if args.quest:
                out_path = f"pred_e_50_test/{model_name}/{dataset}-{args.token_budget}.jsonl"
            else:
                out_path = f"pred_e_50_test/{model_name}/{dataset}.jsonl"
        else:
            data = load_dataset("THUDM/LongBench", dataset, split="test")
            data = data.select(range(1))
            if args.start_index < len(data):
                data = data.select(range(args.start_index, len(data)))
    
            if not os.path.exists(f"pred_50_test/{model_name}"):
                os.makedirs(f"pred_50_test/{model_name}")
            if args.quest:
                out_path = f"pred_50_test/{model_name}/{dataset}-{args.token_budget}.jsonl"
            else:
                out_path = f"pred_50_test/{model_name}/{dataset}.jsonl"

        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]

        # 逐条推理并输出
        with open(out_path, "a", encoding="utf-8") as f:
            for json_obj in tqdm(data):
                pred = get_pred_single(
                    model,
                    tokenizer,
                    json_obj,
                    max_length,
                    max_gen,
                    prompt_format,
                    dataset,
                    device,
                    model_name,
                )
                # 立刻写入 .jsonl
                json.dump(pred, f, ensure_ascii=False)
                f.write("\n")
