#!/usr/bin/env python
# coding: utf-8
"""
Japanese MT-Bench (80 問 default) の回答生成
------------------------------------------------
ローカルモデル or OpenAI GPT-x のどちらでも実行できる。

例) ローカル Qwen3-4B
    python generate_any_jp_mtbench.py \
        --model-path /home/ec2-user/Qwen3-4B \
        --save-dir   outputs/Qwen3-4B \
        --question-file jp_mtbench_80.jsonl

例) GPT-4o
    OPENAI_API_KEY=<key> \
    python generate_any_jp_mtbench.py \
        --model-path gpt-4o \
        --save-dir   outputs/GPT4o \
        --question-file jp_mtbench_80.jsonl \
        --sleep 0.5
"""

import argparse, json, pathlib, time, sys, re, os
from datasets import load_dataset
from tqdm import tqdm

# ────────────── OpenAI helper（インポートは後置き） ──────────────
OPENAI_MODELS = {"gpt-4o", "gpt-4-turbo", "gpt-4.1", "gpt-4o-mini", "o1", "o3"}

def call_openai(model: str, turns: list[str], max_tokens: int, temperature: float):
    from openai import OpenAI                          # 1.x client
    client = OpenAI()
    system_prompt = "あなたは優秀な日本語の AI アシスタントです。指示に答えてください。"
    messages = [{"role": "system", "content": system_prompt}]
    for t in turns:
        messages.append({"role": "user", "content": t})

    rsp = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    return rsp.choices[0].message.content.strip()

# ────────────── Transformers / AWQ helper ──────────────
def load_local_model(model_path: str):
    """AWQ or FP16"""
    if "awq" in model_path.lower():
        from awq import AutoAWQForCausalLM
        model = AutoAWQForCausalLM.from_quantized(
            model_path, device_map="auto", trust_remote_code=True)
    else:
        from transformers import AutoModelForCausalLM
        import torch
        model = AutoModelForCausalLM.from_pretrained(
            model_path, device_map="auto",
            torch_dtype=torch.bfloat16, trust_remote_code=True)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_path,
                                        use_fast=False, trust_remote_code=True)
    # device
    if hasattr(model, "device"):
        device = model.device
    else:  # AWQ
        device = next(model.parameters()).device
    return model, tok, device

def gen_local_answer(model, tok, device, turns, max_new_tokens, temperature):
    prompt = "\n\n".join(turns) + "\n\n答え:"
    import torch
    inputs = tok(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        ids = model.generate(**inputs,
                             max_new_tokens=max_new_tokens,
                             temperature=temperature,
                             do_sample=True)
    return tok.decode(ids[0][inputs["input_ids"].shape[-1]:],
                      skip_special_tokens=True).strip()

# ────────────── main ──────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True,
                    help="ローカルモデル dir か OpenAI モデル名 (gpt-4o など)")
    ap.add_argument("--question-file", required=True,
                    help="'hf-default' で HuggingFace から 80 問取得も可")
    ap.add_argument("--save-dir", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=1024,
                    help="ローカルモデル用 max_new_tokens / OpenAI max_tokens")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--sleep", type=float, default=1.0,
                    help="OpenAI 連続呼び出し間隔 (秒)")
    args = ap.parse_args()

    out_dir = pathlib.Path(args.save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "answers.jsonl"
    if out_path.exists():
        sys.exit(f"{out_path} already exists — rename or delete first.")

    # 質問データ
    ds = (load_dataset("naive-puzzle/japanese-mt-bench", "default", split="train")
          if args.question_file == "hf-default"
          else load_dataset("json", data_files=args.question_file)["train"])

    use_openai = args.model_path in OPENAI_MODELS

    if use_openai:
        print(f"▶ Generating with OpenAI model: {args.model_path}")
    else:
        print(f"▶ Loading local model: {args.model_path}")
        model, tok, device = load_local_model(args.model_path)

    with out_path.open("w", encoding="utf-8") as fw:
        for rec in tqdm(ds, total=len(ds)):
            if use_openai:
                ans = call_openai(args.model_path,
                                  rec["turns"],
                                  max_tokens=args.max_new_tokens,
                                  temperature=args.temperature)
                time.sleep(args.sleep)
            else:
                ans = gen_local_answer(model, tok, device,
                                       rec["turns"],
                                       max_new_tokens=args.max_new_tokens,
                                       temperature=args.temperature)
            fw.write(json.dumps({
                "question_id": int(rec["question_id"]),
                "category":    rec["category"],
                "model":       args.model_path,
                "answer":      ans
            }, ensure_ascii=False) + "\n")

    print(f"\n✔ Saved to {out_path}")

if __name__ == "__main__":
    main()
