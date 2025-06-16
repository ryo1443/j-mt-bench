#!/usr/bin/env python
# coding: utf-8
"""
OPENAI_API_KEY=<key> python evaluate_jp_mtbench.py \
   --answer-file  outputs/Qwen3-4B/answers.jsonl \
   --judge-model  gpt-4o \
   --question-file jp_mtbench_80.jsonl \
   --save-dir     outputs/Qwen3-4B
"""
import argparse, json, pathlib, statistics, re
from tqdm import tqdm
from openai import OpenAI         # ✅ v1.x import style  :contentReference[oaicite:2]{index=2}

client = OpenAI()                 # ✅ クライアント生成  :contentReference[oaicite:3]{index=3}

SYSTEM = ("あなたは評価者です。与えられた日本語の質問と LLM の回答を読み、"
          "0〜10 の整数で品質を評価し、その根拠を簡潔に述べてください。")

TEMPLATE = """### 質問
{question}

### 回答
{answer}

### 指示
1. 回答を内容の正確さ・網羅性・日本語の流暢さで総合評価し、0〜10 の整数を返す。
2. 次の行にその評価理由を 1 文で書く。
フォーマット例:
Score: 8
理由: ...
"""

def score_once(model: str, q: str, a: str) -> int:
    """GPT-4o に 1 件採点してもらい整数スコアを返す（v1 API）。"""
    res = client.chat.completions.create(         # ✅ 新メソッド  :contentReference[oaicite:4]{index=4}
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user",   "content": TEMPLATE.format(question=q, answer=a)}
        ],
    )
    txt = res.choices[0].message.content.strip()
    m = re.search(r"\d+", txt)
    return int(m.group()) if m else -1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--answer-file", required=True)
    ap.add_argument("--judge-model", required=True)
    ap.add_argument("--question-file", required=True)
    ap.add_argument("--save-dir", required=True)
    args = ap.parse_args()

    pathlib.Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    outf = pathlib.Path(args.save_dir) / f"scores_{args.judge_model}.jsonl"
    if outf.exists():
        raise FileExistsError(f"{outf} already exists")

    # 質問ロード
    qmap = {}
    with open(args.question_file, encoding="utf-8") as f:
        for ln in f:
            r = json.loads(ln)
            qmap[str(r["question_id"])] = {
                "text": "\n\n".join(r["turns"]),
                "cat":  r["category"]
            }

    cat, all_scores = {}, []
    with open(args.answer_file, encoding="utf-8") as f, outf.open("w", encoding="utf-8") as fw:
        for ln in tqdm(f):
            ans = json.loads(ln)
            q = qmap[str(ans["question_id"])]
            s = score_once(args.judge_model, q["text"], ans["answer"])
            ans["score_"+args.judge_model] = s
            fw.write(json.dumps(ans, ensure_ascii=False)+"\n")
            cat.setdefault(q["cat"], []).append(s)
            all_scores.append(s)

    print("\n=== Category Averages ===")
    for c, lst in sorted(cat.items()):
        print(f"{c:<10}: {statistics.mean(lst):.2f}")
    print(f"\nOverall     : {statistics.mean(all_scores):.2f}")

if __name__ == "__main__":
    main()
