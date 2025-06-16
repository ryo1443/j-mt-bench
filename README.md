
# j-mt-bench

Japanese MT‑Bench を使って  
* ローカル **LLM / AWQ 量子化モデル**  
* **OpenAI GPT‑4o**  

のどちらでも回答を生成し、GPT‑4 系モデルで自動採点できるスクリプト集です。

```
j-mt-bench/
├─ src/
│   ├─ generate_any_jp_mtbench.py   # 生成（ローカル or GPT-4o）
│   └─ evaluate_jp_mtbench.py       # 採点（GPT-4o / 4.5 / o1 / o3…）
├─ data/
│   └─ jp_mtbench_80.jsonl          # 80 問 default セット
└─ requirements.txt                 # 依存ライブラリ
```

---

## クイックスタート

### 1) ローカル Qwen3‑4B で回答生成

```bash
python src/generate_any_jp_mtbench.py \
       --model-path /path/to/Qwen3-4B \
       --question-file data/jp_mtbench_80.jsonl \
       --save-dir outputs/Qwen3-4B
```

### 2) GPT‑4o で採点

```bash
OPENAI_API_KEY=sk-... \
python src/evaluate_jp_mtbench.py \
       --answer-file outputs/Qwen3-4B/answers.jsonl \
       --judge-model gpt-4o \
       --question-file data/jp_mtbench_80.jsonl \
       --save-dir outputs/Qwen3-4B
```

---

## スクリプト概要

| ファイル | 役割 |
|----------|------|
| `generate_any_jp_mtbench.py` | ローカル LLM または GPT‑4o で 80 問回答を生成し、`answers.jsonl` 出力 |
| `evaluate_jp_mtbench.py` | `answers.jsonl` を GPT‑4 系モデルで採点し、カテゴリ平均・総合平均を表示 |
| `data/jp_mtbench_80.jsonl` | Japanese MT‑Bench “default” 80 問セット |

各スクリプトの先頭 docstring に詳しい使い方を記載しています。  
