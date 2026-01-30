# Twinkle Eval 快速入門

## 安裝

```bash
pip install twinkle-eval
```

或從原始碼安裝：

```bash
cd Eval
pip install -e .
```

## 兩種評測系統

Twinkle Eval 提供兩種獨立的評測系統：

### 1️⃣ 一般評測（標準題庫）

評測模型在 TMMLU+、MMLU、tw-legal 等**標準選擇題**題庫上的表現。

```bash
# 初始化配置
twinkle-eval --init
# → 創建 config.yaml

# 編輯配置
nano config.yaml

# 執行評測
twinkle-eval --config config.yaml
```

### 2️⃣ LLM-as-Judge 評測（品質評估）

使用 LLM 評估另一個 LLM 的**輸出品質**（相關性、忠實度、自定義標準）。

```bash
# 初始化配置
twinkle-eval-judge --init
# → 創建 config_llm_as_judge.yaml

# 編輯配置
nano config_llm_as_judge.yaml

# 執行評測
twinkle-eval-judge --config config_llm_as_judge.yaml
```

## 完整流程示範

### 情境 1: 評測模型在 TMMLU+ 上的表現

```bash
# 步驟 1: 初始化配置
twinkle-eval --init

# 步驟 2: 編輯 config.yaml
# 設定：
# - llm_api.api_key（您的 API 金鑰）
# - model.name（模型名稱）
# - evaluation.dataset_paths（資料集路徑）

# 步驟 3: 準備資料集
mkdir -p dataset
# 將 TMMLU+ 資料放入 dataset/ 目錄

# 步驟 4: 執行評測
twinkle-eval --config config.yaml

# 結果會儲存在 results/ 目錄
```

### 情境 2: 評估 RAG 系統輸出品質

```bash
# 步驟 1: 初始化配置
twinkle-eval-judge --init

# 步驟 2: 編輯 config_llm_as_judge.yaml
# 設定：
# - judge_model.api_key（Judge 模型的 API 金鑰）
# - metrics（啟用 faithfulness 和 answer_relevancy）

# 步驟 3: 準備測試案例
# 創建 test_cases.jsonl：
cat > test_cases.jsonl << 'EOF'
{"input": "What is the Eiffel Tower?", "actual_output": "The Eiffel Tower is a landmark in Paris.", "retrieval_context": ["The Eiffel Tower is located in Paris, France."]}
EOF

# 步驟 4: 執行評測
twinkle-eval-judge --config config_llm_as_judge.yaml

# 結果會儲存在 results/llm_as_judge/ 目錄
```

### 情境 3: 同時使用兩種評測

```bash
# 初始化兩個配置
twinkle-eval --init
twinkle-eval-judge --init

# 分別編輯
nano config.yaml
nano config_llm_as_judge.yaml

# 分別執行
twinkle-eval --config config.yaml
twinkle-eval-judge --config config_llm_as_judge.yaml
```

## 配置文件說明

### `config.yaml` - 一般評測配置

```yaml
llm_api:
  base_url: "https://api.example.com/v1"
  api_key: "your-api-key"

model:
  name: "gemma-3-12b-it"
  temperature: 0.0

evaluation:
  dataset_paths:
    - dataset
  evaluation_method: "box"
  shuffle_options: true
```

**關鍵設定**:
- `llm_api.api_key` - 您的 API 金鑰（必須）
- `model.name` - 要評測的模型名稱（必須）
- `evaluation.dataset_paths` - 資料集目錄（必須）

### `config_llm_as_judge.yaml` - LLM-as-Judge 配置

```yaml
llm_as_judge:
  judge_model:
    type: "openai"
    model_name: "gpt-4"
    api_key: "your-api-key"
    base_url: "https://api.openai.com/v1"

  metrics:
    - type: "answer_relevancy"
      threshold: 0.7

    - type: "faithfulness"
      threshold: 0.8

  test_cases:
    file: "test_cases.jsonl"

  output:
    directory: "results/llm_as_judge"
```

**關鍵設定**:
- `judge_model.api_key` - Judge 模型的 API 金鑰（必須）
- `metrics` - 要使用的評估指標（必須）
- `test_cases` - 測試案例來源（必須）

## 目錄結構

建議的專案結構：

```
your-project/
├── config.yaml                      # 一般評測配置
├── config_llm_as_judge.yaml         # LLM-as-Judge 配置
│
├── dataset/                         # 一般評測資料集
│   ├── tmmlu/
│   │   ├── science.csv
│   │   └── math.csv
│   └── mmlu/
│       └── ...
│
├── test_cases.jsonl                 # LLM-as-Judge 測試案例
│
└── results/                         # 評測結果
    ├── results_20250117.json        # 一般評測結果
    ├── eval_results_20250117.jsonl
    └── llm_as_judge/                # LLM-as-Judge 結果
        ├── llm_as_judge_results_*.json
        └── llm_as_judge_summary_*.json
```

## 常用命令

### 一般評測

```bash
# 創建配置
twinkle-eval --init

# 執行評測（使用預設配置）
twinkle-eval

# 執行評測（指定配置）
twinkle-eval --config config.yaml

# 匯出多種格式
twinkle-eval --export json csv html

# 查看支援的功能
twinkle-eval --list-llms
twinkle-eval --list-strategies
twinkle-eval --list-exporters

# 顯示版本
twinkle-eval --version

# 顯示幫助
twinkle-eval --help
```

### LLM-as-Judge

```bash
# 創建配置
twinkle-eval-judge --init

# 執行評測
twinkle-eval-judge --config config_llm_as_judge.yaml

# 顯示版本
twinkle-eval-judge --version

# 顯示幫助
twinkle-eval-judge --help
```

## 測試案例格式

### 一般評測資料集（CSV）

```csv
question,A,B,C,D,answer
"1+1等於多少？","1","2","3","4","B"
```

### LLM-as-Judge 測試案例（JSONL）

```jsonl
{"input": "What is Python?", "actual_output": "Python is a programming language.", "expected_output": "A programming language"}
{"input": "Explain RAG", "actual_output": "RAG combines retrieval and generation.", "retrieval_context": ["RAG stands for Retrieval-Augmented Generation."]}
```

或 JSON 格式：

```json
[
  {
    "input": "What is Python?",
    "actual_output": "Python is a programming language.",
    "expected_output": "A programming language",
    "metadata": {"category": "tech"}
  }
]
```

## 進階功能

### 使用本地模型

**一般評測**：

```yaml
llm_api:
  base_url: "http://localhost:8000/v1"  # vLLM/Ollama
  api_key: "EMPTY"
  disable_ssl_verify: true

model:
  name: "meta-llama/Llama-3.2-3B-Instruct"
```

**LLM-as-Judge**：

```yaml
llm_as_judge:
  judge_model:
    model_name: "meta-llama/Llama-3.2-3B-Instruct"
    api_key: "EMPTY"
    base_url: "http://localhost:8000/v1"
    disable_ssl_verify: true
```

### 自定義 G-Eval 評估標準

```yaml
llm_as_judge:
  metrics:
    - type: "g_eval"
      name: "Code Quality"
      criteria: |
        評估程式碼品質：
        1. 正確性
        2. 可讀性
        3. 效率
      evaluation_params:
        - "input"
        - "actual_output"
      rubric:
        - score_range: [0, 3]
          description: "品質差"
        - score_range: [4, 6]
          description: "尚可"
        - score_range: [7, 10]
          description: "優秀"
      threshold: 0.7
```

## 常見問題

### Q: 兩種評測有什麼區別？

**一般評測**：
- ✅ 評測標準選擇題（TMMLU+, MMLU）
- ✅ 自動化評分（比對選項）
- ✅ 適合基準測試

**LLM-as-Judge**：
- ✅ 評估開放式輸出品質
- ✅ 使用 LLM 進行語義評估
- ✅ 適合實際應用場景

### Q: 可以共用 API 金鑰嗎？

A: 可以！兩個配置文件可以使用相同的 API 端點和金鑰。

### Q: 結果保存在哪裡？

A:
- 一般評測：`results/`
- LLM-as-Judge：`results/llm_as_judge/`

### Q: 如何選擇使用哪種評測？

| 需求 | 使用 |
|------|------|
| 評測模型在標準題庫上的分數 | 一般評測 |
| 評估 RAG 系統輸出品質 | LLM-as-Judge |
| 評估程式碼生成品質 | LLM-as-Judge |
| 比較不同模型在 MMLU 上的表現 | 一般評測 |
| 分析客服回答相關性 | LLM-as-Judge |

## 更多資源

- 📖 [完整文檔](README.md)
- 🔧 [LLM-as-Judge 詳細指南](twinkle_eval/llm_as_judge/README.md)
- 📝 [配置使用說明](twinkle_eval/llm_as_judge/CONFIG_USAGE.md)
- 💡 [雙配置方法說明](TWO_CONFIG_APPROACH.md)
- 🧪 [範例配置](config_llm_as_judge.yaml)

## 需要幫助？

- GitHub Issues: https://github.com/ai-twinkle/Eval/issues
- Discord: https://discord.gg/Cx737yw4ed

---

**開始使用吧！** 🚀

選擇您需要的評測類型，執行 `--init` 創建配置，然後開始評測！
