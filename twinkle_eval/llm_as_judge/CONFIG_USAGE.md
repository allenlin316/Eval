# 使用配置文件執行 LLM-as-Judge 評估

這份文檔說明如何使用 YAML 配置文件直接執行 LLM-as-Judge 評估，**完全不需要寫程式碼**。

## 快速開始

### 1. 創建範例配置文件

```bash
twinkle-eval-judge --init
```

這會在當前目錄創建 `config_llm_as_judge.yaml` 範例文件。

### 2. 編輯配置文件

打開 `config_llm_as_judge.yaml` 並設定：

```yaml
llm_as_judge:
  judge_model:
    api_key: "your-actual-api-key"  # 設定你的 API 金鑰
    # ... 其他設定
```

### 3. 執行評估

```bash
twinkle-eval-judge --config config_llm_as_judge.yaml
```

就這麼簡單！

## 配置文件結構

### 基本結構

```yaml
llm_as_judge:
  judge_model:      # Judge 模型配置
  metrics:          # 評估指標列表
  test_cases:       # 測試案例
  output:           # 輸出設定
```

### Judge 模型配置

```yaml
judge_model:
  type: "openai"                                    # 模型類型
  model_name: "gpt-4"                               # 模型名稱
  api_key: "sk-..."                                 # API 金鑰
  base_url: "https://api.openai.com/v1"             # API 端點
  temperature: 0.0                                   # 溫度（0.0 = 確定性）
  max_tokens: 4096                                   # 最大輸出 tokens
  timeout: 600                                       # 超時時間（秒）
  max_retries: 3                                     # 重試次數
  disable_ssl_verify: false                          # SSL 驗證
  supports_structured_output: false                  # 結構化輸出支援
  supports_json_mode: true                           # JSON 模式支援
```

#### 使用本地模型（vLLM/Ollama）

```yaml
judge_model:
  type: "openai"
  model_name: "meta-llama/Llama-3.2-3B-Instruct"
  api_key: "EMPTY"
  base_url: "http://localhost:8000/v1"              # 本地 vLLM 服務
  temperature: 0.0
  disable_ssl_verify: true
```

#### 使用 Azure OpenAI

```yaml
judge_model:
  type: "openai"
  model_name: "gpt-4"
  api_key: "your-azure-api-key"
  base_url: "https://your-resource.openai.azure.com/openai/deployments/gpt-4"
  temperature: 0.0
```

### 評估指標配置

可以同時配置多個評估指標：

#### Answer Relevancy（答案相關性）

```yaml
metrics:
  - type: "answer_relevancy"
    enabled: true
    threshold: 0.7                # 成功閾值（0-1）
```

#### Faithfulness（忠實度）- 適用於 RAG

```yaml
metrics:
  - type: "faithfulness"
    enabled: true
    threshold: 0.8
```

#### G-Eval（自定義評估）

基本用法：

```yaml
metrics:
  - type: "g_eval"
    enabled: true
    name: "Correctness"             # 指標名稱
    criteria: |                     # 評估標準
      評估實際輸出是否正確回答了輸入問題。
      考慮：
      1. 答案是否事實正確
      2. 答案是否完整
    evaluation_params:              # 使用的參數
      - "input"
      - "actual_output"
      - "expected_output"
    threshold: 0.7
    strict_mode: false              # false=0-10評分, true=0/1評分
```

使用評分標準（Rubric）：

```yaml
metrics:
  - type: "g_eval"
    enabled: true
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
        description: "品質差 - 有錯誤或難以理解"
      - score_range: [4, 6]
        description: "尚可 - 基本正確但可改進"
      - score_range: [7, 8]
        description: "良好 - 正確且易讀"
      - score_range: [9, 10]
        description: "優秀 - 高品質且高效"
    threshold: 0.7
```

嚴格模式（二元評分）：

```yaml
metrics:
  - type: "g_eval"
    enabled: true
    name: "Contains Code"
    criteria: "檢查輸出是否包含可執行程式碼"
    evaluation_params:
      - "actual_output"
    strict_mode: true               # 只返回 0 或 1
    threshold: 1                    # 必須為 1 才算通過
```

### 測試案例配置

#### 選項 1: 從文件載入

```yaml
test_cases:
  file: "test_cases.jsonl"          # 或 test_cases.json
```

**JSONL 格式** (`test_cases.jsonl`):

```jsonl
{"input": "What is Python?", "actual_output": "Python is a programming language.", "expected_output": "A programming language"}
{"input": "What is 2+2?", "actual_output": "4", "expected_output": "4"}
```

**JSON 格式** (`test_cases.json`):

```json
[
  {
    "input": "What is Python?",
    "actual_output": "Python is a programming language.",
    "expected_output": "A programming language"
  },
  {
    "input": "What is machine learning?",
    "actual_output": "ML is a subset of AI.",
    "expected_output": "A subset of AI",
    "retrieval_context": [
      "Machine learning is part of artificial intelligence.",
      "It allows systems to learn from data."
    ],
    "metadata": {
      "category": "tech",
      "difficulty": "medium"
    }
  }
]
```

#### 選項 2: 直接在配置中定義

```yaml
test_cases:
  cases:
    - input: "What is the capital of France?"
      actual_output: "Paris is the capital of France."
      expected_output: "Paris"

    - input: "Explain quantum computing"
      actual_output: "Quantum computing uses qubits."
      expected_output: "Uses quantum mechanics"
      retrieval_context:
        - "Quantum computing is based on quantum mechanics."
        - "Qubits can exist in superposition."

    - input: "What is 2+2?"
      actual_output: "4"
      expected_output: "4"
      metadata:
        category: "math"
        difficulty: "easy"
```

### 輸出配置

```yaml
output:
  directory: "results"              # 結果輸出目錄
  save_summary: true                # 是否另存摘要文件
```

## 完整範例

### 範例 1: 問答系統評估

```yaml
llm_as_judge:
  judge_model:
    type: "openai"
    model_name: "gpt-4"
    api_key: "sk-..."
    base_url: "https://api.openai.com/v1"
    temperature: 0.0

  metrics:
    - type: "answer_relevancy"
      enabled: true
      threshold: 0.7

    - type: "g_eval"
      enabled: true
      name: "Correctness"
      criteria: "答案是否正確且完整"
      evaluation_params:
        - "input"
        - "actual_output"
        - "expected_output"
      threshold: 0.7

  test_cases:
    file: "qa_test_cases.jsonl"

  output:
    directory: "results"
    save_summary: true
```

### 範例 2: RAG 系統評估

```yaml
llm_as_judge:
  judge_model:
    type: "openai"
    model_name: "gpt-4"
    api_key: "sk-..."
    base_url: "https://api.openai.com/v1"

  metrics:
    - type: "faithfulness"
      enabled: true
      threshold: 0.8

    - type: "answer_relevancy"
      enabled: true
      threshold: 0.7

  test_cases:
    cases:
      - input: "What is the Eiffel Tower's height?"
        actual_output: "The Eiffel Tower is 330 meters tall."
        expected_output: "330 meters"
        retrieval_context:
          - "The Eiffel Tower stands at 330 meters (1,083 ft) tall."
          - "It was built in 1889 for the World's Fair."

  output:
    directory: "rag_results"
```

### 範例 3: 程式碼評估

```yaml
llm_as_judge:
  judge_model:
    type: "openai"
    model_name: "gpt-4"
    api_key: "sk-..."
    base_url: "https://api.openai.com/v1"

  metrics:
    - type: "g_eval"
      enabled: true
      name: "Code Quality"
      criteria: |
        評估程式碼品質：
        1. 正確性：功能是否正確
        2. 可讀性：程式碼是否清晰
        3. 效率：實現是否高效
      evaluation_params:
        - "input"
        - "actual_output"
      rubric:
        - score_range: [0, 3]
          description: "差 - 有錯誤"
        - score_range: [4, 6]
          description: "可 - 基本正確"
        - score_range: [7, 8]
          description: "好 - 正確清晰"
        - score_range: [9, 10]
          description: "優 - 高品質"
      threshold: 0.7

    - type: "g_eval"
      enabled: true
      name: "Has Tests"
      criteria: "檢查是否包含測試程式碼"
      evaluation_params:
        - "actual_output"
      strict_mode: true
      threshold: 1

  test_cases:
    file: "code_submissions.json"

  output:
    directory: "code_eval_results"
```

## 命令列選項

```bash
# 使用預設配置（config.yaml）
twinkle-eval-judge

# 使用自定義配置
twinkle-eval-judge --config my_config.yaml

# 創建範例配置
twinkle-eval-judge --init

# 顯示版本
twinkle-eval-judge --version

# 顯示幫助
twinkle-eval-judge --help
```

## Python API 使用

如果需要在 Python 程式中使用：

```python
from twinkle_eval.llm_as_judge import run_from_config

# 執行評估
results = run_from_config("my_config.yaml")

# 查看結果
print(f"Total test cases: {results['summary']['total_test_cases']}")

for metric_name, stats in results['summary']['metrics'].items():
    print(f"{metric_name}:")
    print(f"  Average: {stats['average_score']:.3f}")
    print(f"  Success Rate: {stats['success_rate']:.1%}")
```

或使用 Runner 類別：

```python
from twinkle_eval.llm_as_judge import LLMAsJudgeConfigRunner

runner = LLMAsJudgeConfigRunner("my_config.yaml")
runner.load_config()
runner.initialize_judge_model()
runner.initialize_metrics()
runner.load_test_cases()

results = runner.run_evaluation(use_async=True)
```

## 結果格式

評估結果會儲存為 JSON 格式：

```json
{
  "timestamp": "2025-01-17T10:30:00",
  "config": { ... },
  "test_results": [
    {
      "test_case": {
        "input": "What is Python?",
        "actual_output": "Python is a programming language."
      },
      "metric_results": [
        {
          "metric_name": "Answer Relevancy",
          "score": 0.9,
          "success": true,
          "reason": "The answer directly addresses the question..."
        }
      ]
    }
  ],
  "summary": {
    "total_test_cases": 10,
    "metrics": {
      "Answer Relevancy": {
        "average_score": 0.85,
        "min_score": 0.6,
        "max_score": 1.0,
        "total_evaluations": 10,
        "success_rate": 0.9
      }
    }
  }
}
```

## 常見問題

### Q: 如何使用本地模型作為 Judge？

A: 配置 `base_url` 指向本地服務：

```yaml
judge_model:
  type: "openai"
  model_name: "your-local-model"
  api_key: "EMPTY"
  base_url: "http://localhost:8000/v1"
  disable_ssl_verify: true
```

### Q: 如何只執行特定指標？

A: 設定其他指標的 `enabled: false`：

```yaml
metrics:
  - type: "answer_relevancy"
    enabled: true        # 執行

  - type: "faithfulness"
    enabled: false       # 不執行
```

### Q: 測試案例必須包含哪些欄位？

A: 只有 `input` 和 `actual_output` 是必須的，其他都是可選：

```yaml
test_cases:
  cases:
    - input: "必須"
      actual_output: "必須"
      expected_output: "可選"
      retrieval_context: ["可選"]
      metadata: {}  # 可選
```

### Q: 如何調整評估的嚴格程度？

A: 調整 `threshold`（閾值）：

```yaml
metrics:
  - type: "answer_relevancy"
    threshold: 0.9      # 更嚴格（需要 0.9 以上才算通過）
```

或使用 `strict_mode`（G-Eval）：

```yaml
metrics:
  - type: "g_eval"
    strict_mode: true   # 二元評分（0 或 1）
    threshold: 1        # 必須完全正確
```

## 最佳實踐

1. **先從小規模測試開始**
   - 先用幾個測試案例驗證配置
   - 確認 Judge 模型運作正常後再擴大規模

2. **合理設定閾值**
   - 根據實際需求調整 threshold
   - 不同指標可以有不同的標準

3. **使用有意義的指標名稱**
   - G-Eval 的 `name` 要清楚描述評估內容
   - 方便後續分析結果

4. **善用 metadata**
   - 為測試案例添加分類、難度等元數據
   - 方便後續分組分析

5. **異步評估提高效率**
   - 預設使用異步模式（更快）
   - 大量測試案例時效果更明顯

## 進階用法

查看完整文檔以了解：
- 如何實現自定義 Judge 模型
- 如何創建自定義評估指標
- API 詳細說明

文檔連結：[LLM-as-Judge README](README.md)
