# LLM-as-Judge Module

這個模組提供了使用大型語言模型（LLM）作為評判者來評估 LLM 輸出品質的框架。設計靈感來自 [deepeval](https://github.com/confident-ai/deepeval)，但實現為獨立模組，讓使用者可以自定義 judge model。

## 功能特色

- **靈活的評估指標**: 支援多種預建指標和自定義評估標準
- **可自定義 Judge 模型**: 輕鬆切換不同的 LLM 作為評判者
- **多種評估策略**: G-Eval、Faithfulness、Answer Relevancy 等
- **結構化輸出**: 使用 Pydantic schemas 確保評估結果的一致性
- **異步支持**: 所有指標都支持同步和異步評估
- **模板驅動**: 提示詞模板化，方便自定義和擴展

## 支援的評估指標

### 1. G-Eval
G-Eval 是一個靈活的框架，允許您使用自定義標準評估 LLM 輸出。

**特點**:
- 自動生成評估步驟
- 支持自定義評分標準（Rubric）
- 可選的嚴格模式（二元評分）

**適用場景**:
- 自定義評估標準
- 特定領域的質量評估
- 需要詳細評分標準的場景

### 2. Faithfulness
Faithfulness 評估 LLM 輸出是否與提供的檢索上下文（retrieval context）在事實上一致。

**特點**:
- 從上下文中提取真相（truths）
- 從輸出中提取聲明（claims）
- 逐一驗證聲明是否被上下文支持

**適用場景**:
- RAG（Retrieval-Augmented Generation）系統
- 需要確保事實準確性的應用
- 檢測幻覺（hallucination）

### 3. Answer Relevancy
Answer Relevancy 評估 LLM 輸出是否切題、是否有效回答輸入問題。

**特點**:
- 評估回答與問題的相關性
- 檢測離題或無關信息
- 確保回答的實用性

**適用場景**:
- 問答系統
- 聊天機器人
- 任何需要確保回答切題的場景

## 快速開始

### 安裝

模組已包含在 `twinkle-eval` 套件中：

```bash
pip install twinkle-eval
```

### 基本使用

```python
from twinkle_eval.llm_as_judge import (
    OpenAIJudgeModel,
    GEval,
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    TestCase,
)

# 1. 創建 Judge 模型
judge = OpenAIJudgeModel(
    model_name="gpt-4",
    api_key="your-api-key",
    base_url="https://api.openai.com/v1",
    temperature=0.0,
)

# 2. 創建測試案例
test_case = TestCase(
    input="What is the capital of France?",
    actual_output="Paris is the capital of France. It's a beautiful city known for the Eiffel Tower.",
    expected_output="Paris",
    retrieval_context=[
        "France is a country in Western Europe.",
        "The capital city of France is Paris.",
        "Paris is known for landmarks like the Eiffel Tower.",
    ],
)

# 3. 使用不同的評估指標

# Answer Relevancy
relevancy_metric = AnswerRelevancyMetric(threshold=0.7)
result = relevancy_metric.measure(test_case, judge)
print(f"Relevancy Score: {result.score:.2f}")
print(f"Success: {result.success}")
print(f"Reason: {result.reason}")

# Faithfulness
faithfulness_metric = FaithfulnessMetric(threshold=0.8)
result = faithfulness_metric.measure(test_case, judge)
print(f"Faithfulness Score: {result.score:.2f}")
print(f"Reason: {result.reason}")

# G-Eval with custom criteria
correctness_metric = GEval(
    name="Correctness",
    criteria="Determine whether the actual output is factually correct based on the expected output",
    evaluation_params=["input", "actual_output", "expected_output"],
    threshold=0.7,
)
result = correctness_metric.measure(test_case, judge)
print(f"Correctness Score: {result.score:.2f}")
print(f"Reason: {result.reason}")
```

### 使用自定義 Judge 模型

您可以透過繼承 `BaseJudgeModel` 來實現自己的 Judge 模型：

```python
from twinkle_eval.llm_as_judge.models import BaseJudgeModel
from typing import Optional, Type
from pydantic import BaseModel

class CustomJudgeModel(BaseJudgeModel):
    def __init__(self, **kwargs):
        super().__init__(model_name="custom-model", **kwargs)
        # 初始化您的模型

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        schema: Optional[Type[BaseModel]] = None,
    ) -> str:
        # 實現同步生成
        # 調用您的模型 API
        pass

    async def a_generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        schema: Optional[Type[BaseModel]] = None,
    ) -> str:
        # 實現異步生成
        pass

# 使用自定義模型
custom_judge = CustomJudgeModel()
metric = AnswerRelevancyMetric(threshold=0.7, judge_model=custom_judge)
result = metric.measure(test_case)
```

### G-Eval 進階用法

#### 使用 Rubric (評分標準)

```python
from twinkle_eval.llm_as_judge.schemas import Rubric

metric = GEval(
    name="Response Quality",
    criteria="Evaluate the overall quality of the response",
    evaluation_params=["input", "actual_output"],
    rubric=[
        Rubric(score_range=(0, 3), description="Poor quality - incorrect or unhelpful"),
        Rubric(score_range=(4, 6), description="Fair quality - partially correct"),
        Rubric(score_range=(7, 8), description="Good quality - mostly correct and helpful"),
        Rubric(score_range=(9, 10), description="Excellent quality - accurate and comprehensive"),
    ],
    threshold=0.7,
)
```

#### 嚴格模式（二元評分）

```python
metric = GEval(
    name="Has Code",
    criteria="Determine if the output contains executable code",
    evaluation_params=["actual_output"],
    strict_mode=True,  # 只返回 0 或 1
    threshold=1,  # 嚴格模式下自動設為 1
)
```

### 異步評估

所有指標都支持異步評估，適合大量測試案例：

```python
import asyncio

async def evaluate_multiple():
    test_cases = [...]  # 多個測試案例
    metric = AnswerRelevancyMetric(threshold=0.7, judge_model=judge)

    tasks = [metric.a_measure(tc) for tc in test_cases]
    results = await asyncio.gather(*tasks)

    for result in results:
        print(f"Score: {result.score}, Success: {result.success}")

asyncio.run(evaluate_multiple())
```

## 架構設計

### 核心組件

```
llm_as_judge/
├── __init__.py                 # 模組入口
├── schemas.py                  # Pydantic 資料結構
├── utils.py                    # 工具函數
├── models/                     # Judge 模型實現
│   ├── base_judge.py          # 抽象基類
│   └── openai_judge.py        # OpenAI 相容實現
├── metrics/                    # 評估指標
│   ├── base_metric.py         # 指標基類
│   ├── g_eval.py              # G-Eval 實現
│   ├── faithfulness.py        # Faithfulness 實現
│   └── answer_relevancy.py    # Answer Relevancy 實現
└── templates/                  # 提示詞模板
    ├── g_eval_template.py
    ├── faithfulness_template.py
    └── answer_relevancy_template.py
```

### 設計模式

1. **策略模式**: 不同的評估指標可互換使用
2. **模板模式**: 所有指標遵循相同的評估流程
3. **工廠模式**: 靈活創建不同的 Judge 模型

## 擴展指南

### 創建自定義評估指標

```python
from twinkle_eval.llm_as_judge.metrics import BaseMetric
from twinkle_eval.llm_as_judge.schemas import MetricResult, TestCase
from typing import Optional

class CustomMetric(BaseMetric):
    def __init__(self, threshold: float = 0.5, judge_model=None):
        super().__init__("CustomMetric", threshold, judge_model)

    def measure(self, test_case: TestCase, judge_model=None) -> MetricResult:
        model = self._get_judge_model(judge_model)

        # 1. 構建評估提示詞
        prompt = f"Evaluate: {test_case.actual_output}"

        # 2. 調用 judge model
        response = model.generate(prompt)

        # 3. 解析結果
        score = ...  # 從回應中提取分數
        reason = ...  # 從回應中提取理由

        # 4. 返回結果
        return self._create_result(score, reason)

    async def a_measure(self, test_case: TestCase, judge_model=None) -> MetricResult:
        # 異步版本
        model = self._get_judge_model(judge_model)
        response = await model.a_generate(prompt)
        # ... 同上
```

## 最佳實踐

1. **選擇合適的 Judge 模型**:
   - 使用強大的模型（如 GPT-4）作為 Judge 以獲得更準確的評估
   - 考慮成本和速度的平衡

2. **設定合理的閾值**:
   - 根據具體應用場景調整 threshold
   - 嚴格場景使用較高閾值（0.8-0.9）
   - 一般場景使用中等閾值（0.6-0.7）

3. **組合多個指標**:
   - 不同指標評估不同方面
   - 綜合多個指標的結果做決策

4. **使用異步評估**:
   - 大量評估時使用 `a_measure()` 提高效率
   - 合理控制並發數避免 API 限流

## 參考文獻

- [G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment](https://arxiv.org/pdf/2303.16634.pdf)
- [DeepEval Framework](https://github.com/confident-ai/deepeval)

## 授權

本模組採用與 Twinkle Eval 相同的 MIT 授權條款。
