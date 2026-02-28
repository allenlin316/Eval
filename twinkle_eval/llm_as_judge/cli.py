"""
Command-line interface for LLM-as-Judge evaluations.
"""

import argparse
import sys

from .config_runner import run_from_config
from .rag_runner import run_rag_evaluation


def main():
    """Main entry point for the LLM-as-Judge CLI."""
    parser = argparse.ArgumentParser(
        description="LLM-as-Judge: Evaluate LLM outputs using LLMs as judges",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run evaluation with default config (config.yaml)
  twinkle-eval-judge

  # Run with custom config
  twinkle-eval-judge --config my_config.yaml

  # Run RAG evaluation (retrieval + generation + LLM-as-Judge)
  twinkle-eval-judge --rag-config config_rag.yaml

  # Create example config files
  twinkle-eval-judge --init
  twinkle-eval-judge --init-rag

For more information, see:
  https://github.com/ai-twinkle/Eval/blob/main/twinkle_eval/llm_as_judge/README.md
        """,
    )

    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="config.yaml",
        help="Path to configuration file (default: config.yaml)",
    )

    parser.add_argument(
        "--rag-config",
        type=str,
        help="Path to RAG evaluation config file (enables RAG evaluation mode)",
    )

    parser.add_argument(
        "--init",
        action="store_true",
        help="Create example configuration file (config_llm_as_judge.yaml)",
    )

    parser.add_argument(
        "--init-rag",
        action="store_true",
        help="Create example RAG evaluation config file (config_rag.yaml)",
    )

    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 1.0.0 (Twinkle Eval LLM-as-Judge)",
    )

    args = parser.parse_args()

    # Handle --init
    if args.init:
        create_example_config()
        return 0

    # Handle --init-rag
    if args.init_rag:
        create_rag_config()
        return 0

    # Run RAG evaluation if --rag-config is specified
    if args.rag_config:
        return run_rag_evaluation_cli(args.rag_config)

    # Run standard LLM-as-Judge evaluation
    try:
        print("Starting LLM-as-Judge evaluation...")
        print(f"Configuration: {args.config}\n")

        results = run_from_config(args.config)

        # Print summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)

        summary = results.get("summary", {})
        print(f"\nTotal test cases: {summary.get('total_test_cases', 0)}")

        print("\nMetric Results:")
        for metric_name, stats in summary.get("metrics", {}).items():
            print(f"\n  {metric_name}:")
            print(f"    Average Score: {stats['average_score']:.3f}")
            print(f"    Success Rate:  {stats['success_rate']:.1%}")
            print(f"    Min/Max:       {stats['min_score']:.3f} / {stats['max_score']:.3f}")

        print("\n" + "=" * 60)

        return 0

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        print("\nTip: Use --init to create an example configuration file", file=sys.stderr)
        return 1

    except ValueError as e:
        print(f"Configuration Error: {e}", file=sys.stderr)
        return 1

    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user", file=sys.stderr)
        return 130

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


def run_rag_evaluation_cli(config_path: str) -> int:
    """Run RAG evaluation from CLI."""
    try:
        print("Starting RAG Evaluation...")
        print(f"Configuration: {config_path}\n")

        results = run_rag_evaluation(config_path)

        return 0

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        print("\nTip: Use --init-rag to create an example RAG configuration file", file=sys.stderr)
        return 1

    except ValueError as e:
        print(f"Configuration Error: {e}", file=sys.stderr)
        return 1

    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user", file=sys.stderr)
        return 130

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


def create_example_config():
    """Create example configuration file for LLM-as-Judge."""
    import os

    target = "config_llm_as_judge.yaml"

    if os.path.exists(target):
        response = input(f"{target} already exists. Overwrite? [y/N] ")
        if response.lower() != 'y':
            print("Cancelled.")
            return

    # Create the config file
    create_minimal_config(target)

    print(f"✓ Created LLM-as-Judge configuration: {target}")
    print("\nNext steps:")
    print("1. Edit config_llm_as_judge.yaml and set your API key")
    print("2. Prepare your test cases (create test_cases.jsonl or define inline)")
    print("3. Run: twinkle-eval-judge --config config_llm_as_judge.yaml")
    print("\nFor general evaluation (TMMLU+, MMLU), use:")
    print("  twinkle-eval --init         # Create config.yaml")
    print("  twinkle-eval --config config.yaml")


def create_minimal_config(target: str):
    """Create a full-featured LLM-as-Judge config file."""
    content = """# LLM-as-Judge 評估配置
# 使用此配置執行 LLM-as-Judge 評估，無需寫程式碼

llm_as_judge:
  # Judge 模型配置
  judge_model:
    type: "openai"                                    # 模型類型（目前支援 "openai"）
    model_name: "gpt-4"                               # 模型名稱
    api_key: "your-api-key-here"                      # 請替換為您的 API 金鑰
    base_url: "https://api.openai.com/v1"             # API 端點

    # 可選配置
    temperature: 0.0                                   # 溫度參數（0.0 = 確定性）
    max_tokens: 4096                                   # 最大輸出 token 數
    timeout: 600                                       # 請求超時時間（秒）
    max_retries: 3                                     # 失敗重試次數
    disable_ssl_verify: false                          # 是否停用 SSL 驗證
    supports_structured_output: false                  # 是否支援結構化輸出
    supports_json_mode: true                           # 是否支援 JSON 模式

  # 評估指標配置
  metrics:
    # 指標 1: 答案相關性
    - type: "answer_relevancy"
      enabled: true
      threshold: 0.7                                   # 成功閾值（0-1）

    # 指標 2: 忠實度（適用於 RAG 系統）
    - type: "faithfulness"
      enabled: false                                   # 設為 true 啟用
      threshold: 0.8

    # 指標 3: G-Eval 自定義評估
    - type: "g_eval"
      enabled: true
      name: "Correctness"
      criteria: |
        評估實際輸出是否正確回答了輸入問題。
        考慮：
        1. 答案是否事實正確
        2. 答案是否完整
      evaluation_params:
        - "input"
        - "actual_output"
        - "expected_output"
      threshold: 0.7
      strict_mode: false                               # false=0-10評分, true=0/1評分

  # 測試案例配置
  test_cases:
    # 選項 1: 資料集路徑（推薦，類似一般評測）
    dataset_paths:
      - "test_cases"                    # 資料夾或檔案路徑

    # 選項 2: 單一檔案
    # file: "test_cases.jsonl"

    # 選項 3: 直接定義
    # cases:
    #   - input: "What is Python?"
    #     actual_output: "Python is a programming language."

  # 輸出配置
  output:
    directory: "results/llm_as_judge"                  # 結果輸出目錄
    save_summary: true                                 # 是否另存摘要文件

# ============================================================================
# 測試案例檔案格式（test_cases.jsonl）
# ============================================================================
# 每行一個 JSON 物件：
# {"input": "What is Python?", "actual_output": "Python is a programming language.", "expected_output": "A language"}
# {"input": "What is 2+2?", "actual_output": "4", "expected_output": "4"}
#
# 或使用 JSON 格式（test_cases.json）：
# [
#   {
#     "input": "What is Python?",
#     "actual_output": "Python is a programming language.",
#     "expected_output": "A language",
#     "retrieval_context": ["Python is a high-level language."],
#     "metadata": {"category": "tech"}
#   }
# ]

# ============================================================================
# 使用方式
# ============================================================================
# 1. 設定 judge_model.api_key（必須）
# 2. 準備測試案例（創建 test_cases.jsonl 或在上面直接定義）
# 3. 執行：twinkle-eval-judge --config config_llm_as_judge.yaml
#
# 使用本地模型（vLLM/Ollama）範例：
#   judge_model:
#     model_name: "meta-llama/Llama-3.2-3B-Instruct"
#     api_key: "EMPTY"
#     base_url: "http://localhost:8000/v1"
#     disable_ssl_verify: true
"""

    with open(target, "w", encoding="utf-8") as f:
        f.write(content)


def create_rag_config():
    """Create example RAG evaluation configuration file."""
    import os

    target = "config_rag.yaml"

    if os.path.exists(target):
        response = input(f"{target} already exists. Overwrite? [y/N] ")
        if response.lower() != 'y':
            print("Cancelled.")
            return

    content = '''# RAG Evaluation Configuration
# This config supports two evaluation stages:
# 1. Retrieval Evaluation (Recall@k, nDCG@k, MRR) - No LLM needed
# 2. Generation + LLM-as-Judge (Faithfulness, Answer Relevance, Accuracy)
#
# Usage: twinkle-eval-judge --rag-config config_rag.yaml

rag_evaluation:
  # ============================================================================
  # Data Configuration
  # ============================================================================
  data:
    # RAG data file (JSONL format with contexts, input, targets)
    rag_file: "dataset/clapnq.jsonl"
    
    # Relevance judgments file (TSV: query-id, corpus-id, score)
    qrels_file: "dataset/qrels/clapnq_qrels.tsv"

  # ============================================================================
  # Retrieval Metrics (No LLM Required)
  # ============================================================================
  retrieval_metrics:
    enabled: true
    k_values: [1, 3, 5, 10]

  # ============================================================================
  # Answer Generation (LLM Required)
  # ============================================================================
  generation:
    enabled: true
    top_k: 5
    prompt_template: |
      Based on the following retrieved documents, answer the question.
      
      Documents:
      {context}
      
      Question: {question}
      
      Answer:

  # ============================================================================
  # Generator Model
  # ============================================================================
  generator:
    llm_api:
      base_url: "https://api.openai.com/v1"
      api_key: "${OPENAI_API_KEY}"
    model:
      name: "gpt-4"
      type: "openai"
      temperature: 0.0
      max_tokens: 1024

  # ============================================================================
  # Judge Model
  # ============================================================================
  judge:
    llm_api:
      base_url: "https://api.openai.com/v1"
      api_key: "${OPENAI_API_KEY}"
    model:
      name: "gpt-4"
      type: "openai"
      temperature: 0.0
      max_tokens: 4096
      supports_json_mode: true

  # ============================================================================
  # LLM-as-Judge Metrics
  # ============================================================================
  llm_judge_metrics:
    enabled: true
    faithfulness:
      enabled: true
      threshold: 0.8
    answer_relevancy:
      enabled: true
      threshold: 0.7
    accuracy:
      enabled: true
      threshold: 0.7

  # ============================================================================
  # Output Configuration
  # ============================================================================
  output:
    directory: "results/rag_evaluation"
    save_summary: true
    save_per_query_retrieval: false
'''

    with open(target, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"✓ Created RAG evaluation configuration: {target}")
    print("\nNext steps:")
    print("1. Edit config_rag.yaml and set your API key")
    print("2. Prepare your RAG data (JSONL) and qrels (TSV)")
    print("3. Run: twinkle-eval-judge --rag-config config_rag.yaml")


if __name__ == "__main__":
    sys.exit(main())
