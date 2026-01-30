import argparse
import copy
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

from twinkle_eval.exceptions import ConfigurationError

from .config import load_config
from .dataset import find_all_evaluation_files
from .evaluators import Evaluator
from .logger import log_error, log_info
from .results_exporters import ResultsExporterFactory


def convert_json_to_html(json_file_path: str) -> int:
    """將 JSON 結果檔案轉換為 HTML 格式

    Args:
        json_file_path: JSON 結果檔案的路徑

    Returns:
        int: 程式退出代碼（0 表示成功，1 表示失敗）
    """
    import json

    try:
        # 檢查輸入檔案是否存在
        if not os.path.exists(json_file_path):
            print(f"❌ 檔案不存在: {json_file_path}")
            return 1

        # 載入 JSON 結果
        with open(json_file_path, "r", encoding="utf-8") as f:
            results = json.load(f)

        # 建立 HTML 輸出器
        html_exporter = ResultsExporterFactory.create_exporter("html")

        # 產生輸出檔案路徑（與輸入檔案同目錄，但副檔名為 .html）
        output_path = os.path.splitext(json_file_path)[0] + ".html"

        # 執行轉換
        exported_file = html_exporter.export(results, output_path)

        print(f"✅ 成功轉換為 HTML: {exported_file}")
        return 0

    except json.JSONDecodeError as e:
        print(f"❌ JSON 檔案格式錯誤: {e}")
        return 1
    except Exception as e:
        print(f"❌ 轉換過程中發生錯誤: {e}")
        return 1


def create_default_config(output_path: str = "config.yaml") -> int:
    """創建預設配置檔案（選擇題評測和 LLM-as-Judge 兩種）

    Args:
        output_path: 輸出檔案路徑，預設為 config.yaml

    Returns:
        int: 程式退出代碼（0 表示成功，1 表示失敗）
    """
    import shutil

    # 定義要創建的配置檔案
    configs_to_create = [
        {
            "output": output_path,
            "template": "config.template.yaml",
            "description": "選擇題評測配置",
            "type": "multiple_choice",
        },
        {
            "output": "config_llm_as_judge.yaml",
            "template": "config_llm_as_judge.template.yaml",
            "description": "LLM-as-Judge 評估配置",
            "type": "llm_as_judge",
        },
    ]

    created_files = []

    try:
        for config_info in configs_to_create:
            target_path = config_info["output"]

            # 檢查檔案是否已存在
            if os.path.exists(target_path):
                response = input(f"⚠️  檔案 '{target_path}' 已存在，是否覆蓋？(y/N): ")
                if response.lower() not in ["y", "yes", "是"]:
                    print(f"   跳過 {target_path}")
                    continue

            # 找到範本檔案
            template_path = os.path.join(
                os.path.dirname(__file__), config_info["template"]
            )

            # LLM-as-Judge 範本可能在子目錄
            if not os.path.exists(template_path) and config_info["type"] == "llm_as_judge":
                template_path = os.path.join(
                    os.path.dirname(__file__),
                    "llm_as_judge",
                    "config_llm_as_judge.template.yaml",
                )

            if not os.path.exists(template_path):
                # 如果沒有範本檔案，使用內建內容創建
                if config_info["type"] == "llm_as_judge":
                    _create_llm_as_judge_config(target_path)
                    created_files.append((target_path, config_info["description"]))
                else:
                    print(f"❌ 找不到配置範本檔案: {template_path}")
                continue
            else:
                # 複製範本檔案
                shutil.copy2(template_path, target_path)
                created_files.append((target_path, config_info["description"]))

        if not created_files:
            print("❌ 沒有創建任何配置檔案")
            return 1

        print()
        print("✅ 配置檔案已創建：")
        for file_path, desc in created_files:
            print(f"   📄 {file_path} - {desc}")

        print()
        print("=" * 60)
        print("📝 選擇題評測 (TMMLU+, MMLU 等)")
        print("=" * 60)
        print("  1. 編輯 config.yaml，設定：")
        print("     - LLM API 設定 (llm_api.base_url, llm_api.api_key)")
        print("     - 模型名稱 (model.name)")
        print("     - 資料集路徑 (evaluation.dataset_paths)")
        print("  2. 執行：twinkle-eval --config config.yaml")

        print()
        print("=" * 60)
        print("📝 LLM-as-Judge 評估（答案相關性、忠實度等）")
        print("=" * 60)
        print("  1. 編輯 config_llm_as_judge.yaml，設定：")
        print("     - LLM API 設定 (llm_as_judge.llm_api)")
        print("     - 模型設定 (llm_as_judge.model)")
        print("     - 評估指標 (llm_as_judge.metrics)")
        print("     - 測試案例 (llm_as_judge.test_cases)")
        print("  2. 執行：twinkle-eval --config config_llm_as_judge.yaml")

        return 0

    except Exception as e:
        print(f"❌ 創建配置檔案時發生錯誤: {e}")
        return 1


def _create_llm_as_judge_config(target_path: str) -> None:
    """創建 LLM-as-Judge 配置檔案

    Args:
        target_path: 目標檔案路徑
    """
    content = '''# LLM-as-Judge 評估配置
# 使用此配置執行 LLM-as-Judge 評估，無需寫程式碼
# 執行方式：twinkle-eval --config config_llm_as_judge.yaml

llm_as_judge:
  # ============================================================================
  # LLM API 設定（與選擇題評測 config.yaml 格式一致）
  # ============================================================================
  llm_api:
    base_url: "https://api.openai.com/v1"             # API 伺服器網址
    api_key: "${OPENAI_API_KEY}"                      # API 金鑰（從 .env 讀取）
    disable_ssl_verify: false                         # 是否停用 SSL 驗證
    api_rate_limit: 5                                 # 每分鐘請求限制（Google 免費層: 5）
    max_retries: 5                                    # API 呼叫失敗時的重試次數
    timeout: 600                                      # API 呼叫的超時時間 (秒)

  # ============================================================================
  # 模型設定（與選擇題評測 config.yaml 格式一致）
  # ============================================================================
  model:
    name: "gpt-4o"                                    # 模型名稱
    type: "openai"                                    # 模型類型：openai, google
    temperature: 0.0                                  # 溫度參數（0.0 = 確定性）
    max_tokens: 4096                                  # 最大輸出 token 數
    supports_structured_output: false                 # 是否支援結構化輸出
    supports_json_mode: true                          # 是否支援 JSON 模式

  # ============================================================================
  # 評估指標配置
  # ============================================================================
  metrics:
    # 指標 1: 答案相關性
    - type: "answer_relevancy"
      enabled: true
      threshold: 0.7                                  # 成功閾值（0-1）

    # 指標 2: 忠實度（適用於 RAG 系統）
    - type: "faithfulness"
      enabled: false                                  # 設為 true 啟用
      threshold: 0.8

    # 指標 3: G-Eval 自定義評估
    # - type: "g_eval"
    #   enabled: true
    #   name: "Correctness"
    #   criteria: |
    #     評估實際輸出是否正確回答了輸入問題。
    #     考慮：
    #     1. 答案是否事實正確
    #     2. 答案是否完整
    #   evaluation_params:
    #     - "input"
    #     - "actual_output"
    #     - "expected_output"
    #   threshold: 0.7
    #   strict_mode: false                            # false=0-10評分, true=0/1評分

  # ============================================================================
  # 測試案例配置
  # ============================================================================
  test_cases:
    # 選項 1: 從檔案載入（推薦用於大量測試）
    file: "test_cases.jsonl"

    # 選項 2: 直接在此定義（會覆蓋上面的 file 選項）
    # cases:
    #   - input: "What is Python?"
    #     actual_output: "Python is a programming language."
    #     expected_output: "A programming language"
    #
    #   - input: "What is 2+2?"
    #     actual_output: "2+2 equals 4."
    #     expected_output: "4"

  # ============================================================================
  # 輸出配置
  # ============================================================================
  output:
    directory: "results/llm_as_judge"                 # 結果輸出目錄
    save_summary: true                                # 是否另存摘要文件

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
# 環境變數設定（.env 文件）
# ============================================================================
# GOOGLE_API_KEY=your_google_api_key      # 從 https://aistudio.google.com/apikey 取得
# OPENAI_API_KEY=your_openai_api_key      # 從 https://platform.openai.com 取得

# ============================================================================
# 模型配置範例
# ============================================================================
#
# 使用 OpenAI GPT（推薦）：
#   llm_api:
#     base_url: "https://api.openai.com/v1"
#     api_key: "${OPENAI_API_KEY}"
#   model:
#     name: "gpt-4o"
#     type: "openai"
#     supports_json_mode: true
#
# 使用 Google Gemini（免費層，每分鐘 5 請求）：
#   llm_api:
#     base_url: "https://generativelanguage.googleapis.com"
#     api_key: "${GOOGLE_API_KEY}"
#     api_rate_limit: 5
#   model:
#     name: "gemini-2.0-flash"
#     type: "google"
#     supports_json_mode: true
#
# 使用本地模型（vLLM/Ollama）：
#   llm_api:
#     base_url: "http://localhost:8000/v1"
#     api_key: "EMPTY"
#     disable_ssl_verify: true
#   model:
#     name: "meta-llama/Llama-3.2-3B-Instruct"
#     type: "openai"
'''
    with open(target_path, "w", encoding="utf-8") as f:
        f.write(content)


class TwinkleEvalRunner:
    """Twinkle Eval 主要執行器類別 - 負責控制整個評測流程"""

    def __init__(self, config_path: str = "config.yaml"):
        """初始化 Twinkle Eval 執行器

        Args:
            config_path: 配置檔案路徑，預設為 config.yaml
        """
        self.config_path = config_path  # 配置檔案路徑
        self.config = None  # 載入的配置字典
        self.start_time = None  # 執行開始時間標記
        self.start_datetime = None  # 執行開始的 datetime 物件
        self.results_dir = "results"  # 結果輸出目錄

    def initialize(self):
        """初始化評測執行器

        載入配置、設定時間標記、建立結果目錄

        Raises:
            Exception: 初始化過程中發生錯誤
        """
        try:
            self.config = load_config(self.config_path)  # 載入配置
            self.start_time = datetime.now().strftime("%Y%m%d_%H%M")  # 生成時間標記
            self.start_datetime = datetime.now()  # 記錄開始時間

            os.makedirs(self.results_dir, exist_ok=True)  # 建立結果目錄

            log_info(f"Twinkle Eval 初始化完成 - {self.start_time}")

        except Exception as e:
            log_error(f"初始化失敗: {e}")
            raise

    def _prepare_config_for_saving(self) -> Dict[str, Any]:
        """準備用於儲存的配置資料，移除敏感資訊

        在儲存配置到結果檔案前，需要移除 API 金鑰等敏感資訊
        和不可序列化的物件實例

        Returns:
            Dict[str, Any]: 清理後的配置字典
        """
        if self.config is None:
            raise ConfigurationError("配置未載入")

        # 移除物件實例（不可序列化）
        if "llm_instance" in self.config:
            del self.config["llm_instance"]

        save_config = copy.deepcopy(self.config)

        # 移除敏感資訊（API 金鑰）
        if "llm_api" in save_config and "api_key" in save_config["llm_api"]:
            del save_config["llm_api"]["api_key"]
        if "evaluation_strategy_instance" in save_config:
            del save_config["evaluation_strategy_instance"]

        return save_config

    def _get_dataset_paths(self) -> List[str]:
        """從配置中取得資料集路徑清單

        支援單一路徑字串或路徑清單，統一轉換為清單格式

        Returns:
            List[str]: 資料集路徑清單
        """
        if self.config is None:
            raise ConfigurationError("配置未載入")

        dataset_paths = self.config["evaluation"]["dataset_paths"]
        if isinstance(dataset_paths, str):
            dataset_paths = [dataset_paths]
        return dataset_paths

    def _evaluate_dataset(self, dataset_path: str, evaluator: Evaluator) -> Dict[str, Any]:
        """評測單一資料集

        對指定資料集中的所有檔案進行評測，支援多次執行並統計結果

        Args:
            dataset_path: 資料集路徑
            evaluator: 評測器實例

        Returns:
            Dict[str, Any]: 資料集評測結果，包含準確率統計和詳細結果
        """
        if self.config is None:
            raise ConfigurationError("配置未載入")

        log_info(f"開始評測資料集: {dataset_path}")

        all_files = find_all_evaluation_files(dataset_path)  # 尋找所有評測檔案
        repeat_runs = self.config["evaluation"].get("repeat_runs", 1)  # 重複執行次數
        prompt_map = self.config["evaluation"].get("datasets_prompt_map", {})  # 資料集語言對應表
        dataset_lang = prompt_map.get(dataset_path, "zh")  # 當前資料集的語言，預設為中文

        results = []  # 儲存所有檔案的評測結果

        for idx, file_path in enumerate(all_files):
            file_accuracies = []  # 當前檔案的準確率結果
            file_results = []  # 當前檔案的詳細結果

            # 對當前檔案進行多次評測
            for run in range(repeat_runs):
                try:
                    file_path_result, accuracy, result_path = evaluator.evaluate_file(
                        file_path, f"{self.start_time}_run{run}", dataset_lang
                    )
                    file_accuracies.append(accuracy)
                    file_results.append((file_path_result, accuracy, result_path))
                except Exception as e:
                    log_error(f"評測檔案 {file_path} 失敗: {e}")
                    continue

            # 為當前檔案計算統計數據
            if file_accuracies:
                mean_accuracy = np.mean(file_accuracies)  # 平均準確率
                std_accuracy = np.std(file_accuracies) if len(file_accuracies) > 1 else 0  # 標準差

                results.append(
                    {
                        "file": file_path,
                        "accuracy_mean": mean_accuracy,
                        "accuracy_std": std_accuracy,
                        "individual_runs": {
                            "accuracies": file_accuracies,
                            "results": [r[2] for r in file_results],
                        },
                    }
                )

            # 進度指示器
            progress = (idx + 1) / len(all_files) * 100
            print(f"\r已執行 {progress:.1f}% ({idx + 1}/{len(all_files)}) ", end="")

        print()  # 進度完成後換行

        # 計算資料集統計數據
        dataset_avg_accuracy = (
            np.mean([r["accuracy_mean"] for r in results]) if results else 0
        )  # 資料集平均準確率
        dataset_avg_std = (
            np.mean([r["accuracy_std"] for r in results]) if results else 0
        )  # 資料集平均標準差

        return {
            "results": results,
            "average_accuracy": dataset_avg_accuracy,
            "average_std": dataset_avg_std,
        }

    def run_evaluation(self, export_formats: Optional[List[str]] = None) -> str:
        """執行完整的評測流程

        這是主要的評測入口點，包含以下步驟：
        1. 建立評測器
        2. 對所有資料集進行評測
        3. 統計和輸出結果

        Args:
            export_formats: 輸出格式清單，預設為 ["json"]

        Returns:
            str: 主要結果檔案路徑
        """
        if self.config is None:
            raise ConfigurationError("配置未載入")

        if export_formats is None:
            export_formats = ["json"]  # 預設輸出格式

        dataset_paths = self._get_dataset_paths()  # 取得資料集路徑
        dataset_results = {}  # 儲存所有資料集的結果

        # 建立評測器
        llm_instance = self.config["llm_instance"]
        evaluation_strategy_instance = self.config["evaluation_strategy_instance"]
        evaluator = Evaluator(llm_instance, evaluation_strategy_instance, self.config)

        # 逐一評測每個資料集
        for dataset_path in dataset_paths:
            try:
                dataset_result = self._evaluate_dataset(dataset_path, evaluator)
                dataset_results[dataset_path] = dataset_result

                message = (
                    f"資料集 {dataset_path} 評測完成，"
                    f"平均正確率: {dataset_result['average_accuracy']:.2%} "
                    f"(±{dataset_result['average_std']:.2%})"
                )
                print(message)
                log_info(message)

            except Exception as e:
                log_error(f"資料集 {dataset_path} 評測失敗: {e}")
                continue

        # 準備最終結果
        current_duration = (
            (datetime.now() - self.start_datetime).total_seconds() if self.start_datetime else 0
        )  # 計算執行時間
        final_results = {
            "timestamp": self.start_time,  # 執行時間標記
            "config": self._prepare_config_for_saving(),  # 清理後的配置
            "dataset_results": dataset_results,  # 所有資料集結果
            "duration_seconds": current_duration,  # 執行時間（秒）
        }

        # 以多種格式輸出結果
        base_output_path = os.path.join(self.results_dir, f"results_{self.start_time}")
        exported_files = ResultsExporterFactory.export_results(
            final_results, base_output_path, export_formats, self.config
        )

        # Google 服務整合
        self._handle_google_services(final_results, export_formats)

        log_info(f"評測完成，結果已匯出至: {', '.join(exported_files)}")
        return exported_files[0] if exported_files else ""

    def _handle_google_services(self, results: Dict[str, Any], export_formats: List[str]):
        """處理 Google 服務整合

        Args:
            results: 評測結果字典
            export_formats: 匯出格式列表
        """
        google_services_config = self.config.get("google_services")
        if not google_services_config:
            return

        # 處理 Google Drive 檔案上傳（最新的 log 和 results）
        google_drive_config = google_services_config.get("google_drive", {})
        if google_drive_config.get("enabled", False):
            try:
                from .google_services import GoogleDriveUploader

                uploader = GoogleDriveUploader(google_drive_config)
                upload_info = uploader.upload_latest_files(self.start_time, "logs", "results")

                if upload_info.get("uploaded_files"):
                    log_info(
                        f"成功建立資料夾: {upload_info['folder_name']} ({upload_info['folder_id']})"
                    )
                    log_info(f"成功上傳 {len(upload_info['uploaded_files'])} 個檔案到 Google Drive")

                    for file_info in upload_info["uploaded_files"]:
                        log_info(f"  - {file_info['type']}: {file_info['file_name']}")
            except Exception as e:
                log_error(f"Google Drive 檔案上傳失敗: {e}")

        # 處理 Google Sheets 結果匯出
        google_sheets_config = google_services_config.get("google_sheets", {})
        if google_sheets_config.get("enabled", False):
            try:
                # 檢查是否已經在 export_formats 中指定 google_sheets
                if "google_sheets" not in export_formats:
                    # 如果用戶沒有明確指定，我們自動執行 Google Sheets 匯出
                    sheets_exporter = ResultsExporterFactory.create_exporter(
                        "google_sheets", google_sheets_config
                    )
                    sheets_url = sheets_exporter.export(results, "google_sheets_export")
                    log_info(f"結果已自動匯出到 Google Sheets: {sheets_url}")
            except Exception as e:
                log_error(f"Google Sheets 結果匯出失敗: {e}")


def create_cli_parser() -> argparse.ArgumentParser:
    """建立命令列介面解析器

    定義所有命令列參數和選項，支援多種評測和查詢功能

    Returns:
        argparse.ArgumentParser: 配置完成的命令列解析器
    """
    parser = argparse.ArgumentParser(
        description="🌟 Twinkle Eval - AI 模型評測工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用範例:
  twinkle-eval --init                   # 創建配置檔案（選擇題評測 + LLM-as-Judge）
  twinkle-eval                          # 使用預設配置執行選擇題評測
  twinkle-eval --config config.yaml    # 使用自定義配置檔（選擇題評測）
  twinkle-eval --config config_llm_as_judge.yaml  # 執行 LLM-as-Judge 評估
  twinkle-eval --export json csv html google_sheets  # 輸出為多種格式
  twinkle-eval --list-llms             # 列出可用的 LLM 類型
  twinkle-eval --list-strategies       # 列出可用的評測策略

LLM-as-Judge 評估:
  當配置檔案包含 llm_as_judge 區塊時，會自動切換到 LLM-as-Judge 評估模式
  支援的評估指標：answer_relevancy, faithfulness, g_eval

結果格式轉換:
  twinkle-eval --convert-to-html results_20240101_1200.json  # 將 JSON 結果轉換為 HTML

效能基準測試:
  twinkle-eval --benchmark                           # 執行預設的基準測試
  twinkle-eval --benchmark --benchmark-requests 50  # 執行 50 個請求的測試
  twinkle-eval --benchmark --benchmark-concurrency 5 --benchmark-rate 2  # 5 並發，2 請求/秒

HuggingFace 資料集下載:
  twinkle-eval --download-dataset cais/mmlu          # 下載 MMLU 所有子集
  twinkle-eval --download-dataset cais/mmlu --dataset-subset anatomy  # 下載特定子集
  twinkle-eval --dataset-info cais/mmlu             # 查看資料集資訊
        """,
    )

    parser.add_argument(
        "--config", "-c", default="config.yaml", help="配置檔案路徑 (預設: config.yaml)，支援選擇題評測和 LLM-as-Judge 兩種模式"
    )

    parser.add_argument(
        "--export",
        "-e",
        nargs="+",
        default=["json"],
        choices=ResultsExporterFactory.get_available_types(),
        help="輸出格式 (預設: json)",
    )

    parser.add_argument("--list-llms", action="store_true", help="列出可用的 LLM 類型")

    parser.add_argument("--list-strategies", action="store_true", help="列出可用的評測策略")

    parser.add_argument("--list-exporters", action="store_true", help="列出可用的輸出格式")

    parser.add_argument("--version", action="store_true", help="顯示版本資訊")

    parser.add_argument("--init", action="store_true", help="創建預設配置檔案（選擇題評測 + LLM-as-Judge）")

    # HuggingFace 資料集下載相關命令
    parser.add_argument(
        "--download-dataset",
        metavar="DATASET_NAME",
        help="從 HuggingFace Hub 下載資料集 (例如: cais/mmlu)",
    )

    parser.add_argument(
        "--dataset-subset",
        metavar="SUBSET",
        help="指定資料集子集名稱 (與 --download-dataset 一起使用)",
    )

    parser.add_argument(
        "--dataset-split",
        metavar="SPLIT",
        default="test",
        help="指定資料集分割 (預設: test)",
    )

    parser.add_argument(
        "--output-dir",
        metavar="DIR",
        default="datasets",
        help="資料集下載輸出目錄 (預設: datasets)",
    )

    parser.add_argument(
        "--dataset-info",
        metavar="DATASET_NAME",
        help="獲取 HuggingFace 資料集資訊",
    )

    parser.add_argument(
        "--convert-to-html",
        metavar="JSON_FILE",
        help="將 JSON 結果檔案轉換為 HTML 格式",
    )

    # Benchmark 相關命令
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="執行 LLM 效能基準測試",
    )

    parser.add_argument(
        "--benchmark-prompt",
        metavar="PROMPT",
        default="請用繁體中文回答：台灣的首都是哪裡？",
        help="基準測試使用的提示文字 (預設: 請用繁體中文回答：台灣的首都是哪裡？)",
    )

    parser.add_argument(
        "--benchmark-requests",
        type=int,
        default=100,
        help="基準測試的總請求數 (預設: 100)",
    )

    parser.add_argument(
        "--benchmark-concurrency",
        type=int,
        default=10,
        help="基準測試的並發請求數 (預設: 10)",
    )

    parser.add_argument(
        "--benchmark-rate",
        type=float,
        help="基準測試的請求速率 (請求/秒，不指定則全速發送)",
    )

    parser.add_argument(
        "--benchmark-duration",
        type=float,
        help="基準測試的最大執行時間 (秒，不指定則執行完所有請求)",
    )

    return parser


def main() -> int:
    """主程式入口點

    處理命令列參數並執行相應的功能，包括查詢功能和主要評測流程

    Returns:
        int: 程式退出代碼（0 表示成功，1 表示失敗）
    """
    parser = create_cli_parser()
    args = parser.parse_args()

    # 處理查詢命令
    if args.list_llms:
        from .models import LLMFactory

        print("可用的 LLM 類型:")
        for llm_type in LLMFactory.get_available_types():
            print(f"  - {llm_type}")
        return 0

    if args.list_strategies:
        from .evaluation_strategies import EvaluationStrategyFactory

        print("可用的評測策略:")
        for strategy in EvaluationStrategyFactory.get_available_types():
            print(f"  - {strategy}")
        return 0

    if args.list_exporters:
        print("可用的輸出格式:")
        for exporter in ResultsExporterFactory.get_available_types():
            print(f"  - {exporter}")
        return 0

    if args.version:
        from . import get_info

        info = get_info()
        print(f"🌟 {info['name']} v{info['version']}")
        print(f"作者: {info['author']}")
        print(f"授權: {info['license']}")
        print(f"網址: {info['url']}")
        return 0

    if args.init:
        return create_default_config()

    # HuggingFace 資料集相關命令
    if args.download_dataset:
        try:
            from .dataset import download_huggingface_dataset

            download_huggingface_dataset(
                dataset_name=args.download_dataset,
                subset=args.dataset_subset,
                split=args.dataset_split,
                output_dir=args.output_dir,
            )
            print(f"✅ 資料集下載完成，已快取到 HuggingFace 目錄")
            return 0
        except Exception as e:
            print(f"❌ 下載資料集失敗: {e}")
            return 1

    if args.dataset_info:
        try:
            from .dataset import list_huggingface_dataset_info

            info = list_huggingface_dataset_info(
                dataset_name=args.dataset_info, subset=args.dataset_subset
            )
            print(f"📊 資料集資訊: {info['dataset_name']}")
            print(f"可用配置: {', '.join(info['configs'])}")
            for config, splits in info["splits"].items():
                print(f"  {config}: {', '.join(splits)}")
            return 0
        except Exception as e:
            print(f"❌ 獲取資料集資訊失敗: {e}")
            return 1

    # JSON 轉 HTML 命令
    if args.convert_to_html:
        try:
            return convert_json_to_html(args.convert_to_html)
        except Exception as e:
            print(f"❌ 轉換失敗: {e}")
            return 1

    # Benchmark 命令
    if args.benchmark:
        try:
            from .benchmark import BenchmarkRunner, print_benchmark_summary, save_benchmark_results
            from .config import load_config

            config = load_config(args.config)
            runner = BenchmarkRunner(config)

            print(f"🚀 開始執行 LLM 效能基準測試")
            print(f"   提示文字: {args.benchmark_prompt}")
            print(f"   請求數量: {args.benchmark_requests}")
            print(f"   並發數量: {args.benchmark_concurrency}")
            if args.benchmark_rate:
                print(f"   請求速率: {args.benchmark_rate} 請求/秒")
            if args.benchmark_duration:
                print(f"   最大時間: {args.benchmark_duration} 秒")
            print("-" * 60)

            metrics = runner.run_benchmark(
                prompt=args.benchmark_prompt,
                num_requests=args.benchmark_requests,
                concurrent_requests=args.benchmark_concurrency,
                request_rate=args.benchmark_rate,
                duration=args.benchmark_duration,
            )

            # 顯示結果摘要
            print_benchmark_summary(metrics)

            # 儲存結果
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_path = f"benchmark_results_{timestamp}.json"
            if "llm_instance" in config:
                del config["llm_instance"]
            if "evaluation_strategy_instance" in config:
                del config["evaluation_strategy_instance"]
            save_benchmark_results(metrics, output_path, config)

            return 0

        except Exception as e:
            print(f"❌ 基準測試失敗: {e}")
            log_error(f"基準測試執行錯誤: {e}")
            return 1

    # 檢查配置檔案是否為 LLM-as-Judge 模式
    try:
        if _is_llm_as_judge_config(args.config):
            return _run_llm_as_judge(args.config)
    except Exception as e:
        log_error(f"檢查配置檔案失敗: {e}")
        return 1

    # 執行選擇題評測
    try:
        runner = TwinkleEvalRunner(args.config)
        runner.initialize()
        runner.run_evaluation(args.export)
    except Exception as e:
        log_error(f"執行失敗: {e}")
        return 1

    return 0


def _is_llm_as_judge_config(config_path: str) -> bool:
    """檢查配置檔案是否為 LLM-as-Judge 模式

    Args:
        config_path: 配置檔案路徑

    Returns:
        bool: 如果配置檔案包含 llm_as_judge 區塊則返回 True
    """
    import yaml

    if not os.path.exists(config_path):
        return False

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config is not None and "llm_as_judge" in config


def _run_llm_as_judge(config_path: str) -> int:
    """執行 LLM-as-Judge 評估

    Args:
        config_path: 配置檔案路徑

    Returns:
        int: 程式退出代碼（0 表示成功，1 表示失敗）
    """
    try:
        from .llm_as_judge.config_runner import run_from_config

        print("🔍 偵測到 LLM-as-Judge 配置，啟動 LLM-as-Judge 評估模式...")
        print(f"   配置檔案: {config_path}")
        print()

        results = run_from_config(config_path)

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
        print(f"❌ 錯誤: {e}", file=sys.stderr)
        print("\n💡 提示: 使用 --init 創建範例配置檔案", file=sys.stderr)
        return 1

    except ValueError as e:
        print(f"❌ 配置錯誤: {e}", file=sys.stderr)
        return 1

    except KeyboardInterrupt:
        print("\n⚠️  使用者中斷執行", file=sys.stderr)
        return 130

    except Exception as e:
        print(f"❌ 錯誤: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())