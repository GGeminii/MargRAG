import json
import argparse
import asyncio
from pathlib import Path
from typing import Any

import yaml

from lightrag.utils import TokenTracker

from megarag.evaluation import (
    evaluate_global_pairwise,
    evaluate_local_correctness,
    save_json,
)
from megarag.llms.openai import openai_complete_if_cache


def load_eval_config(config_path: Path) -> dict:
    """
    功能说明：
        读取评估配置文件，兼容顶层结构与 evaluation 子结构。

    参数：
        - config_path (Path)：配置文件路径。

    返回：
        dict：评估配置字典。
    """
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    # 若配置中存在 evaluation 节点，则优先使用该节点
    return data.get("evaluation", data)


def parse_args() -> argparse.Namespace:
    """
    功能说明：
        解析命令行参数。

    返回：
        argparse.Namespace：解析后的参数对象。
    """
    parser = argparse.ArgumentParser(
        description="Run LLM-based evaluation for global/local QA experiments.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config-file",
        default="addon_params_evaluation.yaml",
        metavar="FILE",
        help="评估配置 YAML 文件路径。",
    )
    parser.add_argument(
        "--mode",
        choices=["global_pairwise", "local_correctness"],
        default=None,
        help="评估模式：global_pairwise 或 local_correctness。",
    )

    # 全局 pairwise 评估输入
    parser.add_argument(
        "--ours-result-file",
        default=None,
        metavar="FILE",
        help="我方结果文件（Answer 1）。",
    )
    parser.add_argument(
        "--baseline-result-file",
        default=None,
        metavar="FILE",
        help="对比结果文件（Answer 2）。",
    )

    # 本地正确性评估输入
    parser.add_argument(
        "--result-file",
        default=None,
        metavar="FILE",
        help="待评估模型结果文件（question + answer）。",
    )
    parser.add_argument(
        "--gt-file",
        default=None,
        metavar="FILE",
        help="本地 QA 标准答案文件。",
    )

    # 评估运行参数
    parser.add_argument("--eval-model", default=None, metavar="MODEL", help="评估器模型名。")
    parser.add_argument("--eval-max-tokens", type=int, default=None, metavar="N", help="评估输出最大 token。")
    parser.add_argument("--concurrency", type=int, default=None, metavar="N", help="评估并发数。")
    parser.add_argument("--max-retries", type=int, default=None, metavar="N", help="单条评估最大重试次数。")
    parser.add_argument("--retry-delay", type=float, default=None, metavar="SECONDS", help="重试基础间隔秒数。")

    parser.add_argument(
        "--output-file",
        default=None,
        metavar="FILE",
        help="评估结果输出 JSON 文件路径。",
    )

    return parser.parse_args()


async def async_main() -> None:
    """
    功能说明：
        异步主流程：读取配置 -> 构建评估 LLM -> 执行评估 -> 落盘 JSON。

    返回：
        None：通过副作用输出评估结果文件。
    """
    args = parse_args()
    cfg = load_eval_config(Path(args.config_file))

    # 命令行参数优先于配置文件参数
    mode = args.mode or cfg.get("mode", "global_pairwise")
    eval_model = args.eval_model or cfg.get("eval_model", "gpt-5.4-mini")
    eval_max_tokens = (
        args.eval_max_tokens
        if args.eval_max_tokens is not None
        else int(cfg.get("eval_max_tokens", 1200))
    )
    concurrency = (
        args.concurrency
        if args.concurrency is not None
        else int(cfg.get("concurrency", 4))
    )
    max_retries = (
        args.max_retries
        if args.max_retries is not None
        else int(cfg.get("max_retries", 3))
    )
    retry_delay = (
        args.retry_delay
        if args.retry_delay is not None
        else float(cfg.get("retry_delay", 1.0))
    )

    # 输出文件路径：优先命令行，其次按 mode 从配置取默认值
    if args.output_file:
        output_file = args.output_file
    else:
        if mode == "global_pairwise":
            output_file = cfg.get("global_output_file", "./evaluation/global_pairwise_eval.json")
        else:
            output_file = cfg.get("local_output_file", "./evaluation/local_correctness_eval.json")

    # Token 统计器：记录评估过程 token 使用量
    token_tracker = TokenTracker()

    async def eval_llm(prompt_text: str, **kwargs: Any) -> str:
        """
        功能说明：
            评估器 LLM 包装函数，统一模型、max_tokens 和 token 统计配置。

        参数：
            - prompt_text (str)：评估提示词文本。
            - kwargs (Any)：额外调用参数。

        返回：
            str：LLM 文本输出。
        """
        # 将 stream 固定为 False，确保返回完整字符串便于 JSON 解析
        kwargs.pop("stream", None)
        return await openai_complete_if_cache(
            model=eval_model,
            prompt=prompt_text,
            max_tokens=eval_max_tokens,
            stream=False,
            token_tracker=token_tracker,
            **kwargs,
        )

    # 根据模式执行不同评估方法
    if mode == "global_pairwise":
        ours_result_file = args.ours_result_file or cfg.get("ours_result_file")
        baseline_result_file = args.baseline_result_file or cfg.get("baseline_result_file")

        if not ours_result_file or not baseline_result_file:
            raise SystemExit("global_pairwise 模式需要提供 --ours-result-file 和 --baseline-result-file（或在配置中提供）。")

        result = await evaluate_global_pairwise(
            eval_llm,
            ours_result_file=ours_result_file,
            baseline_result_file=baseline_result_file,
            max_retries=max_retries,
            retry_delay=retry_delay,
            concurrency=concurrency,
        )
    else:
        result_file = args.result_file or cfg.get("local_result_file")
        gt_file = args.gt_file or cfg.get("local_gt_file")

        if not result_file or not gt_file:
            raise SystemExit("local_correctness 模式需要提供 --result-file 和 --gt-file（或在配置中提供）。")

        result = await evaluate_local_correctness(
            eval_llm,
            result_file=result_file,
            gt_file=gt_file,
            max_retries=max_retries,
            retry_delay=retry_delay,
            concurrency=concurrency,
        )

    # 添加运行元信息，便于复现实验
    result["runtime"] = {
        "mode": mode,
        "eval_model": eval_model,
        "eval_max_tokens": eval_max_tokens,
        "concurrency": concurrency,
        "max_retries": max_retries,
        "retry_delay": retry_delay,
        "token_usage": str(token_tracker),
    }

    # 输出评估结果 JSON
    save_json(output_file, result)
    print(f"Saved evaluation result to: {output_file}")
    print(f"Evaluation token usage: {token_tracker}")


def main() -> None:
    """
    功能说明：
        同步入口，执行异步评估主流程。
    """
    asyncio.run(async_main())


if __name__ == "__main__":
    main()

