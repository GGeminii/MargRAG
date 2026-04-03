import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import matplotlib.pyplot as plt
    _HAS_MATPLOTLIB = True
except Exception:
    plt = None
    _HAS_MATPLOTLIB = False


def load_json(path: str | Path) -> dict:
    """
    功能说明：
        读取 JSON 文件并返回字典对象。

    参数：
        - path (str | Path)：文件路径。

    返回：
        dict：解析后的 JSON 数据。
    """
    file_path = Path(path)
    with file_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str | Path, data: dict) -> None:
    """
    功能说明：
        将字典写入 JSON 文件。

    参数：
        - path (str | Path)：输出路径。
        - data (dict)：待保存数据。

    返回：
        None：通过副作用完成写文件。
    """
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    """
    功能说明：
        解析命令行参数。

    返回：
        argparse.Namespace：参数对象。
    """
    parser = argparse.ArgumentParser(
        description="Visualize evaluation results for global/local QA.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--global-eval-file",
        default=None,
        metavar="FILE",
        help="全局 pairwise 评估结果 JSON 文件。",
    )
    parser.add_argument(
        "--local-eval-file",
        default=None,
        metavar="FILE",
        help="本地 correctness 评估结果 JSON 文件。",
    )
    parser.add_argument(
        "--output-dir",
        default="./evaluation/figures",
        metavar="DIR",
        help="图表输出目录。",
    )
    parser.add_argument(
        "--summary-file",
        default="./evaluation/figures/eval_visual_summary.json",
        metavar="FILE",
        help="可视化统计摘要输出 JSON 文件。",
    )
    return parser.parse_args()


def _plot_global_winner_counts(global_eval: dict, output_dir: Path) -> Tuple[str, dict]:
    """
    功能说明：
        绘制全局评估的按维度赢家计数柱状图。

    参数：
        - global_eval (dict)：global_pairwise 评估结果。
        - output_dir (Path)：输出目录。

    返回：
        Tuple[str, dict]：图像路径 + 图中使用的统计数据。
    """
    winner_counts = global_eval.get("summary", {}).get("winner_counts", {})
    criteria = list(winner_counts.keys())

    # 按维度拆分三种计数，便于分组柱状图展示
    answer1_vals = [winner_counts[c].get("Answer 1", 0) for c in criteria]
    answer2_vals = [winner_counts[c].get("Answer 2", 0) for c in criteria]
    unknown_vals = [winner_counts[c].get("Unknown", 0) for c in criteria]

    x = list(range(len(criteria)))
    width = 0.24

    plt.figure(figsize=(12, 6))
    # Answer 1 计数柱
    plt.bar([i - width for i in x], answer1_vals, width=width, label="Answer 1")
    # Answer 2 计数柱
    plt.bar(x, answer2_vals, width=width, label="Answer 2")
    # Unknown 计数柱
    plt.bar([i + width for i in x], unknown_vals, width=width, label="Unknown")

    plt.xticks(x, criteria, rotation=15)
    plt.ylabel("Count")
    plt.title("Global Pairwise Evaluation: Winner Counts by Criterion")
    plt.legend()
    plt.tight_layout()

    output_path = output_dir / "global_winner_counts.png"
    plt.savefig(output_path, dpi=160)
    plt.close()

    return str(output_path), {
        "criteria": criteria,
        "answer1_counts": answer1_vals,
        "answer2_counts": answer2_vals,
        "unknown_counts": unknown_vals,
    }


def _plot_global_win_rate(global_eval: dict, output_dir: Path) -> Tuple[str, dict]:
    """
    功能说明：
        绘制全局评估按维度胜率对比图（Answer1 vs Answer2）。

    参数：
        - global_eval (dict)：global_pairwise 评估结果。
        - output_dir (Path)：输出目录。

    返回：
        Tuple[str, dict]：图像路径 + 图中使用的胜率数据。
    """
    win_rate = global_eval.get("summary", {}).get("win_rate", {})
    criteria = list(win_rate.keys())

    answer1_rates = [float(win_rate[c].get("answer1_win_rate", 0.0)) for c in criteria]
    answer2_rates = [float(win_rate[c].get("answer2_win_rate", 0.0)) for c in criteria]

    x = list(range(len(criteria)))
    width = 0.35

    plt.figure(figsize=(12, 6))
    # Answer 1 胜率柱
    plt.bar([i - width / 2 for i in x], answer1_rates, width=width, label="Answer 1 Win Rate")
    # Answer 2 胜率柱
    plt.bar([i + width / 2 for i in x], answer2_rates, width=width, label="Answer 2 Win Rate")

    plt.xticks(x, criteria, rotation=15)
    plt.ylim(0.0, 1.0)
    plt.ylabel("Win Rate")
    plt.title("Global Pairwise Evaluation: Win Rate by Criterion")
    plt.legend()
    plt.tight_layout()

    output_path = output_dir / "global_win_rate.png"
    plt.savefig(output_path, dpi=160)
    plt.close()

    return str(output_path), {
        "criteria": criteria,
        "answer1_win_rate": answer1_rates,
        "answer2_win_rate": answer2_rates,
    }


def _plot_local_correctness(local_eval: dict, output_dir: Path) -> Tuple[str, dict]:
    """
    功能说明：
        绘制本地正确性评估分布图（yes/no/unknown）。

    参数：
        - local_eval (dict)：local_correctness 评估结果。
        - output_dir (Path)：输出目录。

    返回：
        Tuple[str, dict]：图像路径 + 统计数据。
    """
    counts = local_eval.get("summary", {}).get("counts", {})
    yes_count = int(counts.get("yes", 0))
    no_count = int(counts.get("no", 0))
    unknown_count = int(counts.get("unknown", 0))
    accuracy = float(local_eval.get("summary", {}).get("accuracy", 0.0))

    labels = ["yes", "no", "unknown"]
    values = [yes_count, no_count, unknown_count]
    colors = ["#2ca02c", "#d62728", "#7f7f7f"]

    plt.figure(figsize=(7, 7))
    # 饼图展示本地正确性分布
    plt.pie(values, labels=labels, autopct="%1.1f%%", startangle=90, colors=colors)
    plt.title(f"Local Correctness Distribution (accuracy={accuracy:.3f})")
    plt.tight_layout()

    output_path = output_dir / "local_correctness_distribution.png"
    plt.savefig(output_path, dpi=160)
    plt.close()

    return str(output_path), {
        "labels": labels,
        "values": values,
        "accuracy": accuracy,
    }


def _plot_cross_method_compare(
    global_eval: dict | None,
    local_eval: dict | None,
    output_dir: Path,
) -> Tuple[str | None, dict]:
    """
    功能说明：
        绘制跨方法对比图：
        - 全局：Overall Winner 中 Answer 1 胜率
        - 本地：Accuracy

    参数：
        - global_eval (dict | None)：全局评估结果。
        - local_eval (dict | None)：本地评估结果。
        - output_dir (Path)：输出目录。

    返回：
        Tuple[str | None, dict]：图像路径（可能为空）+ 对比数据。
    """
    compare_data = {}
    labels: List[str] = []
    values: List[float] = []

    # 读取全局 Overall Winner 的 Answer1 胜率，作为全局方法核心指标
    if global_eval is not None:
        global_win_rate = (
            global_eval.get("summary", {})
            .get("win_rate", {})
            .get("Overall Winner", {})
            .get("answer1_win_rate", 0.0)
        )
        labels.append("Global A1 WinRate")
        values.append(float(global_win_rate))
        compare_data["global_answer1_overall_win_rate"] = float(global_win_rate)

    # 读取本地 accuracy，作为本地方法核心指标
    if local_eval is not None:
        local_acc = float(local_eval.get("summary", {}).get("accuracy", 0.0))
        labels.append("Local Accuracy")
        values.append(local_acc)
        compare_data["local_accuracy"] = local_acc

    # 若无可对比指标，则不生成图
    if not labels:
        return None, compare_data

    plt.figure(figsize=(8, 5))
    # 单图展示两个核心指标，方便快速对比
    plt.bar(labels, values, color=["#1f77b4", "#ff7f0e"][: len(labels)])
    plt.ylim(0.0, 1.0)
    plt.ylabel("Score")
    plt.title("Cross-Method Metric Comparison")
    plt.tight_layout()

    output_path = output_dir / "cross_method_compare.png"
    plt.savefig(output_path, dpi=160)
    plt.close()

    return str(output_path), compare_data


def main() -> None:
    """
    功能说明：
        主入口：读取评估结果 -> 生成可视化图表 -> 输出统计摘要 JSON。
    """
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "inputs": {
            "global_eval_file": args.global_eval_file,
            "local_eval_file": args.local_eval_file,
        },
        "figures": {},
        "metrics": {},
    }

    # 若环境缺少 matplotlib，给出明确报错信息
    if not _HAS_MATPLOTLIB:
        raise SystemExit(
            "matplotlib is required for visualization. Please install it first, e.g. `pip install matplotlib`."
        )

    global_eval = None
    local_eval = None

    # 若提供了全局评估文件，则生成全局相关图表
    if args.global_eval_file:
        global_eval = load_json(args.global_eval_file)
        fig1, data1 = _plot_global_winner_counts(global_eval, output_dir)
        fig2, data2 = _plot_global_win_rate(global_eval, output_dir)
        summary["figures"]["global_winner_counts"] = fig1
        summary["figures"]["global_win_rate"] = fig2
        summary["metrics"]["global_winner_counts"] = data1
        summary["metrics"]["global_win_rate"] = data2

    # 若提供了本地评估文件，则生成本地相关图表
    if args.local_eval_file:
        local_eval = load_json(args.local_eval_file)
        fig3, data3 = _plot_local_correctness(local_eval, output_dir)
        summary["figures"]["local_correctness_distribution"] = fig3
        summary["metrics"]["local_correctness_distribution"] = data3

    # 生成跨方法对比图（若任一可比指标存在）
    fig4, data4 = _plot_cross_method_compare(global_eval, local_eval, output_dir)
    if fig4 is not None:
        summary["figures"]["cross_method_compare"] = fig4
    if data4:
        summary["metrics"]["cross_method_compare"] = data4

    # 输出可视化统计摘要 JSON
    save_json(args.summary_file, summary)
    print(f"Saved visualization summary to: {args.summary_file}")
    for name, path in summary["figures"].items():
        print(f"[FIGURE] {name}: {path}")


if __name__ == "__main__":
    main()
