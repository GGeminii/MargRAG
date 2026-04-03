import re
import json
import asyncio
from pathlib import Path
from typing import Any, Dict, List, Tuple

from megarag.prompt_evaluation import EVAL_PROMPTS


def load_json(path: str | Path) -> dict:
    """
    功能说明：
        读取 JSON 文件并返回字典对象。

    参数：
        - path (str | Path)：输入文件路径。

    返回：
        dict：解析后的 JSON 字典。
    """
    file_path = Path(path)
    with file_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str | Path, data: dict) -> None:
    """
    功能说明：
        将字典对象以 JSON 格式写入文件。

    参数：
        - path (str | Path)：输出文件路径。
        - data (dict)：待保存的字典数据。

    返回：
        None：通过副作用完成写文件。
    """
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _safe_parse_json(text: str) -> dict | None:
    """
    功能说明：
        从 LLM 输出文本中尽可能稳健地提取 JSON 对象。

    参数：
        - text (str)：LLM 返回文本。

    返回：
        dict | None：成功返回 JSON 字典；失败返回 None。
    """
    # 先尝试直接按完整 JSON 解析（最快路径）
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # 再尝试解析 ```json ... ``` 包裹内容
    fenced = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if fenced:
        try:
            obj = json.loads(fenced.group(1))
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    # 最后使用 JSONDecoder 扫描文本中的第一个 JSON 对象
    decoder = json.JSONDecoder()
    for idx, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[idx:])
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue
    return None


def _normalize_winner(value: str) -> str:
    """
    功能说明：
        将 Winner 字段规范为 Answer 1 / Answer 2 / Unknown。

    参数：
        - value (str)：原始 Winner 文本。

    返回：
        str：规范化后的 winner 标签。
    """
    norm = (value or "").strip().lower()
    if "answer 1" in norm or norm in {"1", "a1"}:
        return "Answer 1"
    if "answer 2" in norm or norm in {"2", "a2"}:
        return "Answer 2"
    return "Unknown"


def _normalize_yes_no(value: str) -> str:
    """
    功能说明：
        将 yes/no 判定字段标准化。

    参数：
        - value (str)：原始 is_correct 字段值。

    返回：
        str：标准化结果（yes / no / unknown）。
    """
    norm = (value or "").strip().lower()
    if norm in {"yes", "y", "true", "correct"}:
        return "yes"
    if norm in {"no", "n", "false", "incorrect"}:
        return "no"
    return "unknown"


async def _judge_with_retry(
    llm_func: Any,
    prompt_text: str,
    *,
    max_retries: int = 3,
    retry_delay: float = 1.0,
) -> Tuple[dict | None, str]:
    """
    功能说明：
        调用评估 LLM，并在失败时进行重试。

    参数：
        - llm_func (Any)：异步 LLM 调用函数。
        - prompt_text (str)：评估提示词文本。
        - max_retries (int)：最大重试次数。
        - retry_delay (float)：基础重试等待秒数。

    返回：
        Tuple[dict | None, str]：解析后的 JSON 结果 + 原始输出文本。
    """
    attempt = 0
    last_raw = ""
    while attempt < max_retries:
        attempt += 1
        try:
            # 这里将 prompt 放到 user 输入，避免覆盖业务侧 system_prompt
            raw = await llm_func(prompt_text, stream=False)
            last_raw = raw if isinstance(raw, str) else str(raw)
            parsed = _safe_parse_json(last_raw)
            if parsed is not None:
                return parsed, last_raw
        except Exception as _:
            # 忽略本次异常，进入重试
            pass
        await asyncio.sleep(retry_delay * (2 ** (attempt - 1)))
    return None, last_raw


def _build_global_index_map(results: List[dict]) -> Dict[Any, dict]:
    """
    功能说明：
        将 results 列表构建为 index -> item 的字典映射。

    参数：
        - results (List[dict])：结果列表。

    返回：
        Dict[Any, dict]：按 index 键映射的字典。
    """
    mapping: Dict[Any, dict] = {}
    for item in results:
        idx = item.get("index")
        if idx is not None:
            mapping[idx] = item
    return mapping


async def evaluate_global_pairwise(
    llm_func: Any,
    ours_result_file: str | Path,
    baseline_result_file: str | Path,
    *,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    concurrency: int = 6,
) -> dict:
    """
    功能说明：
        评估方法1（全局 QA）：
        在无标准答案场景下，对两个系统结果进行逐题 pairwise 评估。

    参数：
        - llm_func (Any)：异步 LLM 调用函数。
        - ours_result_file (str | Path)：我方结果文件（Answer 1）。
        - baseline_result_file (str | Path)：对比结果文件（Answer 2）。
        - max_retries (int)：单题最大重试次数。
        - retry_delay (float)：重试基础间隔秒数。
        - concurrency (int)：并发评估数。

    返回：
        dict：包含逐题明细与聚合统计的结果字典。
    """
    ours_json = load_json(ours_result_file)
    base_json = load_json(baseline_result_file)

    ours_results = ours_json.get("results", [])
    base_results = base_json.get("results", [])
    ours_map = _build_global_index_map(ours_results)
    base_map = _build_global_index_map(base_results)

    # 只评估两个结果共同拥有的 index，保证一一对应
    common_indices = sorted(set(ours_map.keys()) & set(base_map.keys()))

    # 评估维度定义，便于统一统计
    criteria = ["Comprehensiveness", "Diversity", "Empowerment", "Overall Winner"]
    stats = {
        c: {"Answer 1": 0, "Answer 2": 0, "Unknown": 0}
        for c in criteria
    }

    sem = asyncio.Semaphore(max(1, concurrency))

    async def _worker(idx: Any) -> dict:
        # 取同 index 的问题与两组答案
        left = ours_map[idx]
        right = base_map[idx]
        query = left.get("question", right.get("question", ""))
        answer1 = left.get("answer", "")
        answer2 = right.get("answer", "")

        # 组装评估 prompt
        prompt_text = EVAL_PROMPTS["global_pairwise"].format(
            query=query,
            answer1=answer1,
            answer2=answer2,
        )

        # 并发限流，避免评估模型并发过高
        async with sem:
            parsed, raw = await _judge_with_retry(
                llm_func,
                prompt_text,
                max_retries=max_retries,
                retry_delay=retry_delay,
            )

        # 失败兜底：填充 unknown 结构
        if parsed is None:
            parsed = {
                "Comprehensiveness": {"Winner": "Unknown", "Explanation": "parse_failed"},
                "Diversity": {"Winner": "Unknown", "Explanation": "parse_failed"},
                "Empowerment": {"Winner": "Unknown", "Explanation": "parse_failed"},
                "Overall Winner": {"Winner": "Unknown", "Explanation": "parse_failed"},
            }

        # 清洗并标准化 Winner 字段
        normalized = {}
        for c in criteria:
            cell = parsed.get(c, {}) if isinstance(parsed.get(c), dict) else {}
            winner = _normalize_winner(str(cell.get("Winner", "Unknown")))
            explanation = str(cell.get("Explanation", "")).strip()
            normalized[c] = {"Winner": winner, "Explanation": explanation}

        return {
            "index": idx,
            "question": query,
            "answer1_source": str(ours_result_file),
            "answer2_source": str(baseline_result_file),
            "evaluation": normalized,
            "raw_judge_output": raw,
        }

    # 并发执行全部 pairwise 评估
    records = await asyncio.gather(*[_worker(i) for i in common_indices])

    # 聚合统计：按各维度统计 winner 分布
    for rec in records:
        for c in criteria:
            w = rec["evaluation"][c]["Winner"]
            if w not in stats[c]:
                w = "Unknown"
            stats[c][w] += 1

    # 计算胜率（只针对有效样本：Answer 1 / Answer 2）
    win_rate = {}
    for c in criteria:
        valid = stats[c]["Answer 1"] + stats[c]["Answer 2"]
        if valid > 0:
            win_rate[c] = {
                "answer1_win_rate": stats[c]["Answer 1"] / valid,
                "answer2_win_rate": stats[c]["Answer 2"] / valid,
            }
        else:
            win_rate[c] = {
                "answer1_win_rate": 0.0,
                "answer2_win_rate": 0.0,
            }

    return {
        "method": "global_pairwise",
        "meta": {
            "ours_result_file": str(ours_result_file),
            "baseline_result_file": str(baseline_result_file),
            "total_common_questions": len(common_indices),
            "criteria": criteria,
        },
        "summary": {
            "winner_counts": stats,
            "win_rate": win_rate,
        },
        "details": records,
    }


async def evaluate_local_correctness(
    llm_func: Any,
    result_file: str | Path,
    gt_file: str | Path,
    *,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    concurrency: int = 6,
) -> dict:
    """
    功能说明：
        评估方法2（本地 QA）：
        基于问题 + 模型回答 + 标准答案，做 yes/no 正确性判定并统计准确率。

    参数：
        - llm_func (Any)：异步 LLM 调用函数。
        - result_file (str | Path)：模型结果文件（含 question/answer）。
        - gt_file (str | Path)：标准答案文件（支持 list 或 {"results":[...]}）。
        - max_retries (int)：单题最大重试次数。
        - retry_delay (float)：重试基础间隔秒数。
        - concurrency (int)：并发评估数。

    返回：
        dict：包含逐题明细与准确率统计的结果字典。
    """
    result_json = load_json(result_file)
    gt_json = load_json(gt_file)

    model_results = result_json.get("results", [])
    # ground truth 兼容两种结构：list / {"results": [...]}
    gt_results = gt_json.get("results", gt_json) if isinstance(gt_json, dict) else gt_json
    if not isinstance(gt_results, list):
        gt_results = []

    result_map = _build_global_index_map(model_results)
    gt_map = _build_global_index_map(gt_results)
    common_indices = sorted(set(result_map.keys()) & set(gt_map.keys()))

    sem = asyncio.Semaphore(max(1, concurrency))

    async def _worker(idx: Any) -> dict:
        # 构造同 index 的输入三元组：问题、模型回答、标准答案
        pred = result_map[idx]
        gt = gt_map[idx]

        query = pred.get("question", gt.get("question", ""))
        result = pred.get("answer", pred.get("result", ""))
        answer = gt.get("answer", gt.get("gt_answer", gt.get("reference_answer", "")))

        # 组装评估 prompt
        prompt_text = EVAL_PROMPTS["local_correctness"].format(
            query=query,
            result=result,
            answer=answer,
        )

        # 并发限流调用评估模型
        async with sem:
            parsed, raw = await _judge_with_retry(
                llm_func,
                prompt_text,
                max_retries=max_retries,
                retry_delay=retry_delay,
            )

        # 失败兜底
        if parsed is None:
            parsed = {
                "is_correct": "unknown",
                "reason": "parse_failed",
            }

        # 字段标准化
        is_correct = _normalize_yes_no(str(parsed.get("is_correct", "unknown")))
        reason = str(parsed.get("reason", "")).strip()

        return {
            "index": idx,
            "question": query,
            "model_response": result,
            "reference_answer": answer,
            "evaluation": {
                "is_correct": is_correct,
                "reason": reason,
            },
            "raw_judge_output": raw,
        }

    # 并发执行
    records = await asyncio.gather(*[_worker(i) for i in common_indices])

    # 统计数量
    yes_count = sum(1 for r in records if r["evaluation"]["is_correct"] == "yes")
    no_count = sum(1 for r in records if r["evaluation"]["is_correct"] == "no")
    unknown_count = sum(1 for r in records if r["evaluation"]["is_correct"] == "unknown")
    valid = yes_count + no_count
    accuracy = (yes_count / valid) if valid > 0 else 0.0

    return {
        "method": "local_correctness",
        "meta": {
            "result_file": str(result_file),
            "gt_file": str(gt_file),
            "total_common_questions": len(common_indices),
        },
        "summary": {
            "counts": {
                "yes": yes_count,
                "no": no_count,
                "unknown": unknown_count,
            },
            "accuracy": accuracy,
        },
        "details": records,
    }

