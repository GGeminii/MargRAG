import re
import json
import time
import yaml
import sys
import torch
import pathlib
import asyncio
import argparse

from pathlib import Path
from typing import List, Tuple, Any, Dict
from datetime import datetime, timezone

from transformers import AutoModel

from megarag import MegaRAG
from megarag.llms.hf import hf_gme_embed
from megarag.llms.openai import gpt_4o_mini_complete

from lightrag.base import QueryParam
from lightrag.utils import TokenTracker, wrap_embedding_func_with_attrs
from lightrag.kg.shared_storage import initialize_pipeline_status

# --- 可选进度条依赖 ---
try:
    import tqdm.auto as tqdmauto  # tqdmauto.tqdm + tqdmauto.tqdm.write
except Exception:
    tqdmauto = None

# 兼容 Markdown 问题格式：- Question N: xxx
QUERY_RE = re.compile(r"-\s*Question\s+\d+:\s+(.+)")


def _now_iso() -> str:
    """返回 UTC 时间戳字符串，便于结果落盘追踪。"""
    return datetime.now(timezone.utc).isoformat()


def _torch_speed_tweaks():
    """开启推理常见加速设置（在可用时生效）。"""
    try:
        torch.set_grad_enabled(False)
    except Exception:
        pass
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")


def initialize_model():
    """初始化嵌入模型。"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "Alibaba-NLP/gme-Qwen2-VL-2B-Instruct"
    embed_model = (
        AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="cuda" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )
        .to(device, non_blocking=True)
        .eval()
    )
    return embed_model


def load_addon_params(config_path: Path) -> dict:
    """读取 YAML 配置，兼容顶层或 addon_params 嵌套结构。"""
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data.get("addon_params", data)


async def initialize_rag(working_dir: Path, addon_params: dict) -> Tuple[MegaRAG, TokenTracker]:
    """初始化 MegaRAG 实例。"""
    embed_model = initialize_model()

    # 通过信号量限制嵌入并发，避免显存峰值过高
    embed_parallel_limit = addon_params.get("embed_parallel_limit", 1)
    _embed_sem = asyncio.Semaphore(embed_parallel_limit)

    @wrap_embedding_func_with_attrs(embedding_dim=1536, max_token_size=32768)
    async def embed_func(texts=[], images=[], is_query=False):
        # 单次嵌入调用限流
        async with _embed_sem:
            return await hf_gme_embed(
                embed_model=embed_model,
                texts=texts,
                images=images,
                is_query=is_query,
            )

    # Token 统计器：用于观察总 token 消耗
    token_tracker = TokenTracker()

    async def llm_func(
        prompt,
        input_images=None,
        system_prompt=None,
        history_messages=None,
        keyword_extraction=False,
        **kwargs,
    ):
        # 统一封装 LLM 调用，写入 token 统计
        return await gpt_4o_mini_complete(
            prompt=prompt,
            input_images=input_images,
            system_prompt=system_prompt,
            history_messages=history_messages,
            keyword_extraction=keyword_extraction,
            token_tracker=token_tracker,
            **kwargs,
        )

    # 构造 RAG 对象并初始化存储
    rag = MegaRAG(
        working_dir=str(working_dir),
        llm_model_func=llm_func,
        embedding_func=embed_func,
        addon_params=addon_params,
    )
    await rag.initialize_storages()
    await initialize_pipeline_status()
    return rag, token_tracker


def extract_queries(file_path: pathlib.Path) -> List[str]:
    """
    读取问题列表，支持两种格式：
    1) JSONL: {"question": "..."} / {"text": "..."}
    2) Markdown: - Question N: ...
    """
    if file_path.suffix.lower() == ".jsonl":
        out = []
        with file_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    q = obj.get("question") or obj.get("text")
                    if q:
                        out.append(str(q))
                except json.JSONDecodeError:
                    # 若该行不是 JSON，则回退到正则提取
                    m = QUERY_RE.search(line.replace("**", ""))
                    if m:
                        out.append(m.group(1).strip())
        return out

    # 默认走文本正则提取模式
    text = file_path.read_text(encoding="utf-8").replace("**", "")
    return [q.strip() for q in QUERY_RE.findall(text)]


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="Run MMKG debate query mode and save results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config-file", default="addon_params.yaml", metavar="FILE")
    parser.add_argument("--input-queries", default="./queries.jsonl", metavar="FILE")
    parser.add_argument("--working-dir", default="./exp", metavar="DIR")
    parser.add_argument("--max-retries", type=int, default=3, metavar="N")
    parser.add_argument("--retry-delay", type=float, default=3.0, metavar="SECONDS")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        metavar="N",
        help="并发问题数（根据 GPU / API 配额调整）。",
    )
    parser.add_argument(
        "--query-mode",
        default="mmkg_debate",
        metavar="MODE",
        help="查询模式，默认 mmkg_debate。",
    )
    parser.add_argument(
        "--print-as-complete",
        action="store_true",
        help="每个问题完成后立即打印结果（而不是按输入顺序统一打印）。",
    )
    parser.add_argument(
        "--output-file",
        default="results_mmkg_debate.json",
        metavar="FILE",
        help="输出文件路径。",
    )
    parser.add_argument(
        "--output-format",
        choices=["jsonl", "json"],
        default="json",
        help="输出格式：jsonl（逐行）或 json（单文件数组+元信息）。",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="仅 jsonl 模式可用：追加写入。",
    )
    # --- 进度条开关 ---
    parser.add_argument("--progress", dest="progress", action="store_true", help="显示进度条。")
    parser.add_argument("--no-progress", dest="progress", action="store_false", help="关闭进度条。")
    parser.set_defaults(progress=True)
    return parser.parse_args()


async def run_query_with_retries(
    rag: MegaRAG,
    question: str,
    param: QueryParam,
    max_retries: int,
    retry_delay: float,
):
    """带重试的单问题执行器。"""
    attempt = 0
    while True:
        attempt += 1
        try:
            # 记录单题耗时
            t0 = time.perf_counter()
            res = await rag.aquery(question, param=param)
            dt = time.perf_counter() - t0
            return res, dt, None
        except Exception as e:
            # 超过重试上限则返回错误
            if attempt >= max_retries:
                return None, None, e
            # 指数退避 + 小抖动，降低瞬时失败概率
            jitter = min(0.5, retry_delay * 0.1)
            await asyncio.sleep(retry_delay * (2 ** (attempt - 1)) + jitter)


async def async_main():
    """异步主流程：初始化 -> 并发查询 -> 落盘。"""
    _torch_speed_tweaks()
    args = parse_args()

    # 确保工作目录存在
    working_dir = Path(args.working_dir)
    working_dir.mkdir(parents=True, exist_ok=True)

    # 读取配置并初始化 RAG
    addon_params = load_addon_params(Path(args.config_file))
    rag, token_tracker = await initialize_rag(working_dir, addon_params)

    # 构造查询参数：默认走 mmkg_debate
    param = QueryParam(
        mode=args.query_mode,
        chunk_top_k=addon_params.get("chunk_top_k", 6),
        enable_rerank=False,
    )

    # 读取问题
    questions = extract_queries(Path(args.input_queries))
    if not questions:
        raise SystemExit(f"No questions found in {args.input_queries}")

    # 并发信号量：控制同时在跑的问题数
    sem = asyncio.Semaphore(max(1, args.concurrency))

    async def worker(idx: int, q: str):
        # 单题执行与结构化记录
        started_at = _now_iso()
        async with sem:
            res, dt, err = await run_query_with_retries(
                rag, q, param, max_retries=args.max_retries, retry_delay=args.retry_delay
            )
        finished_at = _now_iso()
        record: Dict[str, Any] = {
            "index": idx,
            "question": q,
            "answer": res,
            "latency_seconds": dt,
            "error": str(err) if err else None,
            "started_at": started_at,
            "finished_at": finished_at,
            "query_mode": args.query_mode,
        }
        return idx, record

    # 创建所有任务
    tasks = [asyncio.create_task(worker(i, q)) for i, q in enumerate(questions)]

    # 进度条启用条件：参数开启 + 依赖可用 + stderr 为 TTY
    use_bar = (
        args.progress
        and (tqdmauto is not None)
        and (hasattr(sys.stderr, "isatty") and sys.stderr.isatty())
    )
    pbar = tqdmauto.tqdm(total=len(tasks), desc="Processing debate queries", unit="q") if use_bar else None

    # 收集结果：按完成顺序更新进度，但最终可按输入顺序输出
    finished: List[Any] = [None] * len(tasks)
    err_count = 0

    for fut in asyncio.as_completed(tasks):
        idx, rec = await fut
        finished[idx] = rec
        if pbar:
            if rec["error"]:
                err_count += 1
                pbar.set_postfix_str(f"errors={err_count}")
            pbar.update(1)

        # 可选：完成即打印
        if args.print_as_complete:
            line = (
                f"[{idx}] ERROR for question: {rec['question']}\n  {rec['error']}\n"
                if rec["error"]
                else f"[{idx}] ({rec['latency_seconds']:.2f}s) {rec['answer']}\n"
            )
            if pbar:
                tqdmauto.tqdm.write(line)
            else:
                print(line)

    if pbar:
        pbar.close()

    # 若未启用“完成即打印”，则按输入顺序打印
    if not args.print_as_complete:
        for idx, rec in enumerate(finished):
            if rec["error"]:
                print(f"[{idx}] ERROR for question: {rec['question']}\n  {rec['error']}\n")
            else:
                print(f"[{idx}] ({rec['latency_seconds']:.2f}s) {rec['answer']}\n")

    # 输出目录自动创建
    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 按目标格式落盘
    if args.output_format == "jsonl":
        mode = "a" if args.append else "w"
        with out_path.open(mode, encoding="utf-8") as f:
            for rec in finished:
                f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        print(f"Saved {len(finished)} records to {out_path} (JSONL).")
    else:
        payload = {
            "results": finished,
            "metadata": {
                "final_token_usage": str(token_tracker),
                "query_file": str(args.input_queries),
                "query_mode": args.query_mode,
                "generated_at": _now_iso(),
            },
        }
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
        print(f"Saved results + metadata to {out_path} (JSON array).")

    print(f"Final Token Usage: {token_tracker}.")


def main():
    """同步入口：优先启用 uvloop，然后进入异步主流程。"""
    try:
        import uvloop  # 可选加速
        uvloop.install()
    except Exception:
        pass
    asyncio.run(async_main())


if __name__ == "__main__":
    main()

