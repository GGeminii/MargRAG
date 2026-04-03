import os
import asyncio
import traceback
import time, json, uuid, contextvars  # timing + correlation
import atexit, threading
from pathlib import Path
from contextlib import ExitStack

from typing import (
    Any,
    AsyncIterator,
    Callable,
    Iterator,
    cast,
    final,
    Literal,
    Optional,
    List,
    Dict,
)
from functools import partial

from dataclasses import (
    asdict,
    dataclass,
    field
)

from datetime import (
    datetime,
    timezone
)

from lightrag.kg.shared_storage import (
    get_namespace_data,
    get_pipeline_status_lock,
    get_graph_db_lock,
)

from lightrag import LightRAG
from lightrag.utils import (
    Tokenizer,
    always_get_an_event_loop,
    logger,
    clean_text,
    compute_mdhash_id,
    get_content_summary,
    lazy_external_import,
)
from lightrag.base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    DocProcessingStatus,
    DocStatus,
    DocStatusStorage,
    QueryParam,
    StorageNameSpace,
    StoragesStatus,
    DeletionResult,
)

from megarag.operate import (
    chunking_by_token_or_page,
    extract_entities,
    extract_entities_refinement,
    merge_nodes_and_edges,
    kg_query,
    naive_query,
    kg_two_step_query,
    kg_debate_query,
)

from megarag.utils import plot_waterfall_from_jsonl

from megarag.kg import STORAGES

trace_id_var = contextvars.ContextVar("trace_id", default="-")
_span_stack_var = contextvars.ContextVar("span_stack", default=())  # for parent/child spans

_TIMING_JSONL_FILE = None     # type: Optional[Any]
_TIMING_JSONL_LOCK = threading.Lock()

def enable_timing_jsonl(path: str = "logs/timings.jsonl"):
    """
    功能说明：
        初始化并启用 JSONL 计时日志输出。
    
    参数：
        - path (str)：方法执行所需输入参数。
    
    返回：
        Any：方法执行结果；若为 None 表示主要通过副作用完成处理。
    """
    # 初始化计时日志文件（JSONL），供后续所有 stage 统一追加写入。
    global _TIMING_JSONL_FILE
    Path(os.path.dirname(path) or ".").mkdir(parents=True, exist_ok=True)
    _TIMING_JSONL_FILE = open(path, "a", encoding="utf-8", buffering=1)  # line-buffered
    atexit.register(_TIMING_JSONL_FILE.close)
    return _TIMING_JSONL_FILE

_StageListener = Callable[[dict], None]
_STAGE_LISTENERS: List[_StageListener] = []
_STAGE_LISTENERS_LOCK = threading.Lock()

def add_stage_listener(listener: _StageListener) -> None:
    """
    功能说明：
        注册阶段事件监听器，用于进度与性能观测。

    参数：
        - listener (_StageListener)：方法执行所需输入参数。

    返回：
        None：方法执行结果；若为 None 表示主要通过副作用完成处理。
    """
    with _STAGE_LISTENERS_LOCK:
        _STAGE_LISTENERS.append(listener)

def remove_stage_listener(listener: _StageListener) -> None:
    """
    功能说明：
        移除阶段事件监听器。

    参数：
        - listener (_StageListener)：方法执行所需输入参数。

    返回：
        None：方法执行结果；若为 None 表示主要通过副作用完成处理。
    """
    with _STAGE_LISTENERS_LOCK:
        try:
            _STAGE_LISTENERS.remove(listener)
        except ValueError:
            pass

def _notify_stage_listeners(event: dict) -> None:
    # Best-effort, never break the pipeline if UI fails
    """
    功能说明：
        向已注册监听器广播阶段事件。

    参数：
        - event (dict)：方法执行所需输入参数。

    返回：
        None：方法执行结果；若为 None 表示主要通过副作用完成处理。
    """
    # 拷贝监听器快照，避免回调过程中修改原列表导致并发问题。
    with _STAGE_LISTENERS_LOCK:
        listeners = tuple(_STAGE_LISTENERS)
    for fn in listeners:
        try:
            fn(event)
        except Exception:
            pass

class StageTimer:
    """Async context manager for precise wall-clock timing with span hierarchy.
    Emits both: (1) flattened JSON via logger, and (2) full record to JSONL file (if enabled).
    Also notifies listeners so a Rich progress can mirror the stages.
    """
    def __init__(self, name: str, tags: dict | None = None):
        """
        功能说明：
            初始化对象状态、配置项与依赖资源。

        参数：
            - self (Any)：当前类实例本身。
            - name (str)：方法执行所需输入参数。
            - tags (dict | None)：方法执行所需输入参数。

        返回：
            Any：方法执行结果；若为 None 表示主要通过副作用完成处理。
        """
        self.name = name
        self.tags = tags or {}
        self.t0_ns = 0
        self.span_id = uuid.uuid4().hex[:12]
        self.parent_span_id: Optional[str] = None

    async def __aenter__(self):
        """
        功能说明：
            异步上下文入口，完成运行前准备。

        参数：
            - self (Any)：当前类实例本身。

        返回：
            Any：方法执行结果；若为 None 表示主要通过副作用完成处理。
        """
        # 进入阶段时，建立父子 span 关系并记录起始时间。
        stack = _span_stack_var.get()
        self.parent_span_id = stack[-1] if stack else None
        _span_stack_var.set(stack + (self.span_id,))
        self.t0_ns = time.perf_counter_ns()

        # 广播 enter 事件，用于 UI 进度条和外部观测。
        _notify_stage_listeners({
            "event": "enter",
            "ts": time.time(),
            "trace_id": trace_id_var.get(),
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "stage": self.name,
            "tags": self.tags,
        })
        return self

    async def __aexit__(self, exc_type, exc, tb):
        """
        功能说明：
            异步上下文出口，执行资源清理与收尾。

        参数：
            - self (Any)：当前类实例本身。
            - exc_type (Any)：方法执行所需输入参数。
            - exc (Any)：方法执行所需输入参数。
            - tb (Any)：方法执行所需输入参数。

        返回：
            Any：方法执行结果；若为 None 表示主要通过副作用完成处理。
        """
        # 退出阶段时计算耗时并汇总完整记录。
        t1_ns = time.perf_counter_ns()
        dt_ms = (t1_ns - self.t0_ns) / 1e6
        # Full record (kept rich for JSONL)
        record = {
            "type": "stage_timing",
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),  # human UTC
            "trace_id": trace_id_var.get(),
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "stage": self.name,
            "elapsed_ms": dt_ms,
            "ok": exc is None,
            "error": repr(exc) if exc else None,
            "start_ns": self.t0_ns,
            "end_ns": t1_ns,
            "tags": self.tags,
        }
        # Flattened view for your existing logs
        flat = {k: v for k, v in record.items() if k != "tags"} | self.tags
        logger.info(json.dumps(flat, ensure_ascii=False))

        # 如果启用了 JSONL 落盘，则写入结构化计时日志。
        # Persist JSONL if enabled
        if _TIMING_JSONL_FILE:
            with _TIMING_JSONL_LOCK:
                _TIMING_JSONL_FILE.write(json.dumps(record, ensure_ascii=False) + "\n")

        # 广播 exit 事件，驱动 UI 完成当前 stage 展示。
        # Progress/listeners
        _notify_stage_listeners({
            "event": "exit",
            "ts": time.time(),
            "trace_id": trace_id_var.get(),
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "stage": self.name,
            "elapsed_ms": dt_ms,
            "ok": exc is None,
            "error": repr(exc) if exc else None,
            "tags": self.tags,
        })

        # 当前 span 出栈，保持上下文栈一致性。
        # Pop span from stack
        stack = _span_stack_var.get()
        if stack and stack[-1] == self.span_id:
            _span_stack_var.set(stack[:-1])
        else:
            _span_stack_var.set(tuple(s for s in stack if s != self.span_id))

def stage(name: str, **tags) -> StageTimer:
    """
    功能说明：
        创建阶段计时上下文，用于记录耗时与标签。

    参数：
        - name (str)：方法执行所需输入参数。
        - **tags (Any)：方法执行所需输入参数。

    返回：
        StageTimer：方法执行结果；若为 None 表示主要通过副作用完成处理。
    """
    return StageTimer(name, tags)

def timed_coro(name: str, coro_fn: Callable[..., Any], *args, **kwargs) -> Any:
    """
    功能说明：
        包装异步协程并记录执行耗时。

    参数：
        - name (str)：方法执行所需输入参数。
        - coro_fn (Callable[..., Any])：方法执行所需输入参数。
        - *args (Any)：方法执行所需输入参数。
        - **kwargs (Any)：方法执行所需输入参数。

    返回：
        Any：方法执行结果；若为 None 表示主要通过副作用完成处理。
    """
    async def _runner():
        """
        功能说明：
            执行被包装协程并返回结果。

        参数：
            - 无

        返回：
            Any：方法执行结果；若为 None 表示主要通过副作用完成处理。
        """
        # 给任意协程套一层 stage 计时，便于统一统计子任务耗时。
        async with stage(name, **kwargs.get("_tags", {})):
            real_kwargs = {k: v for k, v in kwargs.items() if k != "_tags"}
            return await coro_fn(*args, **real_kwargs)
    return _runner()

class StageProgress:
    """Drives a Rich Progress from StageTimer enter/exit events."""
    def __init__(self):
        """
        功能说明：
            初始化对象状态、配置项与依赖资源。

        参数：
            - self (Any)：当前类实例本身。

        返回：
            Any：方法执行结果；若为 None 表示主要通过副作用完成处理。
        """
        self.progress = None
        self._span2task: Dict[str, int] = {}
        self._span2depth: Dict[str, int] = {}
        self._task_desc: Dict[int, str] = {}
        self._lock = threading.Lock()

        try:
            from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
            from rich.console import Console
            if os.getenv("RAG_PROGRESS", "1") not in ("0", "false", "False"):
                self.progress = Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    TimeElapsedColumn(),
                    transient=True,
                    console=Console(stderr=True),  # keep stdout clean if you log JSON there
                )
        except Exception:
            self.progress = None

    def __enter__(self):
        """
        功能说明：
            上下文入口，启动进度展示或状态追踪。

        参数：
            - self (Any)：当前类实例本身。

        返回：
            Any：方法执行结果；若为 None 表示主要通过副作用完成处理。
        """
        if self.progress:
            self.progress.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        """
        功能说明：
            上下文出口，结束进度展示并清理状态。

        参数：
            - self (Any)：当前类实例本身。
            - exc_type (Any)：方法执行所需输入参数。
            - exc (Any)：方法执行所需输入参数。
            - tb (Any)：方法执行所需输入参数。

        返回：
            Any：方法执行结果；若为 None 表示主要通过副作用完成处理。
        """
        if self.progress:
            try:
                self.progress.stop()
            finally:
                self.progress.__exit__(exc_type, exc, tb)

    def handle(self, ev: dict) -> None:
        """
        功能说明：
            处理阶段事件回调并更新界面/状态。

        参数：
            - self (Any)：当前类实例本身。
            - ev (dict)：方法执行所需输入参数。

        返回：
            None：方法执行结果；若为 None 表示主要通过副作用完成处理。
        """
        if not self.progress:
            return
        # 串行处理 enter/exit 事件，避免并发更新进度条状态。
        with self._lock:
            kind = ev.get("event")
            if kind == "enter":
                parent = ev.get("parent_span_id")
                depth = (self._span2depth.get(parent, -1) + 1) if parent else 0
                self._span2depth[ev["span_id"]] = depth
                indent = "  " * depth
                desc = f"{indent}{ev['stage']}"
                task_id = self.progress.add_task(desc, total=1)
                self._span2task[ev["span_id"]] = task_id
                self._task_desc[task_id] = desc

            elif kind == "exit":
                span_id = ev["span_id"]
                task_id = self._span2task.pop(span_id, None)
                if task_id is not None:
                    base = self._task_desc.get(task_id, "")
                    ok = ev.get("ok", True)
                    ms = ev.get("elapsed_ms")
                    mark = "✓" if ok else "✗"
                    if ms is not None:
                        base = f"{base}  {mark}  ({ms:.0f} ms)"
                    else:
                        base = f"{base}  {mark}"
                    # finish the line
                    self.progress.update(task_id, description=base, completed=1, advance=1)

class MegaRAG(LightRAG):
    def _get_storage_class(self, storage_name: str) -> Callable[..., Any]:
        """
        功能说明：
            根据配置解析并返回对应存储实现类。

        参数：
            - self (Any)：当前类实例本身。
            - storage_name (str)：方法执行所需输入参数。

        返回：
            Callable[..., Any]：方法执行结果；若为 None 表示主要通过副作用完成处理。
        """
        import_path = STORAGES[storage_name]
        storage_class = lazy_external_import(import_path, storage_name)
        return storage_class

    def __post_init__(self):
        """
        功能说明：
            在 dataclass 初始化后补充派生属性与校验。

        参数：
            - self (Any)：当前类实例本身。

        返回：
            Any：方法执行结果；若为 None 表示主要通过副作用完成处理。
        """
        # 强制指定向量存储实现，并绑定自定义分块函数。
        self.vector_storage = "NanoMMVectorDBStorage"
        super().__post_init__()
        self.chunking_func  = chunking_by_token_or_page

    def insert(
        self,
        input: str | list[str],
        split_by_character: str | None = None,
        split_by_character_only: bool = False,
        split_by_page: bool = True,
        ids: str | list[str] | None = None,
        file_paths: str | list[str] | None = None,
    ) -> None:
        """
        功能说明：
            同步入口：将输入文档写入处理队列。

        参数：
            - self (Any)：当前类实例本身。
            - input (str | list[str])：方法执行所需输入参数。
            - split_by_character (str | None)：按指定字符切分文本；为 None 时不启用该规则。
            - split_by_character_only (bool)：是否仅按字符切分文本。
            - split_by_page (bool)：是否按分页边界切分文本。
            - ids (str | list[str] | None)：方法执行所需输入参数。
            - file_paths (str | list[str] | None)：方法执行所需输入参数。

        返回：
            None：方法执行结果；若为 None 表示主要通过副作用完成处理。
        """
        # 同步入口内部仍复用异步流程，这里先开启计时日志再进入事件循环。
        # Auto-enable JSONL sink (override via env RAG_TIMING_JSONL)
        tim_log_dir = os.path.join(self.working_dir, self.workspace, "logs/timings.jsonl")
        enable_timing_jsonl(os.getenv("RAG_TIMING_JSONL", tim_log_dir))

        loop = always_get_an_event_loop()
        loop.run_until_complete(
            self.ainsert(
                input, split_by_character, split_by_character_only, split_by_page, ids, file_paths
            )
        )

    async def ainsert(
        self,
        input: str | list[str],
        split_by_character: str | None = None,
        split_by_character_only: bool = False,
        split_by_page: bool = True,
        ids: str | list[str] | None = None,
        file_paths: str | list[str] | None = None,
    ) -> None:
        """
        功能说明：
            异步入口：将输入文档写入处理队列。

        参数：
            - self (Any)：当前类实例本身。
            - input (str | list[str])：方法执行所需输入参数。
            - split_by_character (str | None)：按指定字符切分文本；为 None 时不启用该规则。
            - split_by_character_only (bool)：是否仅按字符切分文本。
            - split_by_page (bool)：是否按分页边界切分文本。
            - ids (str | list[str] | None)：方法执行所需输入参数。
            - file_paths (str | list[str] | None)：方法执行所需输入参数。

        返回：
            None：方法执行结果；若为 None 表示主要通过副作用完成处理。
        """
        # 先入队，再启动队列消费，保证“提交+处理”语义完整。
        await self.apipeline_enqueue_documents(input, ids, file_paths)
        await self.apipeline_process_enqueue_documents(
            split_by_character, split_by_character_only, split_by_page
        )

    async def _process_entity_relation_graph(
        self, chunk: dict[str, Any], pipeline_status=None, pipeline_status_lock=None
    ) -> list:
        """
        功能说明：
            执行实体关系抽取主流程。

        参数：
            - self (Any)：当前类实例本身。
            - chunk (dict[str, Any])：单个文本块或文本块字典。
            - pipeline_status (Any)：流水线共享状态字典（用于进度展示与消息记录）。
            - pipeline_status_lock (Any)：保护 pipeline_status 的异步锁。

        返回：
            list：方法执行结果；若为 None 表示主要通过副作用完成处理。
        """
        try:
            # 调用实体关系抽取主流程，结果按 chunk 返回供后续合并。
            # Time the whole extraction over all chunks
            chunk_results = await extract_entities(
                chunk,
                global_config=asdict(self),
                pipeline_status=pipeline_status,
                pipeline_status_lock=pipeline_status_lock,
                llm_response_cache=self.llm_response_cache,
                text_chunks_storage=self.text_chunks,
            )
            return chunk_results
        except Exception as e:
            error_msg = f"Failed to extract entities and relationships: {str(e)}"
            logger.error(error_msg)
            async with pipeline_status_lock:
                pipeline_status["latest_message"] = error_msg
                pipeline_status["history_messages"].append(error_msg)
            raise e

    async def _process_entity_relation_graph_refinement(
        self,
        chunk: dict[str, Any],
        chunk_results: list,
        knowledge_graph_inst: BaseGraphStorage,
        entity_vdb: BaseVectorStorage,
        relationships_vdb: BaseVectorStorage,
        global_config: dict[str, str],
        llm_response_cache: BaseKVStorage | None = None,
        pipeline_status=None,
        pipeline_status_lock=None
    ) -> list:
        """
        功能说明：
            执行实体关系抽取精修流程。

        参数：
            - self (Any)：当前类实例本身。
            - chunk (dict[str, Any])：单个文本块或文本块字典。
            - chunk_results (list)：实体关系抽取的中间或最终结果列表。
            - knowledge_graph_inst (BaseGraphStorage)：知识图谱存储实例。
            - entity_vdb (BaseVectorStorage)：实体向量存储实例。
            - relationships_vdb (BaseVectorStorage)：关系向量存储实例。
            - global_config (dict[str, str])：全局运行配置字典。
            - llm_response_cache (BaseKVStorage | None)：LLM 响应缓存存储实例。
            - pipeline_status (Any)：流水线共享状态字典（用于进度展示与消息记录）。
            - pipeline_status_lock (Any)：保护 pipeline_status 的异步锁。

        返回：
            list：方法执行结果；若为 None 表示主要通过副作用完成处理。
        """
        try:
            # 精修阶段在已有结果上再次抽取，提升图谱质量。
            # Time the refinement extraction over all chunks
            async with stage("extract_entities_refine", chunks=len(chunk) if hasattr(chunk, "__len__") else None):
                chunk_results = await extract_entities_refinement(
                    chunk,
                    chunk_results,
                    knowledge_graph_inst,
                    entity_vdb,
                    relationships_vdb,
                    global_config=global_config,
                    pipeline_status=pipeline_status,
                    pipeline_status_lock=pipeline_status_lock,
                    llm_response_cache=llm_response_cache,
                    text_chunks_storage=self.text_chunks,
                )
            return chunk_results
        except Exception as e:
            error_msg = f"Failed to extract entities and relationships: {str(e)}"
            logger.error(error_msg)
            async with pipeline_status_lock:
                pipeline_status["latest_message"] = error_msg
                pipeline_status["history_messages"].append(error_msg)
            raise e

    async def apipeline_process_enqueue_documents(
        self,
        split_by_character: str | None = None,
        split_by_character_only: bool = False,
        split_by_page: bool = True,
    ) -> None:
        """
        功能说明：
            异步处理文档队列：分块、抽取、合并并更新状态。

        参数：
            - self (Any)：当前类实例本身。
            - split_by_character (str | None)：按指定字符切分文本；为 None 时不启用该规则。
            - split_by_character_only (bool)：是否仅按字符切分文本。
            - split_by_page (bool)：是否按分页边界切分文本。

        返回：
            None：方法执行结果；若为 None 表示主要通过副作用完成处理。
        """

        # 读取跨协程共享的流水线状态数据（全局进度、状态）
        # Get pipeline status shared data and lock
        pipeline_status = await get_namespace_data("pipeline_status")
        # 获取保护流水线状态的异步互斥锁，防止多协程同时修改导致数据异常
        pipeline_status_lock = get_pipeline_status_lock()

        # 单工作者保护逻辑：确保同一时间只有一个worker处理队列，避免重复执行
        # Check if another process is already processing the queue
        async with pipeline_status_lock:
            # Ensure only one worker is processing documents
            # 如果当前流水线未处于忙碌状态，开始处理文档
            if not pipeline_status.get("busy", False):
                # 并发获取三种状态的文档：处理中、失败、待处理
                processing_docs, failed_docs, pending_docs = await asyncio.gather(
                    self.doc_status.get_docs_by_status(DocStatus.PROCESSING),
                    self.doc_status.get_docs_by_status(DocStatus.FAILED),
                    self.doc_status.get_docs_by_status(DocStatus.PENDING),
                )

                # 本轮需要处理的文档集合 = 处理中 + 失败 + 待处理
                # 支持失败重试与断点续跑，无需从头开始处理
                to_process_docs: dict[str, DocProcessingStatus] = {}
                to_process_docs.update(processing_docs)
                to_process_docs.update(failed_docs)
                to_process_docs.update(pending_docs)

                # 无待处理文档，直接退出
                if not to_process_docs:
                    logger.info("No documents to process")
                    return

                # 重置本轮任务状态，为前端/日志提供统一的进度初始基线
                pipeline_status.update(
                    {
                        "busy": True,  # 标记流水线忙碌
                        "job_name": "Default Job",  # 默认任务名称
                        "job_start": datetime.now(timezone.utc).isoformat(),  # 任务开始时间
                        "docs": 0,  # 文档总数
                        "batchs": 0,  # 待处理文件总数
                        "cur_batch": 0,  # 已处理文件数
                        "request_pending": False,  # 清除待处理请求标记
                        "latest_message": "",  # 最新状态消息
                    }
                )
                # 清空历史消息列表，保留原列表引用，不破坏共享数据结构
                # Clean history_messages without breaking its shared list reference
                del pipeline_status["history_messages"][:]
            else:
                # 已有worker在运行：仅设置待处理标记，直接返回，不重复执行
                # Another process is busy: set the request flag and return
                pipeline_status["request_pending"] = True
                logger.info(
                    "Another process is already processing the document queue. Request queued."
                )
                return

        # 为本次流水线生成唯一追踪ID，串联所有阶段日志，方便问题排查
        # Create a trace_id for this pipeline run to tie all stage logs together
        trace_id_var.set(uuid.uuid4().hex)

        # ---- Stage-driven progress UI --------------------------------------
        # 初始化阶段进度管理器，用于展示处理进度、耗时统计
        sp = StageProgress()
        # 上下文管理器，自动管理资源释放
        with ExitStack() as stack:
            stack.enter_context(sp)  # 启用进度UI展示（如果支持）
            add_stage_listener(sp.handle)  # 订阅阶段计时事件，更新进度
            try:
                # pipeline_total 统计整个处理循环的总耗时，用于性能监控
                # Total wall-clock timing for the whole pipeline (covers the while-loop)
                async with stage("pipeline_total", job_name=pipeline_status.get("job_name", "Default Job")):
                    # 循环处理文档，直到无文档或无待处理请求
                    # Process documents until no more documents or requests
                    while True:
                        # 无待处理文档，结束循环
                        if not to_process_docs:
                            log_message = "All documents have been processed or are duplicates"
                            logger.info(log_message)
                            pipeline_status["latest_message"] = log_message
                            pipeline_status["history_messages"].append(log_message)
                            break

                        log_message = f"Processing {len(to_process_docs)} document(s)"
                        logger.info(log_message)

                        # 更新批次级进度信息：总文档数、总批次、已处理批次、最新消息
                        # Update pipeline_status: batchs now represents the total number of files to be processed
                        pipeline_status["docs"] = len(to_process_docs)
                        pipeline_status["batchs"] = len(to_process_docs)
                        pipeline_status["cur_batch"] = 0
                        pipeline_status["latest_message"] = log_message
                        pipeline_status["history_messages"].append(log_message)

                        # 获取第一个文档的ID和信息，用于生成任务名称
                        # Get first document's file path and total count for job name
                        first_doc_id, first_doc = next(iter(to_process_docs.items()))
                        first_doc_path = first_doc.file_path

                        # 处理文件路径为空的情况，生成友好的路径前缀
                        # Handle cases where first_doc_path is None
                        if first_doc_path:
                            path_prefix = first_doc_path[:20] + (
                                "..." if len(first_doc_path) > 20 else ""
                            )
                        else:
                            path_prefix = "unknown_source"

                        # 生成任务名称，展示处理文件信息
                        total_files = len(to_process_docs)
                        job_name = f"{path_prefix}[{total_files} files]"
                        pipeline_status["job_name"] = job_name

                        # 已处理文件计数器，用于进度展示
                        # Counter for processed files
                        processed_count = 0
                        # 信号量：限制并发处理的文件数，避免I/O/LLM服务压力过载
                        # Semaphore to limit concurrent file processing
                        semaphore = asyncio.Semaphore(self.max_parallel_insert)

                        async def process_document(
                                doc_id: str,
                                status_doc: DocProcessingStatus,
                                split_by_character: str | None,
                                split_by_character_only: bool,
                                split_by_page: bool,
                                pipeline_status: dict,
                                pipeline_status_lock: asyncio.Lock,
                                semaphore: asyncio.Semaphore,
                        ) -> None:
                            """
                            功能说明：
                                处理单个文档的完整流水线
                                包含：分块 → 入库 → 实体关系抽取 → 合并 → 精修 → 更新状态

                            参数：
                                - doc_id (str)：文档唯一标识
                                - status_doc (DocProcessingStatus)：文档处理状态对象（包含内容与元数据）
                                - split_by_character (str | None)：按指定字符切分文本；为 None 时不启用该规则
                                - split_by_character_only (bool)：是否仅按字符切分文本
                                - split_by_page (bool)：是否按分页边界切分文本
                                - pipeline_status (dict)：流水线共享状态字典（用于进度展示与消息记录）
                                - pipeline_status_lock (asyncio.Lock)：保护 pipeline_status 的异步锁
                                - semaphore: (asyncio.Semaphore)：用于限制并发数量的信号量

                            返回：
                                None：无返回值，通过数据库、共享状态完成处理逻辑
                            """
                            # 标记文件抽取阶段是否成功（抽取+入库）
                            # 单文件处理分两大阶段：抽取（stage1+stage2）与合并/精修
                            file_extraction_stage_ok = False
                            # 申请信号量，控制并发数
                            async with semaphore:
                                # 声明使用外部函数的计数器变量
                                nonlocal processed_count
                                current_file_number = 0

                                # 提前获取文件路径，用于顶层耗时统计标签
                                # Get file_path early so it's available in top-level timing tags
                                file_path = getattr(status_doc, "file_path", "unknown_source")

                                # 单个文件总处理耗时统计
                                # Per-file total timing
                                async with stage("file_total", doc_id=doc_id, file_path=file_path):
                                    # 第一阶段任务列表（并行执行的入库任务）
                                    first_stage_tasks = []
                                    # 实体关系抽取任务
                                    entity_relation_task = None
                                    try:
                                        # 加锁更新全局进度
                                        async with pipeline_status_lock:
                                            # Update processed file count and save current ordinal
                                            processed_count += 1
                                            current_file_number = processed_count
                                            pipeline_status["cur_batch"] = processed_count

                                            log_message = f"Extracting stage {current_file_number}/{total_files}: {file_path}"
                                            logger.info(log_message)
                                            pipeline_status["history_messages"].append(log_message)
                                            log_message = f"Processing d-id: {doc_id}"
                                            logger.info(log_message)
                                            pipeline_status["latest_message"] = log_message
                                            pipeline_status["history_messages"].append(log_message)

                                        # 执行文本分块，附带文档来源与缓存字段
                                        # Generate chunks from the document
                                        async with stage(
                                                "chunking",
                                                doc_id=doc_id,
                                                split_by_character=str(split_by_character),
                                                split_by_page=split_by_page
                                        ):
                                            # 分块参数配置
                                            chunk_args = {
                                                "tokenizer": self.tokenizer,
                                                "content": status_doc.content,
                                                "split_by_character": split_by_character,
                                                "split_by_character_only": split_by_character_only,
                                                "split_by_page": split_by_page,
                                                "overlap_token_size": self.chunk_overlap_token_size,
                                                "max_token_size": self.chunk_token_size,
                                            }

                                            # 执行分块，生成chunk并生成唯一ID，附加文档信息，compute_mdhash_id为给每个分块的文本内容content计算 MD5 唯一标识，自动加上前缀 chunk-
                                            chunks: dict[str, Any] = {
                                                compute_mdhash_id(dp["content"], prefix="chunk-"): {
                                                    **dp,
                                                    "full_doc_id": doc_id,
                                                    "file_path": file_path,  # 给每个分块绑定来源文件路径
                                                    "llm_cache_list": [],  # 初始化LLM缓存列表
                                                }
                                                for dp in self.chunking_func(**chunk_args)
                                            }

                                        # 无分块结果，打印警告
                                        if not chunks:
                                            logger.warning("No document chunks to process")

                                        # Stage1：并行执行数据入库操作（文档状态/全文/chunk KV/chunk向量库）
                                        # Stage 1: Upsert text chunks and docs (in parallel)
                                        # 创建异步任务：更新文档处理状态
                                        doc_status_task = asyncio.create_task(
                                            timed_coro(
                                                "upsert.doc_status",
                                                self.doc_status.upsert,
                                                {
                                                    doc_id: {
                                                        "status": DocStatus.PROCESSING,
                                                        "chunks_count": len(chunks),
                                                        "chunks_list": list(chunks.keys()),  # 保存chunk ID列表
                                                        "content": status_doc.content,
                                                        "content_summary": status_doc.content_summary,
                                                        "content_length": status_doc.content_length,
                                                        "created_at": status_doc.created_at,
                                                        "updated_at": datetime.now(timezone.utc).isoformat(),
                                                        "file_path": file_path,
                                                    }
                                                },
                                                _tags={"doc_id": doc_id}
                                            )
                                        )
                                        # 创建异步任务：chunk向量库入库
                                        chunks_vdb_task = asyncio.create_task(
                                            timed_coro("upsert.chunks_vdb", self.chunks_vdb.upsert, chunks,
                                                       _tags={"doc_id": doc_id})
                                        )
                                        # 创建异步任务：全文内容入库
                                        full_docs_task = asyncio.create_task(
                                            timed_coro(
                                                "upsert.full_docs",
                                                self.full_docs.upsert,
                                                {doc_id: {"content": status_doc.content}},
                                                _tags={"doc_id": doc_id}
                                            )
                                        )
                                        # 创建异步任务：文本分块KV库入库
                                        text_chunks_task = asyncio.create_task(
                                            timed_coro("upsert.text_chunks", self.text_chunks.upsert, chunks,
                                                       _tags={"doc_id": doc_id})
                                        )

                                        # 汇总第一阶段所有并行任务
                                        # Track first-stage tasks (parallel execution)
                                        first_stage_tasks = [
                                            doc_status_task,
                                            chunks_vdb_task,
                                            full_docs_task,
                                            text_chunks_task,
                                        ]

                                        # 并发等待第一阶段所有任务完成（统计总耗时，非任务耗时之和）
                                        # Await the first stage as a group (wall-clock; not the sum of subtasks)
                                        async with stage("stage1_upserts_gather", doc_id=doc_id):
                                            await asyncio.gather(*first_stage_tasks)

                                        # Stage2：文本分块持久化后执行实体关系抽取，保证数据依赖可用
                                        # Stage 2: Entity/relationship extraction (after text_chunks are saved)
                                        async with stage("extract_entities_all_chunks", doc_id=doc_id,
                                                         chunks=len(chunks)):
                                            # 创建并执行实体关系抽取任务
                                            entity_relation_task = asyncio.create_task(
                                                self._process_entity_relation_graph(
                                                    chunks, pipeline_status, pipeline_status_lock
                                                )
                                            )
                                            await entity_relation_task
                                        # 标记抽取阶段完成
                                        file_extraction_stage_ok = True

                                    except Exception as e:
                                        # 异常处理：打印堆栈、记录错误消息、取消未完成任务
                                        # Log error and update pipeline status
                                        logger.error(traceback.format_exc())
                                        error_msg = f"Failed to extract document {current_file_number}/{total_files}: {file_path}"
                                        logger.error(error_msg)
                                        async with pipeline_status_lock:
                                            pipeline_status["latest_message"] = error_msg
                                            pipeline_status["history_messages"].append(
                                                traceback.format_exc()
                                            )
                                            pipeline_status["history_messages"].append(error_msg)

                                            # 取消所有未完成的异步任务，释放资源
                                            # Cancel tasks that are not completed yet
                                            all_tasks = first_stage_tasks + (
                                                [entity_relation_task] if entity_relation_task else []
                                            )
                                            for task in all_tasks:
                                                if task and not task.done():
                                                    task.cancel()

                                        # 异常时持久化LLM缓存，避免缓存丢失
                                        # Persist LLM cache if available
                                        if self.llm_response_cache:
                                            await self.llm_response_cache.index_done_callback()

                                        # 将文档状态标记为失败，记录错误信息
                                        # Mark the document as failed
                                        await self.doc_status.upsert(
                                            {
                                                doc_id: {
                                                    "status": DocStatus.FAILED,
                                                    "error": str(e),
                                                    "content": status_doc.content,
                                                    "content_summary": status_doc.content_summary,
                                                    "content_length": status_doc.content_length,
                                                    "created_at": status_doc.created_at,
                                                    "updated_at": datetime.now(
                                                        timezone.utc
                                                    ).isoformat(),
                                                    "file_path": file_path,
                                                }
                                            }
                                        )

                                    # 抽取阶段成功后，执行合并与可选的多轮精修
                                    # After extraction succeeds, perform merge and optional refinement
                                    if file_extraction_stage_ok:
                                        try:
                                            # 获取实体关系抽取结果
                                            # Retrieve results from the extraction task
                                            chunk_results = await entity_relation_task

                                            # 合并抽取结果：节点/边存入知识图谱，同步向量索引
                                            # Merge nodes/edges into graph and vector stores
                                            async with stage("merge_nodes_and_edges", doc_id=doc_id):
                                                await merge_nodes_and_edges(
                                                    chunk_results=chunk_results,
                                                    knowledge_graph_inst=self.chunk_entity_relation_graph,
                                                    entity_vdb=self.entities_vdb,
                                                    relationships_vdb=self.relationships_vdb,
                                                    global_config=asdict(self),
                                                    pipeline_status=pipeline_status,
                                                    pipeline_status_lock=pipeline_status_lock,
                                                    llm_response_cache=self.llm_response_cache,
                                                    current_file_number=current_file_number,
                                                    total_files=total_files,
                                                    file_path=file_path,
                                                )

                                            # 可选：多轮精修（再抽取+再合并），提升知识图谱质量
                                            # Optional: iterative refinement rounds
                                            for r in range(self.addon_params['entity_refine_max_times']):
                                                await self._insert_refine_done(refine_round=r)

                                                async with stage(f"refine_round_{r}", round=r, doc_id=doc_id):
                                                    # 执行一轮精修处理
                                                    chunk_results = await process_document_refinement(
                                                        doc_id=doc_id,
                                                        status_doc=status_doc,
                                                        chunks=chunks,
                                                        chunk_results=chunk_results,
                                                        knowledge_graph_inst=self.chunk_entity_relation_graph,
                                                        entity_vdb=self.entities_vdb,
                                                        relationships_vdb=self.relationships_vdb,
                                                        global_config=asdict(self),
                                                        pipeline_status=pipeline_status,
                                                        pipeline_status_lock=pipeline_status_lock,
                                                        llm_response_cache=self.llm_response_cache,
                                                        current_file_number=current_file_number,
                                                        total_files=total_files,
                                                        file_path=file_path,
                                                    )

                                            # 最终更新文档状态为处理完成
                                            # Final status update for the document
                                            async with stage("finalize_doc_status", doc_id=doc_id):
                                                await self.doc_status.upsert(
                                                    {
                                                        doc_id: {
                                                            "status": DocStatus.PROCESSED,
                                                            "chunks_count": len(chunks),
                                                            "chunks_list": list(
                                                                chunks.keys()
                                                            ),  # 保留chunk列表用于调试/追溯
                                                            "content": status_doc.content,
                                                            "content_summary": status_doc.content_summary,
                                                            "content_length": status_doc.content_length,
                                                            "created_at": status_doc.created_at,
                                                            "updated_at": datetime.now(
                                                                timezone.utc
                                                            ).isoformat(),
                                                            "file_path": file_path,
                                                        }
                                                    }
                                                )

                                            # 触发插入完成回调，刷新内存索引/状态
                                            # Fire post-insert hook
                                            async with stage("_insert_done", doc_id=doc_id):
                                                await self._insert_done()

                                            # 记录处理完成日志
                                            async with pipeline_status_lock:
                                                log_message = f"Completed processing file {current_file_number}/{total_files}: {file_path}"
                                                logger.info(log_message)
                                                pipeline_status["latest_message"] = log_message
                                                pipeline_status["history_messages"].append(
                                                    log_message
                                                )

                                        except Exception as e:
                                            # 合并/精修阶段异常处理
                                            # Log error and update pipeline status
                                            logger.error(traceback.format_exc())
                                            error_msg = f"Merging stage failed in document {current_file_number}/{total_files}: {file_path}"
                                            logger.error(error_msg)
                                            async with pipeline_status_lock:
                                                pipeline_status["latest_message"] = error_msg
                                                pipeline_status["history_messages"].append(
                                                    traceback.format_exc()
                                                )
                                                pipeline_status["history_messages"].append(
                                                    error_msg
                                                )

                                            # 持久化LLM缓存
                                            # Persist LLM cache if available
                                            if self.llm_response_cache:
                                                await self.llm_response_cache.index_done_callback()

                                            # 标记文档处理失败
                                            # Mark the document as failed
                                            await self.doc_status.upsert(
                                                {
                                                    doc_id: {
                                                        "status": DocStatus.FAILED,
                                                        "error": str(e),
                                                        "content": status_doc.content,
                                                        "content_summary": status_doc.content_summary,
                                                        "content_length": status_doc.content_length,
                                                        "created_at": status_doc.created_at,
                                                        "updated_at": datetime.now().isoformat(),
                                                        "file_path": file_path,
                                                    }
                                                }
                                            )

                        async def process_document_refinement(
                                doc_id: str,
                                status_doc: DocProcessingStatus,
                                chunks: dict[str, Any],
                                chunk_results: list,
                                knowledge_graph_inst: BaseGraphStorage,
                                entity_vdb: BaseVectorStorage,
                                relationships_vdb: BaseVectorStorage,
                                global_config: dict[str, str],
                                pipeline_status: dict = None,
                                pipeline_status_lock=None,
                                llm_response_cache: BaseKVStorage | None = None,
                                current_file_number: int = 0,
                                total_files: int = 0,
                                file_path: str = "unknown_source",
                        ):
                            """
                            功能说明：
                                处理单个文档的一轮精修抽取与合并
                                作用：对已抽取的实体关系进行优化，提升知识图谱准确性

                            参数：
                                - doc_id (str)：文档唯一标识
                                - status_doc (DocProcessingStatus)：文档处理状态对象
                                - chunks (dict[str, Any])：文本分块集合，键为chunk_id
                                - chunk_results (list)：实体关系抽取的中间/最终结果
                                - knowledge_graph_inst (BaseGraphStorage)：知识图谱存储实例
                                - entity_vdb (BaseVectorStorage)：实体向量存储实例
                                - relationships_vdb (BaseVectorStorage)：关系向量存储实例
                                - global_config (dict[str, str])：全局运行配置
                                - pipeline_status (dict)：流水线共享状态字典
                                - pipeline_status_lock (Any)：保护流水线状态的锁
                                - llm_response_cache (BaseKVStorage | None)：LLM响应缓存实例
                                - current_file_number (int)：当前处理文件序号
                                - total_files (int)：批次文件总数
                                - file_path (str)：文档文件路径

                            返回：
                                Any：返回精修后的实体关系结果，无异常则返回有效数据
                            """
                            try:
                                # 精修任务与状态标记
                                entity_relation_refine_task = None
                                file_extraction_stage_ok = False
                                # 创建精修抽取任务
                                entity_relation_refine_task = asyncio.create_task(
                                    self._process_entity_relation_graph_refinement(
                                        chunks,
                                        chunk_results,
                                        knowledge_graph_inst,
                                        entity_vdb,
                                        relationships_vdb,
                                        global_config,
                                        llm_response_cache,
                                        pipeline_status,
                                        pipeline_status_lock
                                    )
                                )
                                file_extraction_stage_ok = True

                                # 抽取成功则合并精修结果
                                if file_extraction_stage_ok:
                                    try:
                                        # 等待精修抽取结果
                                        # Wait for refined extraction results
                                        chunk_results = await entity_relation_refine_task

                                        # 合并精修后的节点与边
                                        # Merge refined results
                                        async with stage("merge_nodes_and_edges_refine", file_path=file_path):
                                            await merge_nodes_and_edges(
                                                chunk_results=chunk_results,
                                                knowledge_graph_inst=knowledge_graph_inst,
                                                entity_vdb=entity_vdb,
                                                relationships_vdb=relationships_vdb,
                                                global_config=global_config,
                                                pipeline_status=pipeline_status,
                                                pipeline_status_lock=pipeline_status_lock,
                                                llm_response_cache=llm_response_cache,
                                                current_file_number=current_file_number,
                                                total_files=total_files,
                                                file_path=file_path,
                                            )
                                        # 返回精修后的结果
                                        return chunk_results
                                    except Exception as e:
                                        # 精修合并阶段异常处理
                                        # Log error and update pipeline status
                                        logger.error(traceback.format_exc())
                                        error_msg = f"Merging (refinement) stage failed in document {current_file_number}/{total_files}: {file_path}"
                                        logger.error(error_msg)
                                        async with pipeline_status_lock:
                                            pipeline_status["latest_message"] = error_msg
                                            pipeline_status["history_messages"].append(
                                                traceback.format_exc()
                                            )
                                            pipeline_status["history_messages"].append(
                                                error_msg
                                            )

                                        # 持久化缓存
                                        # Persist LLM cache if available
                                        if self.llm_response_cache:
                                            await self.llm_response_cache.index_done_callback()

                                        # 标记文档失败
                                        # Mark the document as failed
                                        await self.doc_status.upsert(
                                            {
                                                doc_id: {
                                                    "status": DocStatus.FAILED,
                                                    "error": str(e),
                                                    "content": status_doc.content,
                                                    "content_summary": status_doc.content_summary,
                                                    "content_length": status_doc.content_length,
                                                    "created_at": status_doc.created_at,
                                                    "updated_at": datetime.now().isoformat(),
                                                    "file_path": file_path,
                                                }
                                            }
                                        )

                            except Exception as e:
                                # 精修抽取阶段异常处理
                                # Log error and update pipeline status
                                logger.error(traceback.format_exc())
                                error_msg = f"Failed to extract document {current_file_number}/{total_files}: {file_path}"
                                logger.error(error_msg)
                                async with pipeline_status_lock:
                                    pipeline_status["latest_message"] = error_msg
                                    pipeline_status["history_messages"].append(
                                        traceback.format_exc()
                                    )
                                    pipeline_status["history_messages"].append(error_msg)

                                    # 取消未完成任务
                                    # Cancel tasks that are not completed yet
                                    all_tasks = (
                                        [entity_relation_refine_task]
                                        if entity_relation_refine_task
                                        else []
                                    )
                                    for task in all_tasks:
                                        if task and not task.done():
                                            task.cancel()

                                # 持久化缓存
                                # Persist LLM cache if available
                                if self.llm_response_cache:
                                    await self.llm_response_cache.index_done_callback()

                                # 标记文档失败
                                # Mark the document as failed
                                await self.doc_status.upsert(
                                    {
                                        doc_id: {
                                            "status": DocStatus.FAILED,
                                            "error": str(e),
                                            "content": status_doc.content,
                                            "content_summary": status_doc.content_summary,
                                            "content_length": status_doc.content_length,
                                            "created_at": status_doc.created_at,
                                            "updated_at": datetime.now(
                                                timezone.utc
                                            ).isoformat(),
                                            "file_path": file_path,
                                        }
                                    }
                                )

                        # 为当前批次所有文档创建处理任务，并发执行
                        # Create processing tasks for all documents
                        doc_tasks = []
                        for doc_id, status_doc in to_process_docs.items():
                            doc_tasks.append(
                                process_document(
                                    doc_id,
                                    status_doc,
                                    split_by_character,
                                    split_by_character_only,
                                    split_by_page,
                                    pipeline_status,
                                    pipeline_status_lock,
                                    semaphore,
                                )
                            )

                        # 等待当前批次所有文档处理完成
                        # Wait for all document processing to complete
                        await asyncio.gather(*doc_tasks)

                        # 检查处理期间是否收到新的文档入队请求，有则继续循环处理
                        # Check if there's a pending request to process more documents (with lock)
                        has_pending_request = False
                        async with pipeline_status_lock:
                            has_pending_request = pipeline_status.get("request_pending", False)
                            if has_pending_request:
                                # 清除请求标记，准备处理新文档
                                # Clear the request flag before checking for more documents
                                pipeline_status["request_pending"] = False

                        # 无新请求，退出循环
                        if not has_pending_request:
                            break

                        log_message = "Processing additional documents due to pending request"
                        logger.info(log_message)
                        pipeline_status["latest_message"] = log_message
                        pipeline_status["history_messages"].append(log_message)

                        # 重新获取待处理文档（处理中+失败+待处理）
                        # Check again for pending documents
                        processing_docs, failed_docs, pending_docs = await asyncio.gather(
                            self.doc_status.get_docs_by_status(DocStatus.PROCESSING),
                            self.doc_status.get_docs_by_status(DocStatus.FAILED),
                            self.doc_status.get_docs_by_status(DocStatus.PENDING),
                        )

                        # 重新赋值待处理文档集合
                        to_process_docs = {}
                        to_process_docs.update(processing_docs)
                        to_process_docs.update(failed_docs)
                        to_process_docs.update(pending_docs)
            finally:
                # 最终：移除进度监听器，释放资源
                remove_stage_listener(sp.handle)
        # ---------------------------------------------------------------------

        # 流水线全部处理完成，打印日志
        log_message = "Document processing pipeline completed"
        logger.info(log_message)
        # 无论正常结束/异常退出，都重置忙碌状态（加锁保证安全）
        # Always reset busy status when done or if an exception occurs (with lock)
        async with pipeline_status_lock:
            pipeline_status["busy"] = False
            pipeline_status["latest_message"] = log_message
            pipeline_status["history_messages"].append(log_message)

        # 尝试生成性能瀑布图，辅助定位处理瓶颈；失败不影响主流程
        # Optional: produce a waterfall chart for the latest trace
        try:
            # 拼接日志文件路径
            tim_log_dir = os.path.join(self.working_dir, self.workspace, "logs/timings.jsonl")
            wat_log_dir = os.path.join(self.working_dir, self.workspace, "logs/waterfall.png")
            # 生成瀑布图
            plot_waterfall_from_jsonl(tim_log_dir, outfile=wat_log_dir)
        except Exception as _:
            # 生成图表失败，忽略异常
            pass

    async def _insert_refine_done(
        self, refine_round
    ) -> None:
        """
        功能说明：
            在精修轮次完成后执行落盘与回调。

        参数：
            - self (Any)：当前类实例本身。
            - refine_round (Any)：方法执行所需输入参数。

        返回：
            None：方法执行结果；若为 None 表示主要通过副作用完成处理。
        """
        # 将当前内存索引快照到“精修轮次目录”，便于对比与回滚分析。
        new_filename2old_filename = {}
        work_dir = os.path.join(self.working_dir, self.workspace, 'iterative_refinement', f'r{refine_round}')
        os.makedirs(work_dir, exist_ok=True)

        def _save_vdb(storage_inst):
            """
            功能说明：
                将向量数据库索引状态持久化到存储。

            参数：
                - storage_inst (Any)：方法执行所需输入参数。

            返回：
                Any：方法执行结果；若为 None 表示主要通过副作用完成处理。
            """
            # 临时重定向存储文件路径，触发 index_done_callback 完成落盘。
            _client_file_name = storage_inst._client_file_name
            client_file_name  = os.path.join(
                work_dir, f"vdb_{storage_inst.namespace}.json"
            )
            new_filename2old_filename[client_file_name] = _client_file_name
            storage_inst._client.storage_file = client_file_name
            return storage_inst.index_done_callback()

        tasks = [
            _save_vdb(cast(StorageNameSpace, storage_inst))
            for storage_inst in [  # type: ignore
                self.entities_vdb,
                self.relationships_vdb,
                self.chunks_vdb,
            ]
            if storage_inst is not None
        ]
        await asyncio.gather(*tasks)

        for storage_inst in [
            self.entities_vdb,
            self.relationships_vdb,
            self.chunks_vdb,
        ]:
            storage_inst._client.storage_file = new_filename2old_filename[
                storage_inst._client.storage_file
            ]

        # 图数据库以 GraphML 形式单独持久化。
        # Persist graphml for the graph namespace
        storage_inst = cast(StorageNameSpace, self.chunk_entity_relation_graph)
        _graphml_xml_file = storage_inst._graphml_xml_file
        graphml_xml_file = os.path.join(
            work_dir, f"graph_{storage_inst.namespace}.graphml"
        )
        storage_inst._graphml_xml_file = graphml_xml_file
        await storage_inst.index_done_callback()
        storage_inst._graphml_xml_file = _graphml_xml_file

        log_message = "In memory DB persist to disk"
        logger.info(log_message)

    async def aquery(
        self,
        query: str,
        param: QueryParam = QueryParam(),
        system_prompt: str | None = None,
    ) -> str | AsyncIterator[str]:
        """
        功能说明：
            异步查询入口，根据策略路由到对应检索/图谱流程。

        参数：
            - self (Any)：当前类实例本身。
            - query (str)：用户输入的查询问题。
            - param (QueryParam)：方法执行所需输入参数。
            - system_prompt (str | None)：方法执行所需输入参数。

        返回：
            str | AsyncIterator[str]：方法执行结果；若为 None 表示主要通过副作用完成处理。
        """
        # 每次查询生成独立的追踪ID，便于问题追踪与性能分析
        trace_id_var.set(uuid.uuid4().hex)

        # 进入总查询阶段，记录整体查询耗时与模式
        async with stage("query_total", mode=param.mode or "default"):
            # 将当前实例转为字典，获取全局配置
            global_config = asdict(self)
            # 保存原始用户查询，用于后续向量检索
            param.original_query = query

            # 根据mode路由到知识图谱相关查询流程
            if param.mode in ["local", "global", "hybrid", "mix"]:
                async with stage("kg_query"):
                    response = await kg_query(
                        query.strip(),
                        self.chunk_entity_relation_graph,
                        self.entities_vdb,
                        self.relationships_vdb,
                        self.text_chunks,
                        param,
                        global_config,
                        hashing_kv=self.llm_response_cache,
                        system_prompt=system_prompt,
                        chunks_vdb=self.chunks_vdb,
                    )
            # 朴素检索模式：仅基于文本块向量检索
            elif param.mode == "naive":
                async with stage("naive_query"):
                    response = await naive_query(
                        query.strip(),
                        self.chunks_vdb,
                        self.text_chunks,
                        param,
                        global_config,
                        hashing_kv=self.llm_response_cache,
                        system_prompt=system_prompt,
                    )
            # 直通LLM模式：不做知识检索，直接调用大模型
            elif param.mode == "bypass":
                # 使用指定的LLM方法，未指定则使用全局默认方法
                use_llm_func = param.model_func or global_config["llm_model_func"]
                # 为当前LLM调用设置优先级
                use_llm_func = partial(use_llm_func, _priority=8)

                # 未指定流式输出时，默认开启
                param.stream = True if param.stream is None else param.stream
                async with stage("bypass_llm", stream=param.stream):
                    response = await use_llm_func(
                        query.strip(),
                        system_prompt=system_prompt,
                        history_messages=param.conversation_history,
                        stream=param.stream,
                    )
            # 知识图谱两步查询模式
            elif param.mode == "mix_two_step":
                async with stage("kg_two_step_query"):
                    response = await kg_two_step_query(
                        query.strip(),
                        self.chunk_entity_relation_graph,
                        self.entities_vdb,
                        self.relationships_vdb,
                        self.text_chunks,
                        param,
                        global_config,
                        hashing_kv=self.llm_response_cache,
                        system_prompt=system_prompt,
                        chunks_vdb=self.chunks_vdb,
                    )
            # 结构触发 + 多智能体辩论 + 裁决 的检索增强模式
            elif param.mode == "mmkg_debate":
                async with stage("kg_debate_query"):
                    response = await kg_debate_query(
                        query.strip(),
                        self.chunk_entity_relation_graph,
                        self.entities_vdb,
                        self.relationships_vdb,
                        self.text_chunks,
                        param,
                        global_config,
                        hashing_kv=self.llm_response_cache,
                        system_prompt=system_prompt,
                        chunks_vdb=self.chunks_vdb,
                    )
            # 未知模式，抛出异常
            else:
                raise ValueError(f"Unknown mode {param.mode}")

        # 查询完成后执行收尾操作
        await self._query_done()
        # 返回最终响应结果
        return response
