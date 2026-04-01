import re
import os
import json
import time
import asyncio

from collections import (
    Counter, 
    defaultdict
)
from functools import partial

from lightrag.base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    TextChunkSchema,
    QueryParam,
)

from lightrag.utils import (
    logger,
    clean_str,
    compute_mdhash_id,
    Tokenizer,
    is_float_regex,
    normalize_extracted_info,
    pack_user_ass_to_openai_messages,
    split_string_by_multi_markers,
    truncate_list_by_token_size,
    process_combine_contexts,
    compute_args_hash,
    handle_cache,
    save_to_cache,
    CacheData,
    get_conversation_turns,
    # use_llm_func_with_cache,
    update_chunk_cache_list,
    remove_think_tags,
)

from lightrag.kg.shared_storage import get_storage_keyed_lock
from lightrag.operate import (
    _merge_nodes_then_upsert,
    _merge_edges_then_upsert,
    _build_query_context,
    _get_node_data,
    _get_edge_data,
    _find_most_related_text_unit_from_entities,
    _find_related_text_unit_from_relationships,
    process_chunks_unified,
    get_keywords_from_query,
)

from lightrag.constants import (
    GRAPH_FIELD_SEP,
    DEFAULT_MAX_ENTITY_TOKENS,
    DEFAULT_MAX_RELATION_TOKENS,
    DEFAULT_MAX_TOTAL_TOKENS,
    DEFAULT_RELATED_CHUNK_NUMBER,
)

from megarag.prompt import PROMPTS
from megarag.utils import use_llm_func_with_cache

from typing import (
    Any, 
    AsyncIterator
)

def chunking_by_token_size(
    tokenizer: Tokenizer,
    content: str,
    split_by_character: str | None = None,
    split_by_character_only: bool = False,
    overlap_token_size: int = 128,
    max_token_size: int = 1024,
) -> list[dict[str, Any]]:
    """
    功能说明：
        按 token 数量对原始文本进行分块，便于后续向量化与检索。
    
    参数：
        - tokenizer (Tokenizer)：方法执行所需输入参数。
        - content (str)：待处理的文本内容。
        - split_by_character (str | None)：按指定字符切分文本；为 None 时不启用该规则。
        - split_by_character_only (bool)：是否仅按字符切分文本。
        - overlap_token_size (int)：方法执行所需输入参数。
        - max_token_size (int)：方法执行所需输入参数。
    
    返回：
        list[dict[str, Any]]：方法执行结果；若为 None 表示主要通过副作用完成处理。
    """
    # 先整体编码，后续按窗口切片并保留重叠区，降低语义断裂风险。
    tokens = tokenizer.encode(content)
    results: list[dict[str, Any]] = []
    # 优先按字符切分（如段落/换行），再视情况补充 token 级切分。
    if split_by_character:
        raw_chunks = content.split(split_by_character)
        new_chunks = []
        if split_by_character_only:
            # 仅按字符切，不再做超长片段二次切分。
            for chunk in raw_chunks:
                _tokens = tokenizer.encode(chunk)
                new_chunks.append((len(_tokens), chunk))
        else:
            # 字符切后若片段仍过长，再按 token 窗口二次切分。
            for chunk in raw_chunks:
                _tokens = tokenizer.encode(chunk)
                if len(_tokens) > max_token_size:
                    for start in range(
                        0, len(_tokens), max_token_size - overlap_token_size
                    ):
                        chunk_content = tokenizer.decode(
                            _tokens[start : start + max_token_size]
                        )
                        new_chunks.append(
                            (min(max_token_size, len(_tokens) - start), chunk_content)
                        )
                else:
                    new_chunks.append((len(_tokens), chunk))
        for index, (_len, chunk) in enumerate(new_chunks):
            results.append(
                {
                    "tokens": _len,
                    "content": chunk.strip(),
                    "chunk_order_index": index,
                }
            )
    else:
        # 未指定字符切分时，直接走纯 token 窗口分块。
        for index, start in enumerate(
            range(0, len(tokens), max_token_size - overlap_token_size)
        ):
            chunk_content = tokenizer.decode(tokens[start : start + max_token_size])
            results.append(
                {
                    "tokens": min(max_token_size, len(tokens) - start),
                    "content": chunk_content.strip(),
                    "chunk_order_index": index,
                }
            )
    return results

def chunking_by_token_or_page(
    tokenizer: Tokenizer,
    content: str,
    split_by_character: str | None = None,
    split_by_character_only: bool = False,
    split_by_page: bool = True,
    overlap_token_size: int = 128,
    max_token_size: int = 1024,
) -> list[dict[str, Any]]:
    """
    功能说明：
        根据 token 或分页信息进行分块，兼顾语义完整性与分页边界。
    
    参数：
        - tokenizer (Tokenizer)：方法执行所需输入参数。
        - content (str)：待处理的文本内容。
        - split_by_character (str | None)：按指定字符切分文本；为 None 时不启用该规则。
        - split_by_character_only (bool)：是否仅按字符切分文本。
        - split_by_page (bool)：是否按分页边界切分文本。
        - overlap_token_size (int)：方法执行所需输入参数。
        - max_token_size (int)：方法执行所需输入参数。
    
    返回：
        list[dict[str, Any]]：方法执行结果；若为 None 表示主要通过副作用完成处理。
    """
    results: list[dict[str, Any]] = []
    # 分页模式用于多页文档（如 PDF 转 JSON）并携带页图/配图元数据。
    if split_by_page:
        try:
            content = json.loads(content)
        except json.JSONDecodeError as err:
            msg = "Using [split_by_page] requires the content argument to be valid JSON."
            logger.error(msg)
            raise ValueError(msg) from err
        
        # 逐页构建 chunk，空页跳过，仅图片页补默认文字提示。
        for index in range(len(content)):
            page_img = content[str(index)]['page_image']
            fig_imgs = content[str(index)]['figure_images']
            text     = content[str(index)]['text']
            
            # Skip the empty page.
            if len(fig_imgs) == 0 and len(text) == 0:
                continue
            elif len(text) == 0:
                text = "Please see the Figures."

            # encode 把文本变成模型语言
            # decode 再变回人类语言
            # 合起来 标准化清洗文本
            tokens = tokenizer.encode(text)
            chunk_content = tokenizer.decode(tokens)
            results.append(
                {
                    "tokens": len(tokens),
                    "content": chunk_content.strip(),
                    "page_img": page_img,
                    "fig_imgs": fig_imgs,
                    "chunk_order_index": index,
                }
            )
    else:
        # 非分页模式回退到通用 token 分块逻辑。
        results = chunking_by_token_size(
            tokenizer,
            content,
            split_by_character,
            split_by_character_only,
            overlap_token_size,
            max_token_size
        )
    return results

async def _get_vector_context(
    query: str,
    chunks_vdb: BaseVectorStorage,
    query_param: QueryParam,
) -> list[dict]:
    """
    功能说明：
        从向量库检索与当前查询相关的上下文片段。
    
    参数：
        - query (str)：用户输入的查询问题。
        - chunks_vdb (BaseVectorStorage)：方法执行所需输入参数。
        - query_param (QueryParam)：方法执行所需输入参数。
    
    返回：
        list[dict]：方法执行结果；若为 None 表示主要通过副作用完成处理。
    """
    try:
        # 优先使用 chunk_top_k（若配置），否则退回 top_k。
        # Use chunk_top_k if specified, otherwise fall back to top_k
        search_top_k = query_param.chunk_top_k or query_param.top_k
        results = await chunks_vdb.query(query, top_k=search_top_k, ids=query_param.ids)
        if not results:
            return []

        # 仅保留包含 content 的有效命中，并补齐来源元数据。
        valid_chunks = []
        for result in results:
            if "content" in result:
                chunk_with_metadata = {
                    "content": result["content"],
                    "created_at": result.get("created_at", None),
                    "file_path": result.get("file_path", "unknown_source"),
                    "id": result.get("id", []),
                    "source_type": "vector",  # Mark the source type
                }
                valid_chunks.append(chunk_with_metadata)

        logger.info(
            f"Naive query: {len(valid_chunks)} chunks (chunk_top_k: {search_top_k})"
        )
        return valid_chunks

    except Exception as e:
        logger.error(f"Error in _get_vector_context: {e}")
        return []
async def _build_query_context_for_refine(
    query: str,
    ll_keywords: str,
    hl_keywords: str,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage,
    query_param: QueryParam,
    chunks_vdb: BaseVectorStorage = None,
):
    """
    功能说明：
        为【实体抽取精修阶段】构建高质量提示词上下文。
        从知识图谱、实体向量库、关系向量库、文本分块中检索相关信息，
        按 Token 预算裁剪，最终生成 LLM 可直接使用的 KG + 文本上下文。
        是精修抽取能否准确的核心函数。

    参数：
        - query (str)：查询语句（实体关键词拼接）
        - ll_keywords (str)：实体级关键词（低级关键词）
        - hl_keywords (str)：关系级关键词（高级关键词）
        - knowledge_graph_inst (BaseGraphStorage)：知识图谱存储实例
        - entities_vdb (BaseVectorStorage)：实体向量库
        - relationships_vdb (BaseVectorStorage)：关系向量库
        - text_chunks_db (BaseKVStorage)：文本分块键值存储
        - query_param (QueryParam)：查询参数（模式、Token 上限、TopK）
        - chunks_vdb (BaseVectorStorage, optional)：文本分块向量库

    返回：
        str | None：格式化好的上下文字符串（实体+关系+文本），无数据则返回 None
    """
    logger.info(f"Process {os.getpid()} building query context...")

    # ===================== 初始化各类上下文容器 =====================
    # 统一收集来自向量检索与图检索的所有上下文，后续统一做 Token 裁剪
    all_chunks = []                # 文本分块
    entities_context = []          # 实体上下文
    relations_context = []         # 关系上下文

    # 保存原始数据，用于后续反查对应的文本分块
    original_node_datas = []
    original_edge_datas = []

    # ===================== 根据查询模式获取子图数据 =====================
    # local  = 只查实体（节点）
    # global = 只查关系（边）
    # hybrid/mix = 同时查实体+关系
    if query_param.mode == "local":
        # 从实体向量库 + 图谱获取局部节点数据
        (
            entities_context,
            relations_context,
            node_datas,
            use_relations,
        ) = await _get_node_data(
            ll_keywords,
            knowledge_graph_inst,
            entities_vdb,
            query_param,
        )
        original_node_datas = node_datas
        original_edge_datas = use_relations

    elif query_param.mode == "global":
        # 从关系向量库 + 图谱获取全局边数据
        (
            entities_context,
            relations_context,
            edge_datas,
            use_entities,
        ) = await _get_edge_data(
            hl_keywords,
            knowledge_graph_inst,
            relationships_vdb,
            query_param,
        )
        original_edge_datas = edge_datas
        original_node_datas = use_entities

    else:  # hybrid 或 mix 模式
        # 并行获取实体 + 关系数据
        ll_data = await _get_node_data(
            ll_keywords,
            knowledge_graph_inst,
            entities_vdb,
            query_param,
        )
        hl_data = await _get_edge_data(
            hl_keywords,
            knowledge_graph_inst,
            relationships_vdb,
            query_param,
        )

        (ll_entities_context, ll_relations_context, ll_node_datas, ll_edge_datas) = ll_data
        (hl_entities_context, hl_relations_context, hl_edge_datas, hl_node_datas) = hl_data

        # mix 模式额外从向量库检索文本块
        if query_param.mode == "mix" and chunks_vdb:
            vector_chunks = await _get_vector_context(
                query,
                chunks_vdb,
                query_param,
            )
            all_chunks.extend(vector_chunks)

        # 合并来自实体与关系检索的原始数据
        original_node_datas = ll_node_datas + hl_node_datas
        original_edge_datas = ll_edge_datas + hl_edge_datas

        # 合并实体与关系上下文
        entities_context = process_combine_contexts(
            ll_entities_context, hl_entities_context
        )
        relations_context = process_combine_contexts(
            hl_relations_context, ll_relations_context
        )

    logger.info(
        f"Initial context: {len(entities_context)} entities, {len(relations_context)} relations, {len(all_chunks)} chunks"
    )

    # ===================== Token 预算控制：裁剪实体与关系 =====================
    # 如果有分词器，按 Token 上限裁剪实体、关系，防止提示词过长
    tokenizer = text_chunks_db.global_config.get("tokenizer")
    if tokenizer:
        # 从配置中读取各类 Token 上限
        max_entity_tokens = getattr(
            query_param,
            "max_entity_tokens",
            text_chunks_db.global_config.get(
                "max_entity_tokens", DEFAULT_MAX_ENTITY_TOKENS
            ),
        )
        max_relation_tokens = getattr(
            query_param,
            "max_relation_tokens",
            text_chunks_db.global_config.get(
                "max_relation_tokens", DEFAULT_MAX_RELATION_TOKENS
            ),
        )
        max_total_tokens = getattr(
            query_param,
            "max_total_tokens",
            text_chunks_db.global_config.get(
                "max_total_tokens", DEFAULT_MAX_TOTAL_TOKENS
            ),
        )

        # -------------------- 裁剪实体 --------------------
        if entities_context:
            original_entity_count = len(entities_context)

            # 清理文件路径中的特殊分隔符
            for entity in entities_context:
                if "file_path" in entity and entity["file_path"]:
                    entity["file_path"] = entity["file_path"].replace(
                        GRAPH_FIELD_SEP, ";"
                    )
            # 精简实体字段，减少 Token 占用
            concise_entities_context = []
            for entity in entities_context:
                concise_entities_context.append({
                    'entity'     : entity['entity'],
                    'type'       : entity['type'],
                    'description': entity['description'],
                })
            # 按 Token 长度截断
            entities_context = truncate_list_by_token_size(
                concise_entities_context,
                key=lambda x: json.dumps(x, ensure_ascii=False),
                max_token_size=max_entity_tokens,
                tokenizer=tokenizer,
            )
            if len(entities_context) < original_entity_count:
                logger.debug(
                    f"Truncated entities: {original_entity_count} -> {len(entities_context)} (entity max tokens: {max_entity_tokens})"
                )

        # -------------------- 裁剪关系 --------------------
        if relations_context:
            original_relation_count = len(relations_context)

            # 清理文件路径中的特殊分隔符
            for relation in relations_context:
                if "file_path" in relation and relation["file_path"]:
                    relation["file_path"] = relation["file_path"].replace(
                        GRAPH_FIELD_SEP, ";"
                    )
            # 精简关系字段
            concise_relations_context = []
            for relation in relations_context:
                concise_relations_context.append({
                    'source_entity': relation['entity1'],
                    'target_entity': relation['entity2'],
                    'description': relation['description'],
                })
            # 按 Token 长度截断
            relations_context = truncate_list_by_token_size(
                concise_relations_context,
                key=lambda x: json.dumps(x, ensure_ascii=False),
                max_token_size=max_relation_tokens,
                tokenizer=tokenizer,
            )
            if len(relations_context) < original_relation_count:
                logger.debug(
                    f"Truncated relations: {original_relation_count} -> {len(relations_context)} (relation max tokens: {max_relation_tokens})"
                )

    # ===================== 根据裁剪后的实体/关系 反查文本块 =====================
    logger.info("Getting text chunks based on truncated entities and relations...")

    # 过滤出最终保留的实体
    final_node_datas = []
    if entities_context and original_node_datas:
        final_entity_names = {e["entity"] for e in entities_context}
        seen_nodes = set()
        for node in original_node_datas:
            name = node.get("entity_name")
            if name in final_entity_names and name not in seen_nodes:
                final_node_datas.append(node)
                seen_nodes.add(name)

    # 过滤出最终保留的关系
    final_edge_datas = []
    if relations_context and original_edge_datas:
        final_relation_pairs = {(r["source_entity"], r["target_entity"]) for r in relations_context}
        seen_edges = set()
        for edge in original_edge_datas:
            src, tgt = edge.get("src_id"), edge.get("tgt_id")
            if src is None or tgt is None:
                src, tgt = edge.get("src_tgt", (None, None))

            pair = (src, tgt)
            if pair in final_relation_pairs and pair not in seen_edges:
                final_edge_datas.append(edge)
                seen_edges.add(pair)

    # 并行获取实体 & 关系对应的原始文本分块
    text_chunk_tasks = []

    if final_node_datas:
        text_chunk_tasks.append(
            _find_most_related_text_unit_from_entities(
                final_node_datas,
                query_param,
                text_chunks_db,
                knowledge_graph_inst,
            )
        )

    if final_edge_datas:
        text_chunk_tasks.append(
            _find_related_text_unit_from_relationships(
                final_edge_datas,
                query_param,
                text_chunks_db,
            )
        )

    # 执行并行查询
    if text_chunk_tasks:
        text_chunk_results = await asyncio.gather(*text_chunk_tasks)
        for chunks in text_chunk_results:
            if chunks:
                all_chunks.extend(chunks)

    # 动态计算文本块可用 token，上下文总量受 max_total_tokens 约束。
    # Apply token processing to chunks if tokenizer is available
    text_units_context = []
    if tokenizer and all_chunks:
        # 先计算 KG 部分占用，再计算系统提示占用，剩余给 chunk。
        # Calculate dynamic token limit for text chunks
        entities_str = json.dumps(entities_context, ensure_ascii=False)
        relations_str = json.dumps(relations_context, ensure_ascii=False)

        # Calculate base context tokens (entities + relations + template)
        kg_context_template = """-----Entities(KG)-----

```json
{entities_str}
```

-----Relationships(KG)-----

```json
{relations_str}
```

-----Document Chunks(DC)-----

```json
[]
```

"""
        kg_context = kg_context_template.format(
            entities_str=entities_str, relations_str=relations_str
        )
        # 计算图谱上下文（实体+关系）的 token 长度，用于后续总长度控制
        kg_context_tokens = len(tokenizer.encode(kg_context))

        # 计算对话历史的 token 长度（如果有上下文对话）
        history_context = ""
        if query_param.conversation_history:
            history_context = get_conversation_turns(
                query_param.conversation_history, query_param.history_turns
            )
        history_tokens = (
            len(tokenizer.encode(history_context)) if history_context else 0
        )

        # 系统提示、回答格式、用户额外指令相关配置
        user_prompt = query_param.user_prompt if query_param.user_prompt else ""
        response_type = (
            query_param.response_type
            if query_param.response_type
            else "Multiple Paragraphs"
        )

        # 获取系统提示词模板（默认是 RAG 回答提示词）
        sys_prompt_template = text_chunks_db.global_config.get(
            "system_prompt_template", PROMPTS["rag_response"]
        )

        # 构造一个不含实际上下文的示例提示词，仅用于计算**模板本身的 token 开销**
        sample_sys_prompt = sys_prompt_template.format(
            history=history_context,
            context_data="",  # 上下文为空，只算模板基础长度
            response_type=response_type,
            user_prompt=user_prompt,
        )
        # 计算系统提示模板的 token 长度
        sys_prompt_template_tokens = len(tokenizer.encode(sample_sys_prompt))

        # 总系统提示开销 = 模板长度 + 用户问题长度
        query_tokens = len(tokenizer.encode(query))
        sys_prompt_overhead = sys_prompt_template_tokens + query_tokens

        # 安全缓冲区（防止计算后刚好超限）
        buffer_tokens = 100

        # 计算：总允许长度 - 已用长度 = 文本块还能使用多少 token
        used_tokens = kg_context_tokens + sys_prompt_overhead + buffer_tokens
        available_chunk_tokens = max_total_tokens - used_tokens

        logger.debug(
            f"Token allocation - Total: {max_total_tokens}, History: {history_tokens}, SysPrompt: {sys_prompt_overhead}, KG: {kg_context_tokens}, Buffer: {buffer_tokens}, Available for chunks: {available_chunk_tokens}"
        )

        # 根据动态计算出的剩余 token，对文本块做**二次截断**，保证不超限
        if all_chunks:
            # 精简文本块结构，只保留内容和路径
            temp_chunks = [
                {"content": chunk["content"], "file_path": chunk["file_path"]}
                for chunk in all_chunks
            ]

            # 按可用 token 长度统一裁剪文本块
            truncated_chunks = await process_chunks_unified(
                query=query,
                chunks=temp_chunks,
                query_param=query_param,
                global_config=text_chunks_db.global_config,
                source_type="mixed",
                chunk_token_limit=available_chunk_tokens,  # 使用动态计算的上限
            )

            # 重新构建最终文本块上下文
            for i, chunk in enumerate(truncated_chunks):
                text_units_context.append(
                    {
                        "id": i + 1,
                        "content": chunk["content"],
                        "file_path": chunk.get("file_path", "unknown_source"),
                    }
                )

            logger.debug(
                f"Re-truncated chunks for dynamic token limit: {len(temp_chunks)} -> {len(text_units_context)} (chunk available tokens: {available_chunk_tokens})"
            )

    # 输出最终构建的上下文规模日志
    logger.info(
        f"Final context: {len(entities_context)} entities, {len(relations_context)} relations, {len(text_units_context)} chunks"
    )

    # 如果实体和关系都为空，说明没有检索到任何内容，返回 None
    if not entities_context and not relations_context:
        return None

    # 将实体、关系、文本块转为 JSON 字符串
    entities_str = json.dumps(entities_context, ensure_ascii=False)
    relations_str = json.dumps(relations_context, ensure_ascii=False)
    text_units_str = json.dumps(text_units_context, ensure_ascii=False)

    result = f"""-----Entities(KG)-----

```json
{entities_str}
```

-----Relationships(KG)-----

```json
{relations_str}
```

-----Document Chunks(DC)-----

```json
{text_units_str}
```

"""
    return result


async def _build_query_context_with_image(
        query: str,
        ll_keywords: str,
        hl_keywords: str,
        knowledge_graph_inst: BaseGraphStorage,
        entities_vdb: BaseVectorStorage,
        relationships_vdb: BaseVectorStorage,
        text_chunks_db: BaseKVStorage,
        query_param: QueryParam,
        chunks_vdb: BaseVectorStorage = None,
):
    """
    功能说明：
        构建带多模态（图片）的知识图谱检索上下文，是 KG-RAG 的核心上下文构造函数。
        根据查询模式（local/global/hybrid/mix）检索实体、关系、文本块、图片，
        并做 Token 裁剪，最终返回可直接喂给 LLM 的结构化上下文。

    参数：
        - query (str)：用户原始问题
        - ll_keywords (str)：低级实体关键词（用于局部检索、实体检索）
        - hl_keywords (str)：高级关系关键词（用于全局检索、关系检索）
        - knowledge_graph_inst (BaseGraphStorage)：知识图谱存储实例
        - entities_vdb (BaseVectorStorage)：实体向量库
        - relationships_vdb (BaseVectorStorage)：关系向量库
        - text_chunks_db (BaseKVStorage)：文本分块键值数据库
        - query_param (QueryParam)：查询参数
        - chunks_vdb (BaseVectorStorage)：文本块向量库（mix 模式使用）

    返回：
        tuple[str, list]：
            第一个值：拼接好的完整上下文字符串（实体+关系+文本块+图片）
            第二个值：图片路径列表，用于多模态 LLM 输入
    """
    # 打印日志：当前进程开始构建查询上下文
    logger.info(f"Process {os.getpid()} building query context...")

    # 与精修流程逻辑类似，但额外汇总页面图片用于多模态输入
    # 收集所有来源的文本块（实体来源、关系来源、向量来源）
    all_chunks = []
    # 实体上下文列表
    entities_context = []
    # 关系上下文列表
    relations_context = []
    # 页面图片路径列表（多模态）
    page_imgs = []

    # 保存原始实体与关系数据，用于后续反向查找文本块
    original_node_datas = []
    original_edge_datas = []

    # 根据查询模式（local/global/hybrid/mix）执行不同的图谱检索策略
    # 处理局部模式（local）：专注检索【实体】
    if query_param.mode == "local":
        (
            entities_context,
            relations_context,
            node_datas,
            use_relations,
        ) = await _get_node_data(
            ll_keywords,  # 低级关键词
            knowledge_graph_inst,  # 图谱实例
            entities_vdb,  # 实体向量库
            query_param,  # 查询参数
        )
        # 保存原始实体与关系
        original_node_datas = node_datas
        original_edge_datas = use_relations

    # 处理全局模式（global）：专注检索【关系】
    elif query_param.mode == "global":
        (
            entities_context,
            relations_context,
            edge_datas,
            use_entities,
        ) = await _get_edge_data(
            hl_keywords,  # 高级关键词
            knowledge_graph_inst,  # 图谱实例
            relationships_vdb,  # 关系向量库
            query_param,  # 查询参数
        )
        # 保存原始关系与实体
        original_edge_datas = edge_datas
        original_node_datas = use_entities

    # 混合模式（hybrid / mix）：同时检索实体 + 关系
    else:
        # 并行执行实体检索
        ll_data = await _get_node_data(
            ll_keywords,
            knowledge_graph_inst,
            entities_vdb,
            query_param,
        )
        # 并行执行关系检索
        hl_data = await _get_edge_data(
            hl_keywords,
            knowledge_graph_inst,
            relationships_vdb,
            query_param,
        )

        # 拆解局部（实体）检索结果
        (ll_entities_context, ll_relations_context, ll_node_datas, ll_edge_datas) = (
            ll_data
        )
        # 拆解全局（关系）检索结果
        (hl_entities_context, hl_relations_context, hl_edge_datas, hl_node_datas) = (
            hl_data
        )

        # mix 模式：额外做向量检索，并从文本块中提取图片
        if query_param.mode == "mix" and chunks_vdb:
            # 从向量库检索相关文本块
            vector_chunks = await _get_vector_context(
                query,
                chunks_vdb,
                query_param,
            )
            # 提取 chunk_id
            chunk_ids = [d['id'] for d in vector_chunks]
            # 从数据库获取完整分块
            chunks = await text_chunks_db.get_by_ids(chunk_ids)
            # 如果分块包含图片，提取图片路径
            if 'page_img' in chunks[0]:
                page_imgs = [chunk['page_img'] for chunk in chunks]

            # 将向量检索的块加入总块列表
            all_chunks.extend(vector_chunks)

        # 合并两种检索方式的原始数据
        original_node_datas = ll_node_datas + hl_node_datas
        original_edge_datas = ll_edge_datas + hl_edge_datas

        # 合并实体上下文（去重、合并）
        entities_context = process_combine_contexts(
            ll_entities_context, hl_entities_context
        )
        # 合并关系上下文
        relations_context = process_combine_contexts(
            hl_relations_context, ll_relations_context
        )

    # 打印初始检索到的数量日志
    logger.info(
        f"Initial context: {len(entities_context)} entities, {len(relations_context)} relations, {len(all_chunks)} chunks"
    )

    # ====================== 统一 Token 长度控制 ======================
    # 获取分词器，用于裁剪超长上下文
    tokenizer = text_chunks_db.global_config.get("tokenizer")
    if tokenizer:
        # 从查询参数/全局配置获取最大 Token 限制
        # 实体最大 Token
        max_entity_tokens = getattr(
            query_param,
            "max_entity_tokens",
            text_chunks_db.global_config.get(
                "max_entity_tokens", DEFAULT_MAX_ENTITY_TOKENS
            ),
        )
        # 关系最大 Token
        max_relation_tokens = getattr(
            query_param,
            "max_relation_tokens",
            text_chunks_db.global_config.get(
                "max_relation_tokens", DEFAULT_MAX_RELATION_TOKENS
            ),
        )
        # 总上下文最大 Token
        max_total_tokens = getattr(
            query_param,
            "max_total_tokens",
            text_chunks_db.global_config.get(
                "max_total_tokens", DEFAULT_MAX_TOTAL_TOKENS
            ),
        )

        # 如果有实体上下文，清理格式并按 Token 裁剪
        if entities_context:
            original_entity_count = len(entities_context)

            # 替换文件路径中的分隔符，避免解析异常
            for entity in entities_context:
                if "file_path" in entity and entity["file_path"]:
                    entity["file_path"] = entity["file_path"].replace(
                        GRAPH_FIELD_SEP, ";"
                    )

            # 按 Token 长度裁剪实体列表
            entities_context = truncate_list_by_token_size(
                entities_context,
                key=lambda x: json.dumps(x, ensure_ascii=False),
                max_token_size=max_entity_tokens,
                tokenizer=tokenizer,
            )
            # 打印裁剪日志
            if len(entities_context) < original_entity_count:
                logger.debug(
                    f"Truncated entities: {original_entity_count} -> {len(entities_context)} (entity max tokens: {max_entity_tokens})"
                )

        # 如果有关系上下文，清理格式并按 Token 裁剪
        if relations_context:
            original_relation_count = len(relations_context)

            # 替换路径分隔符
            for relation in relations_context:
                if "file_path" in relation and relation["file_path"]:
                    relation["file_path"] = relation["file_path"].replace(
                        GRAPH_FIELD_SEP, ";"
                    )

            # 按 Token 长度裁剪关系列表
            relations_context = truncate_list_by_token_size(
                relations_context,
                key=lambda x: json.dumps(x, ensure_ascii=False),
                max_token_size=max_relation_tokens,
                tokenizer=tokenizer,
            )
            # 打印裁剪日志
            if len(relations_context) < original_relation_count:
                logger.debug(
                    f"Truncated relations: {original_relation_count} -> {len(relations_context)} (relation max tokens: {max_relation_tokens})"
                )

    # ====================== 根据裁剪后的实体/关系，反向查找关联文本块 ======================
    logger.info("Getting text chunks based on truncated entities and relations...")

    # 筛选出裁剪后最终保留的实体数据
    final_node_datas = []
    if entities_context and original_node_datas:
        # 提取最终实体名称集合
        final_entity_names = {e["entity"] for e in entities_context}
        seen_nodes = set()
        # 从原始数据中过滤出保留的实体
        for node in original_node_datas:
            name = node.get("entity_name")
            if name in final_entity_names and name not in seen_nodes:
                final_node_datas.append(node)
                seen_nodes.add(name)

    # 筛选出裁剪后最终保留的关系数据
    final_edge_datas = []
    if relations_context and original_edge_datas:
        # 提取最终关系对 (实体1, 实体2)
        final_relation_pairs = {(r["entity1"], r["entity2"]) for r in relations_context}
        seen_edges = set()
        # 从原始数据中过滤出保留的关系
        for edge in original_edge_datas:
            src, tgt = edge.get("src_id"), edge.get("tgt_id")
            if src is None or tgt is None:
                src, tgt = edge.get("src_tgt", (None, None))

            pair = (src, tgt)
            if pair in final_relation_pairs and pair not in seen_edges:
                final_edge_datas.append(edge)
                seen_edges.add(pair)

    # 并行任务：从实体/关系反向查找关联文本块
    text_chunk_tasks = []

    # 从实体找文本块
    if final_node_datas:
        text_chunk_tasks.append(
            _find_most_related_text_unit_from_entities(
                final_node_datas,
                query_param,
                text_chunks_db,
                knowledge_graph_inst,
            )
        )

    # 从关系找文本块
    if final_edge_datas:
        text_chunk_tasks.append(
            _find_related_text_unit_from_relationships(
                final_edge_datas,
                query_param,
                text_chunks_db,
            )
        )

    # 并行执行，提高速度
    if text_chunk_tasks:
        text_chunk_results = await asyncio.gather(*text_chunk_tasks)
        for chunks in text_chunk_results:
            if chunks:
                all_chunks.extend(chunks)

    # 对文本块进行动态 Token 预算裁剪，构建最终上下文
    text_units_context = []
    if tokenizer and all_chunks:
        # 计算已用 Token：实体 + 关系
        entities_str = json.dumps(entities_context, ensure_ascii=False)
        relations_str = json.dumps(relations_context, ensure_ascii=False)

        # Calculate base context tokens (entities + relations + template)
        kg_context_template = """-----Entities(KG)-----

```json
{entities_str}
```

-----Relationships(KG)-----

```json
{relations_str}
```

-----Document Chunks(DC)-----

```json
[]
```

"""
        # 使用模板格式化知识图谱上下文，将实体和关系序列化为字符串填入模板
        kg_context = kg_context_template.format(
            entities_str=entities_str, relations_str=relations_str
        )
        # 计算知识图谱上下文占用的token数量
        kg_context_tokens = len(tokenizer.encode(kg_context))

        # 动态计算实际系统提示词的固定开销（不包含文本块上下文）
        # 1. 计算对话历史占用的token数
        history_context = ""
        if query_param.conversation_history:
            # 从对话历史中提取指定轮数的内容
            history_context = get_conversation_turns(
                query_param.conversation_history, query_param.history_turns
            )
        # 计算历史对话的token长度，无历史则为0
        history_tokens = (
            len(tokenizer.encode(history_context)) if history_context else 0
        )

        # 2. 计算系统提示词模板本身的token开销（不包含动态上下文内容）
        # 获取用户自定义提示词，无则使用空字符串
        user_prompt = query_param.user_prompt if query_param.user_prompt else ""
        # 获取回答格式类型，无则默认多段落格式
        response_type = (
            query_param.response_type
            if query_param.response_type
            else "Multiple Paragraphs"
        )

        # 从全局配置获取系统提示词模板，无配置则使用默认的RAG提示词模板
        sys_prompt_template = text_chunks_db.global_config.get(
            "system_prompt_template", PROMPTS["rag_response"]
        )

        # 构造一个空上下文的示例提示词，专门用于计算固定开销的token长度
        sample_sys_prompt = sys_prompt_template.format(
            history=history_context,  # 填入历史对话
            context_data="",  # 上下文内容置空，仅计算模板开销
            response_type=response_type,  # 填入回答格式要求
            user_prompt=user_prompt,  # 填入用户自定义提示词
        )
        # 计算系统提示词模板的固定token开销
        sys_prompt_template_tokens = len(tokenizer.encode(sample_sys_prompt))

        # 系统提示词总固定开销 = 模板本身token + 用户问题token
        query_tokens = len(tokenizer.encode(query))  # 计算用户问题的token长度
        sys_prompt_overhead = sys_prompt_template_tokens + query_tokens

        # 安全缓冲token：预留100个token防止总长度超出模型上限
        buffer_tokens = 100

        # 计算可分配给【文本块】的剩余token数量
        # 已用token = 图谱上下文 + 提示词固定开销 + 安全缓冲
        used_tokens = kg_context_tokens + sys_prompt_overhead + buffer_tokens
        # 可用文本块token = 模型最大允许token - 已用token
        available_chunk_tokens = max_total_tokens - used_tokens

        # 打印token分配明细日志，便于调试
        logger.debug(
            f"Token allocation - Total: {max_total_tokens}, History: {history_tokens}, SysPrompt: {sys_prompt_overhead}, KG: {kg_context_tokens}, Buffer: {buffer_tokens}, Available for chunks: {available_chunk_tokens}"
        )

        # 使用动态计算的token限制，重新处理文本块
        if all_chunks:
            # 构造临时文本块列表，仅保留内容和文件路径字段
            temp_chunks = [
                {"content": chunk["content"], "file_path": chunk["file_path"]}
                for chunk in all_chunks
            ]

            # 调用统一分块处理函数，按动态可用token长度裁剪文本块
            truncated_chunks = await process_chunks_unified(
                query=query,
                chunks=temp_chunks,
                query_param=query_param,
                global_config=text_chunks_db.global_config,
                source_type="mixed",  # 数据源类型：混合来源（实体+关系+向量）
                chunk_token_limit=available_chunk_tokens,  # 传入动态计算的可用token上限
            )

            # 使用裁剪后的文本块，重新构建最终的文本单元上下文
            for i, chunk in enumerate(truncated_chunks):
                text_units_context.append(
                    {
                        "id": i + 1,  # 文本块序号
                        "content": chunk["content"],  # 文本块内容
                        "file_path": chunk.get("file_path", "unknown_source"),  # 来源文件路径
                    }
                )

            # 打印文本块裁剪日志
            logger.debug(
                f"Re-truncated chunks for dynamic token limit: {len(temp_chunks)} -> {len(text_units_context)} (chunk available tokens: {available_chunk_tokens})"
            )

    # 打印最终构建完成的上下文统计信息
    logger.info(
        f"Final context: {len(entities_context)} entities, {len(relations_context)} relations, {len(text_units_context)} chunks"
    )
    # 关键判断：如果既没有实体也没有关系，说明无有效参考信息，无法生成回答
    if not entities_context and not relations_context:
        return None
    # 将最终的实体、关系、文本块上下文序列化为JSON字符串
    entities_str = json.dumps(entities_context, ensure_ascii=False)
    relations_str = json.dumps(relations_context, ensure_ascii=False)
    text_units_str = json.dumps(text_units_context, ensure_ascii=False)
    # 构建图片信息JSON：将图片路径转为多模态模型可识别的格式
    page_imgs_str = json.dumps({f"image_{i}": f"filename:{p.split('/')[-1]}" for i, p in enumerate(page_imgs)},
                                   ensure_ascii=False)
    result = f"""-----Entities(KG)-----

```json
{entities_str}
```

-----Relationships(KG)-----

```json
{relations_str}
```

-----Document Chunks(DC)-----

```json
{text_units_str}
```

-----Page Images (PI)-----
```json
{page_imgs_str}
```
"""
    return result, page_imgs

async def _handle_single_entity_extraction(
    record_attributes: list[str],
    chunk_key: str,
    file_path: str = "unknown_source",
):
    """
    功能说明：
        处理单条实体抽取结果，进行格式校验、文本清洗、字段标准化，
        将 LLM 输出的原始文本 → 转换为系统可使用的标准实体结构体。
        属于实体抽取流程中的“结果解析与清洗”核心步骤。

    参数：
        - record_attributes (list[str])：从 LLM 返回结果中拆分出的实体属性列表
        - chunk_key (str)：当前实体来源的文本分块唯一 ID
        - file_path (str)：实体来自的文档文件路径，用于溯源、日志、引用

    返回：
        dict | None：
            - 成功 → 返回标准格式的实体字典
            - 失败 → 返回 None（格式错误/空值/非法数据）
    """
    # ===================== 格式合法性校验 =====================
    # 检查属性长度是否足够，且第一条标记是 "entity"，否则不是有效实体
    if len(record_attributes) < 4 or '"entity"' not in record_attributes[0]:
        return None

    # ===================== 清洗 & 校验：实体名称 =====================
    # 清洗实体名称字符串，去除多余空格、符号
    entity_name = clean_str(record_attributes[1]).strip()
    # 实体名称不能为空，否则记录警告并丢弃该实体
    if not entity_name:
        logger.warning(
            f"Entity extraction error: empty entity name in: {record_attributes}"
        )
        return None

    # 对实体名称进行标准化处理（统一大小写、去冗余、统一格式）
    entity_name = normalize_extracted_info(entity_name, is_entity=True)

    # 再次校验：标准化后实体名称不能为空
    if not entity_name or not entity_name.strip():
        logger.warning(
            f"Entity extraction error: entity name became empty after normalization. Original: '{record_attributes[1]}'"
        )
        return None

    # ===================== 清洗 & 校验：实体类型 =====================
    # 清洗实体类型字段，并去除多余引号
    entity_type = clean_str(record_attributes[2]).strip('"')
    # 实体类型不能为空或格式异常
    if not entity_type.strip() or entity_type.startswith('("'):
        logger.warning(
            f"Entity extraction error: invalid entity type in: {record_attributes}"
        )
        return None

    # ===================== 清洗 & 校验：实体描述 =====================
    # 清洗实体描述文本
    entity_description = clean_str(record_attributes[3])
    # 对描述文本进行标准化
    entity_description = normalize_extracted_info(entity_description)

    # 实体描述不能为空
    if not entity_description.strip():
        logger.warning(
            f"Entity extraction error: empty description for entity '{entity_name}' of type '{entity_type}'"
        )
        return None

    # ===================== 返回标准结构化实体 =====================
    # 返回清洗、校验、标准化后的标准实体数据
    return dict(
        entity_name=entity_name,        # 标准化后的实体名称
        entity_type=entity_type,        # 实体类型（人物、地点、事件等）
        description=entity_description, # 实体描述
        source_id=chunk_key,            # 来源分块ID
        file_path=file_path,            # 来源文件路径
    )

async def _handle_single_relationship_extraction(
    record_attributes: list[str],
    chunk_key: str,
    file_path: str = "unknown_source",
):
    """
    功能说明：
        处理单条关系抽取结果并转换为标准结构。
    
    参数：
        - record_attributes (list[str])：方法执行所需输入参数。
        - chunk_key (str)：方法执行所需输入参数。
        - file_path (str)：文档文件路径或来源标识。
    
    返回：
        Any：方法执行结果；若为 None 表示主要通过副作用完成处理。
    """
    # 关系记录需至少包含头实体、尾实体、描述、关键词（及可选权重）。
    if len(record_attributes) < 5 or '"relationship"' not in record_attributes[0]:
        return None
    # add this record as edge
    source = clean_str(record_attributes[1])
    target = clean_str(record_attributes[2])

    # Normalize source and target entity names
    source = normalize_extracted_info(source, is_entity=True)
    target = normalize_extracted_info(target, is_entity=True)

    # Check if source or target became empty after normalization
    if not source or not source.strip():
        logger.warning(
            f"Relationship extraction error: source entity became empty after normalization. Original: '{record_attributes[1]}'"
        )
        return None

    if not target or not target.strip():
        logger.warning(
            f"Relationship extraction error: target entity became empty after normalization. Original: '{record_attributes[2]}'"
        )
        return None

    if source == target:
        logger.debug(
            f"Relationship source and target are the same in: {record_attributes}"
        )
        return None

    edge_description = clean_str(record_attributes[3])
    edge_description = normalize_extracted_info(edge_description)

    edge_keywords = normalize_extracted_info(
        clean_str(record_attributes[4]), is_entity=True
    )
    edge_keywords = edge_keywords.replace("，", ",")

    edge_source_id = chunk_key
    weight = (
        float(record_attributes[-1].strip('"').strip("'"))
        if is_float_regex(record_attributes[-1].strip('"').strip("'"))
        else 1.0
    )
    return dict(
        src_id=source,
        tgt_id=target,
        weight=weight,
        description=edge_description,
        keywords=edge_keywords,
        source_id=edge_source_id,
        file_path=file_path,
    )

async def merge_nodes_and_edges(
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
) -> None:
    """
    功能说明：
        合并所有文本分块的实体（节点）与关系（边）抽取结果，
        执行去重、融合、持久化到知识图谱，并同步更新向量数据库。
        是构建知识图谱的**最终合并入库**核心函数。

    参数：
        - chunk_results (list)：所有分块的实体、关系抽取结果集合
        - knowledge_graph_inst (BaseGraphStorage)：知识图谱存储引擎实例
        - entity_vdb (BaseVectorStorage)：实体向量库实例（用于检索）
        - relationships_vdb (BaseVectorStorage)：关系向量库实例
        - global_config (dict[str, str])：全局配置
        - pipeline_status (dict)：全局状态日志
        - pipeline_status_lock (Any)：状态日志异步锁
        - llm_response_cache (BaseKVStorage | None)：LLM 缓存
        - current_file_number (int)：当前处理第几个文件
        - total_files (int)：总文件数
        - file_path (str)：当前文件路径

    返回：
        None：无返回值，直接写入图谱与向量库
    """
    # ===================== 第一步：汇总所有实体与关系 =====================
    # 把所有 chunk 的抽取结果汇总到一起，统一处理，避免多次写入冲突
    all_nodes = defaultdict(list)  # 汇总所有实体：key=实体名，value=实体列表
    all_edges = defaultdict(list)   # 汇总所有关系：key=关系键，value=关系列表

    # 遍历每个分块的结果
    for maybe_nodes, maybe_edges in chunk_results:
        # 收集实体
        for entity_name, entities in maybe_nodes.items():
            all_nodes[entity_name].extend(entities)

        # 收集关系：对无向图，将关系两端排序，确保 A-B 和 B-A 视为同一条关系
        for edge_key, edges in maybe_edges.items():
            sorted_edge_key = tuple(sorted(edge_key))
            all_edges[sorted_edge_key].extend(edges)

    # 统计本次合并的实体与关系总数
    total_entities_count = len(all_nodes)
    total_relations_count = len(all_edges)

    # ===================== 第二步：输出日志，进入合并阶段 =====================
    log_message = f"Merging stage {current_file_number}/{total_files}: {file_path}"
    logger.info(log_message)
    async with pipeline_status_lock:
        pipeline_status["latest_message"] = log_message
        pipeline_status["history_messages"].append(log_message)

    # ===================== 第三步：设置并发控制，防止模型/数据库过载 =====================
    # 并发数 = LLM 最大并发数 × 2，保证处理速度同时不压垮服务
    graph_max_async = global_config.get("llm_model_max_async", 4) * 2
    semaphore = asyncio.Semaphore(graph_max_async)

    # 输出处理规模日志
    log_message = f"Processing: {total_entities_count} entities and {total_relations_count} relations (async: {graph_max_async})"
    logger.info(log_message)
    async with pipeline_status_lock:
        pipeline_status["latest_message"] = log_message
        pipeline_status["history_messages"].append(log_message)

    # ===================== 第四步：定义【实体合并】任务（带锁防冲突） =====================
    async def _locked_process_entity_name(entity_name, entities):
        """
        功能说明：
            带锁安全处理单个实体：合并同名实体 → 写入图谱 → 更新向量库
            使用 keyed lock 保证同名实体同一时间只被一个任务处理，避免重复创建
        """
        async with semaphore:
            workspace = global_config.get("workspace", "")
            namespace = f"{workspace}:GraphDB" if workspace else "GraphDB"
            # 按实体名加锁，防止并发竞态
            async with get_storage_keyed_lock(
                [entity_name], namespace=namespace, enable_logging=False
            ):
                # 合并同名实体，并插入/更新到知识图谱
                entity_data = await _merge_nodes_then_upsert(
                    entity_name,
                    entities,
                    knowledge_graph_inst,
                    global_config,
                    pipeline_status,
                    pipeline_status_lock,
                    llm_response_cache,
                )
                # 如果启用了实体向量库，同步插入向量索引
                if entity_vdb is not None:
                    data_for_vdb = {
                        compute_mdhash_id(entity_data["entity_name"], prefix="ent-"): {
                            "entity_name": entity_data["entity_name"],
                            "entity_type": entity_data["entity_type"],
                            "content": f"{entity_data['entity_name']}\n{entity_data['description']}",
                            "source_id": entity_data["source_id"],
                            "file_path": entity_data.get("file_path", "unknown_source"),
                        }
                    }
                    await entity_vdb.upsert(data_for_vdb)
                return entity_data

    # ===================== 第五步：定义【关系合并】任务（带锁防冲突） =====================
    async def _locked_process_edges(edge_key, edges):
        """
        功能说明：
            带锁安全处理单条关系：合并重复关系 → 写入图谱 → 更新向量库
        """
        async with semaphore:
            workspace = global_config.get("workspace", "")
            namespace = f"{workspace}:GraphDB" if workspace else "GraphDB"
            # 对关系两端排序，保证无向图锁一致
            sorted_edge_key = sorted([edge_key[0], edge_key[1]])
            # 按关系两端加锁
            async with get_storage_keyed_lock(
                sorted_edge_key,
                namespace=namespace,
                enable_logging=False,
            ):
                # 合并关系并插入/更新到知识图谱
                edge_data = await _merge_edges_then_upsert(
                    edge_key[0],
                    edge_key[1],
                    edges,
                    knowledge_graph_inst,
                    global_config,
                    pipeline_status,
                    pipeline_status_lock,
                    llm_response_cache,
                )
                if edge_data is None:
                    return None

                # 如果启用了关系向量库，同步插入向量索引
                if relationships_vdb is not None:
                    data_for_vdb = {
                        compute_mdhash_id(
                            edge_data["src_id"] + edge_data["tgt_id"], prefix="rel-"
                        ): {
                            "src_id": edge_data["src_id"],
                            "tgt_id": edge_data["tgt_id"],
                            "keywords": edge_data["keywords"],
                            "content": f"{edge_data['src_id']}\t{edge_data['tgt_id']}\n{edge_data['keywords']}\n{edge_data['description']}",
                            "source_id": edge_data["source_id"],
                            "file_path": edge_data.get("file_path", "unknown_source"),
                            "weight": edge_data.get("weight", 1.0),
                        }
                    }
                    await relationships_vdb.upsert(data_for_vdb)
                return edge_data

    # ===================== 第六步：创建所有异步任务 =====================
    tasks = []

    # 添加实体处理任务
    for entity_name, entities in all_nodes.items():
        tasks.append(
            asyncio.create_task(_locked_process_entity_name(entity_name, entities))
        )

    # 添加关系处理任务
    for edge_key, edges in all_edges.items():
        tasks.append(asyncio.create_task(_locked_process_edges(edge_key, edges)))

    # ===================== 第七步：执行任务 + 快速失败机制 =====================
    # 只要任意任务报错，立即停止所有任务，提高稳定性
    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)

    # 检查是否有任务失败
    for task in done:
        if task.exception():
            # 取消所有未完成任务
            for pending_task in pending:
                pending_task.cancel()

            # 等待取消完成
            if pending:
                await asyncio.wait(pending)

            # 抛出异常，通知上层
            raise task.exception()

    # 所有任务执行成功，流程结束

async def extract_entities(
        chunks: dict[str, TextChunkSchema],
        global_config: dict[str, str],
        pipeline_status: dict = None,
        pipeline_status_lock=None,
        llm_response_cache: BaseKVStorage | None = None,
        text_chunks_storage: BaseKVStorage | None = None,
) -> list:
    """
    功能说明：
        从文本分块中批量抽取实体与关系，是知识图谱构建的核心入口方法
        支持普通文本/多模态（带图片）两种抽取模式，使用LLM完成结构化信息提取

    参数：
        - chunks (dict[str, TextChunkSchema])：文本分块字典，key=chunk_id，value=分块内容结构体
        - global_config (dict[str, str])：全局配置字典，包含LLM模型、提示词、语言、实体类型等核心参数
        - pipeline_status (dict)：流水线全局状态字典，用于记录处理进度、日志、状态消息
        - pipeline_status_lock (Any)：异步互斥锁，保护 pipeline_status 多协程安全修改
        - llm_response_cache (BaseKVStorage | None)：LLM响应缓存实例，用于重复请求加速、避免重复调用
        - text_chunks_storage (BaseKVStorage | None)：文本分块持久化存储实例，用于分块数据读写

    返回：
        list：返回所有分块的实体关系抽取结果列表，每个元素对应一个chunk的抽取结果
    """
    # 初始化抽取所需核心组件：LLM调用函数、最大重试/补全次数
    # 从全局配置中获取LLM模型调用方法（不同大模型统一封装接口）
    use_llm_func: callable = global_config["llm_model_func"]
    # 实体抽取最大补全次数（LLM未抽取完整时，允许循环补全抽取）
    entity_extract_max_gleaning = global_config["entity_extract_max_gleaning"]

    # 判断分块是否包含图片数据，用于区分普通文本/多模态抽取
    # 获取第一个分块的所有key，判断是否包含图片字段
    chunk_keys = list(list(chunks.values())[0].keys())
    # 如果分块包含 page_img 字段，判定为多模态数据（文本+图片）
    _content_images = True if "page_img" in chunk_keys else False

    # 加载实体抽取提示词模板（根据是否带图片选择普通/多模态提示词）
    # 默认：纯文本实体抽取提示词
    PROMPTS_ENTITY_EXTRACTION = PROMPTS["entity_extraction"]
    # 默认：纯文本实体抽取示例
    PROMPTS_ENTITY_EXTRACTION_EXAMPLES = PROMPTS["entity_extraction_examples"]
    # 如果是多模态数据（带图片），替换为多模态专用提示词
    if _content_images:
        PROMPTS_ENTITY_EXTRACTION = PROMPTS["multimodal_entity_extraction_init"]
        PROMPTS_ENTITY_EXTRACTION_EXAMPLES = PROMPTS["multimodal_entity_extraction_examples"]

    # 将分块字典转为有序列表，保证处理顺序稳定
    ordered_chunks = list(chunks.items())
    # 从配置中读取多语言、实体类型、示例数量等提示词参数
    # 处理语言：默认中文
    language = global_config["addon_params"].get(
        "language", PROMPTS["DEFAULT_LANGUAGE"]
    )
    # 待抽取的实体类型：默认包含人名、地名、组织机构、时间、日期、数值等
    entity_types = global_config["addon_params"].get(
        "entity_types", PROMPTS["DEFAULT_ENTITY_TYPES"]
    )
    # 提示词中使用的示例数量
    example_number = global_config["addon_params"].get("example_number", None)
    # 根据配置截取示例数量：如果配置了有效数字，使用前N个示例
    if example_number and example_number < len(PROMPTS_ENTITY_EXTRACTION_EXAMPLES):
        examples = "\n".join(
            PROMPTS_ENTITY_EXTRACTION_EXAMPLES[: int(example_number)]
        )
    # 否则使用全部示例
    else:
        examples = "\n".join(PROMPTS_ENTITY_EXTRACTION_EXAMPLES)

    # 构建基础提示词上下文变量：定义分隔符、实体类型、语言
    example_context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],  # 元组分隔符
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],  # 记录分隔符
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],  # 抽取结束标记
        entity_types=", ".join(entity_types),  # 实体类型字符串
        language=language,  # 输出语言
    )
    # 将基础变量填充到示例模板中，生成完整的示例文本
    examples = examples.format(**example_context_base)

    # 主抽取提示词（不带参数填充）
    entity_extract_prompt = PROMPTS_ENTITY_EXTRACTION
    # 构建完整提示词上下文：包含分隔符、实体类型、示例、语言
    context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        entity_types=",".join(entity_types),
        examples=examples,
        language=language,
    )

    # 构建“继续抽取”提示词：LLM返回结果不完整时，用于触发继续抽取
    continue_prompt = PROMPTS["entity_continue_extraction"].format(**context_base)
    # 构建“是否需要继续抽取”判断提示词：用于判断实体是否抽取完整
    if_loop_prompt = PROMPTS["entity_if_loop_extraction"]

    # 初始化进度计数器：已处理分块数量
    processed_chunks = 0
    # 总分块数量
    total_chunks = len(ordered_chunks)

    async def _process_extraction_result(
            result: str, chunk_key: str, file_path: str = "unknown_source"
    ):
        """
        功能说明：
            清洗、解析、标准化 LLM 单次实体关系抽取结果
            将原始字符串输出 → 结构化实体（节点）+ 关系（边）
            供后续知识图谱合并使用

        参数：
            - result (str)：LLM 返回的原始抽取结果字符串
            - chunk_key (str)：当前文本分块的唯一 ID
            - file_path (str)：文档文件路径，用于日志、溯源

        返回：
            tuple[defaultdict, defaultdict]：(实体集合, 关系集合)
            实体：key=实体名，value=实体详情列表
            关系：key=(源实体ID, 目标实体ID)，value=关系详情列表
        """
        # 初始化容器：存储待确认的实体（节点）、关系（边）
        maybe_nodes = defaultdict(list)
        maybe_edges = defaultdict(list)

        # 按分隔符拆分 LLM 输出的多条记录（实体/关系/结束标记）
        records = split_string_by_multi_markers(
            result,
            [context_base["record_delimiter"], context_base["completion_delimiter"]],
        )

        # 遍历每一条抽取记录
        for record in records:
            # 正则提取括号内的内容，去除格式干扰
            record = re.search(r"\((.*)\)", record)
            if record is None:
                continue
            record = record.group(1)

            # 按元组分隔符拆分属性字段
            record_attributes = split_string_by_multi_markers(
                record, [context_base["tuple_delimiter"]]
            )

            # 尝试解析为【实体】
            if_entities = await _handle_single_entity_extraction(
                record_attributes, chunk_key, file_path
            )
            if if_entities is not None:
                # 实体存入节点容器：key=实体名称
                maybe_nodes[if_entities["entity_name"]].append(if_entities)
                continue

            # 尝试解析为【关系】
            if_relation = await _handle_single_relationship_extraction(
                record_attributes, chunk_key, file_path
            )
            if if_relation is not None:
                # 关系存入边容器：key=(源实体id, 目标实体id)
                maybe_edges[(if_relation["src_id"], if_relation["tgt_id"])].append(
                    if_relation
                )

        # 返回结构化实体与关系
        return maybe_nodes, maybe_edges

    async def _process_single_content(chunk_key_dp: tuple[str, TextChunkSchema]):
        """
        功能说明：
            处理**单个文本分块**的完整实体关系抽取流程
            流程：首次抽取 → 多轮补全抽取 → 结果合并 → 状态更新

        参数：
            - chunk_key_dp (tuple[str, TextChunkSchema])：(分块ID, 分块内容结构体)

        返回：
            tuple[defaultdict, defaultdict]：当前分块抽取的(实体集合, 关系集合)
        """
        # 声明使用外层函数的分块计数器
        nonlocal processed_chunks

        # 解构分块 ID 与分块内容
        chunk_key = chunk_key_dp[0]
        chunk_dp = chunk_key_dp[1]
        content = chunk_dp["content"]

        # 处理多模态：如果分块带图片，收集图像列表
        images = []
        if "page_img" in chunk_dp:
            images = [chunk_dp["page_img"]] + chunk_dp["fig_imgs"]

        # 获取文件路径（无则使用默认值）
        file_path = chunk_dp.get("file_path", "unknown_source")

        # 缓存键收集器：用于批量记录该分块使用了哪些 LLM 缓存
        cache_keys_collector = []

        # ===================== 首次实体关系抽取 =====================
        # 填充首次抽取提示词
        hint_prompt = entity_extract_prompt.format(
            **{**context_base, "input_text": content}
        )
        # 带缓存调用 LLM 进行抽取
        final_result = await use_llm_func_with_cache(
            hint_prompt,
            use_llm_func,
            input_images=images,
            llm_response_cache=llm_response_cache,
            cache_type="extract",
            chunk_id=chunk_key,
            cache_keys_collector=cache_keys_collector,
        )
        # 构建对话历史：用户提示 + LLM 回复，用于后续补全抽取
        history = pack_user_ass_to_openai_messages(hint_prompt, final_result)

        # 解析首次抽取结果 → 实体、关系
        maybe_nodes, maybe_edges = await _process_extraction_result(
            final_result, chunk_key, file_path
        )

        # ===================== 多轮补全抽取（Gleaning）=====================
        # 仅补充【新实体/新关系】，避免结果膨胀
        for now_glean_index in range(entity_extract_max_gleaning):
            # 执行一轮补全抽取
            glean_result = await use_llm_func_with_cache(
                continue_prompt,
                use_llm_func,
                input_images=images,
                llm_response_cache=llm_response_cache,
                history_messages=history,
                cache_type="extract",
                chunk_id=chunk_key,
                cache_keys_collector=cache_keys_collector,
            )

            # 追加对话历史
            history += pack_user_ass_to_openai_messages(continue_prompt, glean_result)

            # 解析本轮补全结果
            glean_nodes, glean_edges = await _process_extraction_result(
                glean_result, chunk_key, file_path
            )

            # 合并：只新增【不存在的实体名/关系对】，不重复添加
            for entity_name, entities in glean_nodes.items():
                if entity_name not in maybe_nodes:
                    maybe_nodes[entity_name].extend(entities)
            for edge_key, edges in glean_edges.items():
                if edge_key not in maybe_edges:
                    maybe_edges[edge_key].extend(edges)

            # 到达最大补全次数，退出
            if now_glean_index == entity_extract_max_gleaning - 1:
                break

            # 判断是否需要继续补全
            if_loop_result: str = await use_llm_func_with_cache(
                if_loop_prompt,
                use_llm_func,
                llm_response_cache=llm_response_cache,
                history_messages=history,
                cache_type="extract",
                cache_keys_collector=cache_keys_collector,
            )
            # 格式化判断结果
            if_loop_result = if_loop_result.strip().strip('"').strip("'").lower()
            # LLM 返回不是 yes → 停止补全
            if if_loop_result != "yes":
                break

        # ===================== 批量写回 LLM 缓存记录 =====================
        # 将本次分块使用的所有缓存键写入分块数据，便于溯源、调试
        if cache_keys_collector and text_chunks_storage:
            await update_chunk_cache_list(
                chunk_key,
                text_chunks_storage,
                cache_keys_collector,
                "entity_extraction",
            )

        # 更新处理进度计数器
        processed_chunks += 1
        entities_count = len(maybe_nodes)
        relations_count = len(maybe_edges)
        log_message = f"Chunk {processed_chunks} of {total_chunks} extracted {entities_count} Ent + {relations_count} Rel"
        logger.info(log_message)

        # 更新全局流水线状态（日志展示）
        if pipeline_status is not None:
            async with pipeline_status_lock:
                pipeline_status["latest_message"] = log_message
                pipeline_status["history_messages"].append(log_message)

        # 返回当前分块的抽取结果，供全局合并使用
        return maybe_nodes, maybe_edges

    # ===================== 分块级并发控制 =====================
    # 从全局配置获取最大并发数，避免 LLM 并发过载
    chunk_max_async = global_config.get("llm_model_max_async", 4)
    # 异步信号量：限制同时处理的分块数量
    semaphore = asyncio.Semaphore(chunk_max_async)

    async def _process_with_semaphore(chunk):
        """
        功能说明：
            带信号量限流的包装函数，保证 LLM 调用并发安全

        参数：
            - chunk (tuple)：单个分块数据

        返回：
            Any：_process_single_content 的返回结果
        """
        async with semaphore:
            return await _process_single_content(chunk)

    # 创建所有分块的处理任务
    tasks = []
    for c in ordered_chunks:
        task = asyncio.create_task(_process_with_semaphore(c))
        tasks.append(task)

    # ===================== 任务执行：异常快速失败策略 =====================
    # 只要任意任务抛出异常，立即停止所有任务，提高稳定性
    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)

    # 检查是否有任务失败
    for task in done:
        if task.exception():
            # 取消所有未完成的任务
            for pending_task in pending:
                pending_task.cancel()

            # 等待取消完成
            if pending:
                await asyncio.wait(pending)

            # 抛出异常，通知上层处理
            raise task.exception()

    # 所有任务成功 → 收集结果
    chunk_results = [task.result() for task in tasks]

    # 返回所有分块的抽取结果，供后续合并节点与关系
    return chunk_results


async def extract_entities_refinement(
        chunks: dict[str, TextChunkSchema],
        chunk_results: list,
        knowledge_graph_inst: BaseGraphStorage,
        entity_vdb: BaseVectorStorage,
        relationships_vdb: BaseVectorStorage,
        global_config: dict[str, str],
        pipeline_status: dict = None,
        pipeline_status_lock=None,
        llm_response_cache: BaseKVStorage | None = None,
        text_chunks_storage: BaseKVStorage | None = None,
) -> list:
    """
    功能说明：
        实体关系抽取**精修阶段**。
        基于第一阶段的抽取结果，结合**知识图谱已有上下文**，
        进行更精准、更完整的二次抽取，提升实体与关系的准确率。
        核心：带图谱上下文的再抽取 + 结果融合。

    参数：
        - chunks (dict[str, TextChunkSchema])：文本分块字典 {chunk_id: 分块内容}
        - chunk_results (list)：第一阶段抽取的实体/关系原始结果
        - knowledge_graph_inst (BaseGraphStorage)：知识图谱存储实例
        - entity_vdb (BaseVectorStorage)：实体向量库
        - relationships_vdb (BaseVectorStorage)：关系向量库
        - global_config (dict[str, str])：全局配置
        - pipeline_status (dict)：流水线状态（日志、进度）
        - pipeline_status_lock (Any)：状态更新异步锁
        - llm_response_cache (BaseKVStorage | None)：LLM 响应缓存
        - text_chunks_storage (BaseKVStorage | None)：文本分块存储

    返回：
        list：精修后的所有分块实体/关系结果，用于后续合并入库
    """
    # ===================== 第一步：构建第一阶段结果索引 =====================
    # 把第一阶段抽取结果按 chunk_id 建立字典，方便精修时快速查找
    chunk_results_at_stage_one = {
        list(res[0].values())[0][0]['source_id']: {
            'nodes': res[0],
            'edges': res[1],
        } for res in chunk_results
    }

    # ===================== 第二步：加载配置与提示词 =====================
    # 获取 LLM 调用函数 & 最大补全抽取次数
    use_llm_func: callable = global_config["llm_model_func"]
    entity_extract_max_gleaning = global_config["addon_params"]["entity_extract_max_gleaning"]

    # 判断分块是否包含图片（多模态）
    chunk_keys = list(list(chunks.values())[0].keys())
    _content_images = True if "page_img" in chunk_keys else False

    # 根据是否有图片，选择普通抽取 / 多模态抽取提示词
    PROMPTS_ENTITY_EXTRACTION = PROMPTS["entity_extraction"]
    PROMPTS_ENTITY_EXTRACTION_EXAMPLES = PROMPTS["entity_extraction_examples"]
    if _content_images:
        PROMPTS_ENTITY_EXTRACTION = PROMPTS["multimodal_entity_extraction_refine"]
        PROMPTS_ENTITY_EXTRACTION_EXAMPLES = PROMPTS["multimodal_entity_extraction_examples"]

    # 把分块字典转为有序列表，保证按顺序处理
    ordered_chunks = list(chunks.items())

    # 加载语言、实体类型、示例配置
    language = global_config["addon_params"].get(
        "language", PROMPTS["DEFAULT_LANGUAGE"]
    )
    entity_types = global_config["addon_params"].get(
        "entity_types", PROMPTS["DEFAULT_ENTITY_TYPES"]
    )
    example_number = global_config["addon_params"].get("example_number", None)
    if example_number and example_number < len(PROMPTS_ENTITY_EXTRACTION_EXAMPLES):
        examples = "\n".join(
            PROMPTS_ENTITY_EXTRACTION_EXAMPLES[: int(example_number)]
        )
    else:
        examples = "\n".join(PROMPTS_ENTITY_EXTRACTION_EXAMPLES)

    # 构造提示词基础变量（分隔符、实体类型、语言）
    example_context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        entity_types=", ".join(entity_types),
        language=language,
    )
    # 填充示例格式
    examples = examples.format(**example_context_base)

    entity_extract_prompt = PROMPTS_ENTITY_EXTRACTION
    context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        entity_types=",".join(entity_types),
        examples=examples,
        language=language,
    )

    # 加载“继续抽取”和“是否继续判断”提示词
    continue_prompt = PROMPTS["entity_continue_extraction"].format(**context_base)
    if_loop_prompt = PROMPTS["entity_if_loop_extraction"]

    # 分块处理计数器
    processed_chunks = 0
    total_chunks = len(ordered_chunks)

    # ===================== 核心函数1：解析单条抽取结果 =====================
    async def _process_extraction_result(
            result: str, chunk_key: str, file_path: str = "unknown_source"
    ):
        """
        功能说明：
            解析 LLM 返回的抽取字符串，清洗、拆分、格式化为实体/关系结构
            与第一阶段解析逻辑完全一致，保证结构兼容
        """
        # 存储待确认的实体与关系
        maybe_nodes = defaultdict(list)
        maybe_edges = defaultdict(list)

        # 按记录分隔符拆分多条结果
        records = split_string_by_multi_markers(
            result,
            [context_base["record_delimiter"], context_base["completion_delimiter"]],
        )

        # 遍历每条记录
        for record in records:
            # 提取括号内内容
            record = re.search(r"\((.*)\)", record)
            if record is None:
                continue
            record = record.group(1)
            # 按元组分隔符拆分属性
            record_attributes = split_string_by_multi_markers(
                record, [context_base["tuple_delimiter"]]
            )

            # 解析为实体
            if_entities = await _handle_single_entity_extraction(
                record_attributes, chunk_key, file_path
            )
            if if_entities is not None:
                maybe_nodes[if_entities["entity_name"]].append(if_entities)
                continue

            # 解析为关系
            if_relation = await _handle_single_relationship_extraction(
                record_attributes, chunk_key, file_path
            )
            if if_relation is not None:
                maybe_edges[(if_relation["src_id"], if_relation["tgt_id"])].append(
                    if_relation
                )

        return maybe_nodes, maybe_edges

    # ===================== 核心函数2：处理单个分块（精修逻辑） =====================
    async def _process_single_content(chunk_key_dp: tuple[str, TextChunkSchema], chunk_results_s1: dict):
        """
        功能说明：
            处理**单个分块的精修抽取**
            流程：查询子图 → 注入图谱上下文 → 精修抽取 → 合并第一阶段结果
        """
        nonlocal processed_chunks

        # 解构分块 ID 和内容
        chunk_key = chunk_key_dp[0]
        chunk_dp = chunk_key_dp[1]
        content = chunk_dp["content"]

        # ===================== 精修核心：查询相关子图 =====================
        # 根据第一阶段的实体/关系，从图谱中查询相关子图，作为精修上下文
        page_relevant_subgraph = await _search_subgraph(
            nodes=chunk_results_s1['nodes'],
            edges=chunk_results_s1['edges']
        )

        # 处理多模态图片
        images = []
        if "page_img" in chunk_dp:
            images = [chunk_dp["page_img"]] + chunk_dp["fig_imgs"]
        file_path = chunk_dp.get("file_path", "unknown_source")

        # 缓存键收集器
        cache_keys_collector = []

        # ===================== 第一步：精修抽取（注入图谱上下文） =====================
        hint_prompt = entity_extract_prompt.format(
            **{
                **context_base,
                "input_text": content,
                "kg_context": page_relevant_subgraph if page_relevant_subgraph else "empty"
            }
        )
        # 带缓存调用 LLM
        final_result = await use_llm_func_with_cache(
            hint_prompt,
            use_llm_func,
            input_images=images,
            llm_response_cache=llm_response_cache,
            cache_type="extract",
            chunk_id=chunk_key,
            cache_keys_collector=cache_keys_collector,
        )
        # 构建对话历史
        history = pack_user_ass_to_openai_messages(hint_prompt, final_result)

        # 解析结果
        maybe_nodes, maybe_edges = await _process_extraction_result(
            final_result, chunk_key, file_path
        )

        # ===================== 合并第一阶段结果，保证不丢失 =====================
        # 只新增第一阶段有、但精修结果没有的实体
        for entity_name, entities in chunk_results_s1["nodes"].items():
            if entity_name not in maybe_nodes:
                maybe_nodes[entity_name].extend(entities)
        # 只新增第一阶段有、但精修结果没有的关系
        for edge_key, edges in chunk_results_s1["edges"].items():
            if edge_key not in maybe_edges:
                maybe_edges[edge_key].extend(edges)

        # ===================== 多轮补全抽取 =====================
        for now_glean_index in range(entity_extract_max_gleaning):
            # 继续补抽取
            glean_result = await use_llm_func_with_cache(
                continue_prompt,
                use_llm_func,
                input_images=images,
                llm_response_cache=llm_response_cache,
                history_messages=history,
                cache_type="extract",
                chunk_id=chunk_key,
                cache_keys_collector=cache_keys_collector,
            )
            history += pack_user_ass_to_openai_messages(continue_prompt, glean_result)

            # 解析补抽取结果
            glean_nodes, glean_edges = await _process_extraction_result(
                glean_result, chunk_key, file_path
            )

            # 合并：只新增不存在的实体/关系
            for entity_name, entities in glean_nodes.items():
                if entity_name not in maybe_nodes:
                    maybe_nodes[entity_name].extend(entities)
            for edge_key, edges in glean_edges.items():
                if edge_key not in maybe_edges:
                    maybe_edges[edge_key].extend(edges)

            # 到达最大次数，退出
            if now_glean_index == entity_extract_max_gleaning - 1:
                break

            # 判断是否需要继续补全
            if_loop_result: str = await use_llm_func_with_cache(
                if_loop_prompt,
                use_llm_func,
                llm_response_cache=llm_response_cache,
                history_messages=history,
                cache_type="extract",
                cache_keys_collector=cache_keys_collector,
            )
            if_loop_result = if_loop_result.strip().strip('"').strip("'").lower()
            if if_loop_result != "yes":
                break

        # 批量更新缓存记录
        if cache_keys_collector and text_chunks_storage:
            await update_chunk_cache_list(
                chunk_key,
                text_chunks_storage,
                cache_keys_collector,
                "entity_extraction",
            )

        # 更新日志
        processed_chunks += 1
        entities_count = len(maybe_nodes)
        relations_count = len(maybe_edges)
        log_message = f"Chunk {processed_chunks} of {total_chunks} extracted {entities_count} Ent + {relations_count} Rel"
        logger.info(log_message)
        if pipeline_status is not None:
            async with pipeline_status_lock:
                pipeline_status["latest_message"] = log_message
                pipeline_status["history_messages"].append(log_message)

        return maybe_nodes, maybe_edges

    # ===================== 核心函数3：查询相关子图（精修专用） =====================
    async def _search_subgraph(nodes, edges):
        """
        功能说明：
            根据当前分块的实体与关系，从知识图谱中查询相关子图
            用于注入到提示词中，提升精修抽取准确性
        """
        # 构造查询关键词：实体名 + 关系关键词
        ll_keywords = [k for k, v in nodes.items()]
        hl_keywords = [dp['keywords'] for v in edges.values() for dp in v]

        if not ll_keywords and not hl_keywords:
            return None
        ll_keywords_str = ", ".join(ll_keywords) if ll_keywords else ""
        hl_keywords_str = ", ".join(hl_keywords) if hl_keywords else ""

        # 查询模式：混合 / 全局 / 局部
        query_mode = "hybrid"
        if not ll_keywords_str:
            query_mode = "global"
        elif not hl_keywords_str:
            query_mode = "local"

        # 构造查询参数
        # mode: 查询模式
        # - local：只查当前 chunk 实体的直接邻居（小而准）
        # - global：全局向量检索，不限制范围（广而全）
        # - hybrid：混合模式（先局部 + 后全局）
        # enable_rerank：用 LLM 再重新排序一次，把最相关的放最前面
        query_param = QueryParam(mode=query_mode, enable_rerank=False)
        query_param.top_k = global_config['addon_params']['refine_subgraph_top_k']
        query_param.max_token_for_global_context = global_config['addon_params'][
            'refine_subgraph_max_token_for_global_context']
        query_param.max_token_for_local_context = global_config['addon_params'][
            'refine_subgraph_max_token_for_local_context']
        query_param.max_token_for_text_unit = global_config['addon_params']['refine_subgraph_max_token_for_text_unit']

        # 构建精修用查询上下文
        subgraph = await _build_query_context_for_refine(
            query=", ".join([ll_keywords_str, hl_keywords_str]),
            ll_keywords=ll_keywords_str,
            hl_keywords=hl_keywords_str,
            knowledge_graph_inst=knowledge_graph_inst,
            entities_vdb=entity_vdb,
            relationships_vdb=relationships_vdb,
            text_chunks_db=text_chunks_storage,
            query_param=query_param,
        )
        # 精修只需要图谱结构，移除文档块文本
        subgraph = subgraph.split('-----Document Chunks(DC)-----')[0]
        return subgraph

    # ===================== 并发控制 =====================
    chunk_max_async = global_config.get("llm_model_max_async", 4)
    semaphore = asyncio.Semaphore(chunk_max_async)

    async def _process_with_semaphore(chunk, chunk_results_s1):
        """带信号量限流的包装函数"""
        async with semaphore:
            return await _process_single_content(chunk, chunk_results_s1)

    # 创建所有分块的处理任务
    tasks = []
    for c in ordered_chunks:
        task = asyncio.create_task(_process_with_semaphore(c, chunk_results_at_stage_one[c[0]]))
        tasks.append(task)

    # ===================== 执行任务：快速失败机制 =====================
    # 任一任务失败，立即取消所有任务
    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)

    # 处理异常
    for task in done:
        if task.exception():
            for pending_task in pending:
                pending_task.cancel()
            if pending:
                await asyncio.wait(pending)
            raise task.exception()

    # 收集所有分块的精修结果
    chunk_results = [task.result() for task in tasks]

    # 返回精修后的结果，用于合并入库
    return chunk_results

async def naive_query(
    query: str,
    chunks_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage,
    query_param: QueryParam,
    global_config: dict[str, str],
    hashing_kv: BaseKVStorage | None = None,
    system_prompt: str | None = None,
) -> str | AsyncIterator[str]:
    """
    功能说明：
        执行基础检索问答流程（朴素RAG，不依赖复杂图推理），仅通过向量检索文本块生成回答。
        核心流程：缓存检查 → 向量检索 → 上下文截断 → 提示词组装 → LLM生成 → 结果缓存。

    参数：
        - query (str)：用户输入的查询问题
        - chunks_vdb (BaseVectorStorage)：文本分块向量数据库，用于相似度检索
        - text_chunks_db (BaseKVStorage)：文本分块键值存储，用于根据ID获取完整分块内容
        - query_param (QueryParam)：查询参数对象，包含模型、流式、重排、上下文条数等配置
        - global_config (dict[str, str])：全局运行配置字典，包含分词器、最大token数、默认模型等
        - hashing_kv (BaseKVStorage | None)：LLM问答缓存存储实例，用于缓存重复查询结果
        - system_prompt (str | None)：自定义系统提示词，不传入则使用默认提示词

    返回：
        str | AsyncIterator[str]：返回最终回答字符串；若开启流式返回，则返回异步迭代器
    """
    # 选择模型函数：优先使用查询参数中指定的模型，否则使用全局配置的模型，并设置查询优先级为5
    if query_param.model_func:
        use_model_func = query_param.model_func
    else:
        use_model_func = global_config["llm_model_func"]
        # 为查询相关的LLM函数应用更高优先级(5)
        use_model_func = partial(use_model_func, _priority=5)

    # 先查询缓存，若命中直接返回，降低重复查询的性能开销
    # 处理缓存逻辑
    args_hash = compute_args_hash(query_param.mode, query)
    cached_response, quantized, min_val, max_val = await handle_cache(
        hashing_kv, args_hash, query, query_param.mode, cache_type="query"
    )
    if cached_response is not None:
        return cached_response

    # 获取全局分词器，用于计算token数量
    tokenizer: Tokenizer = global_config["tokenizer"]

    # 从向量数据库中检索与查询问题相关的文本分块
    chunks = await _get_vector_context(query, chunks_vdb, query_param)

    # 未检索到任何分块，直接返回预设的失败响应
    if chunks is None or len(chunks) == 0:
        return PROMPTS["fail_response"]

    # 动态计算分块可用token数量，避免提示词总长度超出模型限制
    # 计算分块动态token限制
    # 从查询参数获取最大总token数，无则使用全局配置默认值
    max_total_tokens = getattr(
        query_param,
        "max_total_tokens",
        global_config.get("max_total_tokens", DEFAULT_MAX_TOTAL_TOKENS),
    )

    # 计算对话历史占用的token数
    history_context = ""
    if query_param.conversation_history:
        history_context = get_conversation_turns(
            query_param.conversation_history, query_param.history_turns
        )
    history_tokens = len(tokenizer.encode(history_context)) if history_context else 0

    # 计算系统提示词模板（不含上下文内容）的token数
    user_prompt = query_param.user_prompt if query_param.user_prompt else ""
    response_type = (
        query_param.response_type
        if query_param.response_type
        else "Multiple Paragraphs"
    )

    # 使用传入的系统提示词或默认提示词模板
    sys_prompt_template = (
        system_prompt if system_prompt else PROMPTS["naive_rag_response"]
    )

    # 创建空内容的示例系统提示词，用于计算固定开销token
    sample_sys_prompt = sys_prompt_template.format(
        content_data="",  # 空内容用于计算开销
        response_type=response_type,
        history=history_context,
        user_prompt=user_prompt,
    )
    sys_prompt_template_tokens = len(tokenizer.encode(sample_sys_prompt))

    # 系统提示词总开销 = 模板token + ·
    query_tokens = len(tokenizer.encode(query))
    sys_prompt_overhead = sys_prompt_template_tokens + query_tokens

    # 安全缓冲token，防止溢出
    buffer_tokens = 100

    # 计算可分配给文本分块的token数量
    used_tokens = sys_prompt_overhead + buffer_tokens
    available_chunk_tokens = max_total_tokens - used_tokens

    # 打印token分配日志，便于调试
    logger.debug(
        f"Naive query token allocation - Total: {max_total_tokens}, History: {history_tokens}, SysPrompt: {sys_prompt_overhead}, Buffer: {buffer_tokens}, Available for chunks: {available_chunk_tokens}"
    )

    # 统一处理分块：重排序、截断，得到符合token限制的最终上下文片段
    # 使用统一处理函数处理分块，传入动态计算的token限制
    processed_chunks = await process_chunks_unified(
        query=query,
        chunks=chunks,
        query_param=query_param,
        global_config=global_config,
        source_type="vector",
        chunk_token_limit=available_chunk_tokens,  # 传入动态限制值
    )

    logger.info(f"Final context: {len(processed_chunks)} chunks")

    # 构建上下文内容，组织分块信息，并补充分页图片引用（多模态场景）
    # 从处理后的分块构建文本单元上下文
    text_units_context, chunk_ids = [], []
    for i, chunk in enumerate(processed_chunks):
        text_units_context.append(
            {
                "id": i + 1,
                "content": chunk["content"],
                "file_path": chunk.get("file_path", "unknown_source"),
            }
        )
        chunk_ids.append(chunk['id'])
    # 根据分块ID从数据库获取完整信息
    chunks = await text_chunks_db.get_by_ids(chunk_ids)
    page_imgs_str = ""
    # 提取分块中的图片信息，用于多模态问答
    if 'page_img' in chunks[0]:
        page_imgs = [chunk['page_img'] for chunk in chunks]
        page_imgs_str = json.dumps({f"image_{i}": f"filename:{p.split('/')[-1]}" for i, p in enumerate(page_imgs)}, ensure_ascii=False)

    # 将上下文信息序列化为JSON字符串
    text_units_str = json.dumps(text_units_context, ensure_ascii=False)
    context = f"""
---Document Chunks---

```json
{text_units_str}
```

-----Page Images (PI)-----
```json
{page_imgs_str}
```
"""
    # 如果用户只需要获取上下文内容，直接返回拼接好的上下文
    if query_param.only_need_context:
        return context
    # 重新处理对话历史（兼容上层逻辑，确保历史正确加载）
    history_context = ""
    if query_param.conversation_history:
        history_context = get_conversation_turns(
            query_param.conversation_history, query_param.history_turns
        )

    # 组装最终系统提示词（包含上下文、历史对话、用户自定义要求）
    # 构建系统提示词
    user_prompt = (
        query_param.user_prompt
        if query_param.user_prompt
        else PROMPTS["DEFAULT_USER_PROMPT"]
    )
    sys_prompt_temp = system_prompt if system_prompt else PROMPTS["naive_rag_response"]
    sys_prompt = sys_prompt_temp.format(
        content_data=context,
        response_type=query_param.response_type,
        history=history_context,
        user_prompt=user_prompt,
    )

    # 如果用户只需要生成提示词，不进行LLM调用，直接返回提示词
    if query_param.only_need_prompt:
        return sys_prompt

    # 计算本次发送给LLM的总token数，用于日志调试
    len_of_prompts = len(tokenizer.encode(query + sys_prompt))
    logger.debug(
        f"[naive_query] Sending to LLM: {len_of_prompts:,} tokens (Query: {len(tokenizer.encode(query))}, System: {len(tokenizer.encode(sys_prompt))})"
    )

    # 调用LLM模型生成回答，支持流式输出与多模态图片输入
    response = await use_model_func(
        query,
        input_images=page_imgs,
        system_prompt=sys_prompt,
        stream=query_param.stream,
    )

    # 对返回的字符串结果做清洗：去除提示词、角色标记、用户问题等冗余内容
    if isinstance(response, str) and len(response) > len(sys_prompt):
        response = (
            response[len(sys_prompt):]
            .replace(sys_prompt, "")
            .replace("user", "")
            .replace("model", "")
            .replace(query, "")
            .replace("<system>", "")
            .replace("</system>", "")
            .strip()
        )

    # 如果全局开启了LLM缓存，将本次问答结果存入缓存
    if hashing_kv.global_config.get("enable_llm_cache"):
        # 保存结果到缓存
        await save_to_cache(
            hashing_kv,
            CacheData(
                args_hash=args_hash,
                content=response,
                prompt=query,
                quantized=quantized,
                min_val=min_val,
                max_val=max_val,
                mode=query_param.mode,
                cache_type="query",
            ),
        )

    # 返回最终生成的回答（字符串或异步迭代器）
    return response


async def kg_query(
        query: str,
        knowledge_graph_inst: BaseGraphStorage,
        entities_vdb: BaseVectorStorage,
        relationships_vdb: BaseVectorStorage,
        text_chunks_db: BaseKVStorage,
        query_param: QueryParam,
        global_config: dict[str, str],
        hashing_kv: BaseKVStorage | None = None,
        system_prompt: str | None = None,
        chunks_vdb: BaseVectorStorage = None,
) -> str | AsyncIterator[str]:
    """
    功能说明：
        执行知识图谱增强查询流程，综合实体关系进行回答。

    参数：
        - query (str)：用户输入的查询问题。
        - knowledge_graph_inst (BaseGraphStorage)：知识图谱存储实例。
        - entities_vdb (BaseVectorStorage)：实体向量数据库实例。
        - relationships_vdb (BaseVectorStorage)：关系向量存储实例。
        - text_chunks_db (BaseKVStorage)：文本分块键值存储实例。
        - query_param (QueryParam)：查询参数对象。
        - global_config (dict[str, str])：全局运行配置字典。
        - hashing_kv (BaseKVStorage | None)：LLM缓存存储实例。
        - system_prompt (str | None)：自定义系统提示词。
        - chunks_vdb (BaseVectorStorage)：文本分块向量库（混合模式使用）。

    返回：
        str | AsyncIterator[str]：返回回答字符串或流式迭代器。
    """
    # 选择LLM模型函数：优先查询参数，其次全局配置，提升优先级为5
    if query_param.model_func:
        use_model_func = query_param.model_func
    else:
        use_model_func = global_config["llm_model_func"]
        # 为查询相关的LLM函数应用更高优先级(5)
        use_model_func = partial(use_model_func, _priority=5)

    # 检查缓存，命中则直接返回缓存结果
    # 处理缓存
    args_hash = compute_args_hash(query_param.mode, query)
    cached_response, quantized, min_val, max_val = await handle_cache(
        hashing_kv, args_hash, query, query_param.mode, cache_type="query"
    )
    if cached_response is not None:
        return cached_response

    # 从用户问题中抽取高低优先级关键词，用于知识图谱检索
    hl_keywords, ll_keywords = await get_keywords_from_query(
        query, query_param, global_config, hashing_kv
    )

    # 打印关键词日志，便于调试
    logger.debug(f"High-level keywords: {hl_keywords}")
    logger.debug(f"Low-level  keywords: {ll_keywords}")

    # 关键词缺失时自动降级查询模式，保证流程可用
    # 处理关键词为空的异常情况
    if hl_keywords == [] and ll_keywords == []:
        logger.warning("low_level_keywords and high_level_keywords is empty")
        return PROMPTS["fail_response"]
    if ll_keywords == [] and query_param.mode in ["local", "hybrid"]:
        logger.warning(
            "low_level_keywords is empty, switching from %s mode to global mode",
            query_param.mode,
        )
        query_param.mode = "global"
    if hl_keywords == [] and query_param.mode in ["global", "hybrid"]:
        logger.warning(
            "high_level_keywords is empty, switching from %s mode to local mode",
            query_param.mode,
        )
        query_param.mode = "local"

    # 将关键词列表转为逗号分隔字符串
    ll_keywords_str = ", ".join(ll_keywords) if ll_keywords else ""
    hl_keywords_str = ", ".join(hl_keywords) if hl_keywords else ""

    # 构建包含实体、关系、文本块、图片的综合上下文
    # 构建查询上下文（含图片）
    context, page_imgs = await _build_query_context_with_image(
        query,
        ll_keywords_str,
        hl_keywords_str,
        knowledge_graph_inst,
        entities_vdb,
        relationships_vdb,
        text_chunks_db,
        query_param,
        chunks_vdb,
    )

    # 如果只需要上下文，直接返回上下文
    if query_param.only_need_context:
        return context if context is not None else PROMPTS["fail_response"]
    # 上下文为空，返回失败响应
    if context is None:
        return PROMPTS["fail_response"]

    # 处理对话历史
    history_context = ""
    if query_param.conversation_history:
        history_context = get_conversation_turns(
            query_param.conversation_history, query_param.history_turns
        )

    # 构建系统提示词
    user_prompt = (
        query_param.user_prompt
        if query_param.user_prompt
        else PROMPTS["DEFAULT_USER_PROMPT"]
    )
    sys_prompt_temp = system_prompt if system_prompt else PROMPTS["rag_response"]
    sys_prompt = sys_prompt_temp.format(
        context_data=context,
        response_type=query_param.response_type,
        history=history_context,
        user_prompt=user_prompt,
    )

    # 如果只需要提示词，直接返回提示词
    if query_param.only_need_prompt:
        return sys_prompt

    # 计算提示词token数并打印调试日志
    tokenizer: Tokenizer = global_config["tokenizer"]
    len_of_prompts = len(tokenizer.encode(query + sys_prompt))
    logger.debug(
        f"[kg_query] Sending to LLM: {len_of_prompts:,} tokens (Query: {len(tokenizer.encode(query))}, System: {len(tokenizer.encode(sys_prompt))})"
    )
    # 调用LLM生成回答
    response = await use_model_func(
        query,
        system_prompt=sys_prompt,
        stream=query_param.stream,
        input_images=page_imgs,
    )
    # 清洗返回结果，去除冗余内容
    if isinstance(response, str) and len(response) > len(sys_prompt):
        response = (
            response.replace(sys_prompt, "")
            .replace("user", "")
            .replace("model", "")
            .replace(query, "")
            .replace("<system>", "")
            .replace("</system>", "")
            .strip()
        )

    # 开启缓存则写入缓存
    if hashing_kv.global_config.get("enable_llm_cache"):
        # 保存到缓存
        await save_to_cache(
            hashing_kv,
            CacheData(
                args_hash=args_hash,
                content=response,
                prompt=query,
                quantized=quantized,
                min_val=min_val,
                max_val=max_val,
                mode=query_param.mode,
                cache_type="query",
            ),
        )

    # 返回最终回答
    return response


async def kg_two_step_query(
        query: str,
        knowledge_graph_inst: BaseGraphStorage,
        entities_vdb: BaseVectorStorage,
        relationships_vdb: BaseVectorStorage,
        text_chunks_db: BaseKVStorage,
        query_param: QueryParam,
        global_config: dict[str, str],
        hashing_kv: BaseKVStorage | None = None,
        system_prompt: str | None = None,
        chunks_vdb: BaseVectorStorage = None,
) -> str | AsyncIterator[str]:
    """
    功能说明：
        执行两阶段图谱查询，先并行检索图谱+朴素向量结果，再融合生成最终答案。

    参数：
        - query (str)：用户输入的查询问题。
        - knowledge_graph_inst (BaseGraphStorage)：知识图谱存储实例。
        - entities_vdb (BaseVectorStorage)：实体向量库。
        - relationships_vdb (BaseVectorStorage)：关系向量库。
        - text_chunks_db (BaseKVStorage)：文本分块存储。
        - query_param (QueryParam)：查询参数。
        - global_config (dict[str, str])：全局配置。
        - hashing_kv (BaseKVStorage | None)：缓存实例。
        - system_prompt (str | None)：自定义提示词。
        - chunks_vdb (BaseVectorStorage)：分块向量库。

    返回：
        str | AsyncIterator[str]：最终回答字符串或流式迭代器。
    """
    # 选择LLM模型函数
    if query_param.model_func:
        use_model_func = query_param.model_func
    else:
        use_model_func = global_config["llm_model_func"]
        # 为查询相关的LLM函数应用更高优先级(5)
        use_model_func = partial(use_model_func, _priority=5)
    # 检查缓存
    args_hash = compute_args_hash(query_param.mode, query)
    cached_response, quantized, min_val, max_val = await handle_cache(
        hashing_kv, args_hash, query, query_param.mode, cache_type="query"
    )
    if cached_response is not None:
        return cached_response

    # 构造混合模式参数，用于并行执行图谱查询
    hybrid_param = QueryParam(
        mode='hybrid',
        chunk_top_k=query_param.chunk_top_k,
        enable_rerank=False
    )
    # 创建知识图谱查询任务
    kg_task = kg_query(
        query,
        knowledge_graph_inst,
        entities_vdb,
        relationships_vdb,
        text_chunks_db,
        hybrid_param,
        global_config,
        hashing_kv=hashing_kv,
        system_prompt=system_prompt,
        chunks_vdb=chunks_vdb,
    )
    # 构造朴素RAG参数
    naive_param = QueryParam(
        mode='naive',
        chunk_top_k=query_param.chunk_top_k,
        enable_rerank=False
    )
    # 创建朴素向量查询任务
    naive_task = naive_query(
        query,
        chunks_vdb,
        text_chunks_db,
        naive_param,
        global_config,
        hashing_kv=hashing_kv,
        system_prompt=system_prompt,
    )
    # 并行执行两个任务，提升效率
    kg_response, naive_response = await asyncio.gather(kg_task, naive_task)

    # 组装两阶段提示词，融合两个结果
    sys_prompt = PROMPTS["rag_two_step_response"].format(
        query=query,
        kg_answer=kg_response,
        image_answer=naive_response,
    )
    # 调用LLM做最终融合生成
    response = await use_model_func(
        query,
        system_prompt=sys_prompt,
        stream=query_param.stream
    )
    # 清洗结果
    if isinstance(response, str) and len(response) > len(sys_prompt):
        response = (
            response.replace(sys_prompt, "")
            .replace("user", "")
            .replace("model", "")
            .replace(query, "")
            .replace("<system>", "")
            .replace("</system>", "")
            .strip()
        )

    # 写入缓存
    if hashing_kv.global_config.get("enable_llm_cache"):
        # 保存到缓存
        await save_to_cache(
            hashing_kv,
            CacheData(
                args_hash=args_hash,
                content=response,
                prompt=query,
                quantized=quantized,
                min_val=min_val,
                max_val=max_val,
                mode=query_param.mode,
                cache_type="query",
            ),
        )

    # 返回最终融合后的回答
    return response
