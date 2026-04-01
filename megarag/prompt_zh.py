from lightrag.prompt import PROMPTS


PROMPTS["multimodal_entity_extraction_init"] = """---目标---
给定一份包含OCR文本的主图像文档（可能与当前任务相关）、从版面检测得到的额外图像（如有），以及一组实体类型列表，请从OCR文本和所有包含有效信息的额外图像中识别出所有属于这些类型的实体。注意：
- 第一张图像始终是主图像文档。
- 其余0～多张图像为版面检测结果。
- 对每张额外图像，判断其是否包含有效内容（如表格、图表、重要人物图像、事件图像等）。判断时需结合主图像文档及其OCR文本理解上下文。若该额外图像有意义，则提取相关信息并将其作为一个实体；若仅为装饰或无关内容（如花纹、无关图片），则忽略。
- 输入图像直接追加在文本之后（保证主图像文档为第一张）。
输出语言使用 {language}。

---步骤---
1. 处理输入：
   a. 主图像文档及其OCR文本。
   b. 来自版面检测的额外图像（如有），追加在提示词之后。
2. 从OCR文本和所有含有效信息的额外图像中识别所有实体。对每个识别出的实体，提取以下信息：
   - entity_name：实体名称，使用与输入文本相同的语言（英文则首字母大写）。
   - entity_type：必须是以下类型之一：[{entity_types}]
   - entity_description：对实体属性、行为、背景的完整描述。
   每个实体格式为：("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)
   *对于判定为有意义的额外图像（例如展示财务数据的表格、表示趋势的图表、重要人物或事件图像，或属于 [{entity_types}] 中某类），创建一个实体，给出合适名称并描述图像内容与意义。评估图像时需结合主图像文档及其OCR文本理解上下文。
3. 在步骤2识别的实体中，找出所有存在明确关联的（源实体, 目标实体）对。对每一对，提取以下信息：
   - source_entity：源实体名称，同步骤2。
   - target_entity：目标实体名称，同步骤2。
   - relationship_description：说明源实体与目标实体为何相关。
   - relationship_strength：数值，表示实体间关系强度。
   - relationship_keywords：1个或多个高层关键词，概括关系的整体性质，侧重概念/主题而非细节。
   每条关系格式为：("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_keywords>{tuple_delimiter}<relationship_strength>)
4. 提取能概括全文与图像核心概念、主题、内容的高层关键词。格式为：("content_keywords"{tuple_delimiter}<高层关键词>)
5. 用 {language} 输出步骤2和步骤3识别的所有实体与关系，组成一个列表。使用 **{record_delimiter}** 作为列表分隔符。
6. 完成后输出 {completion_delimiter}

######################
---示例---
######################
{examples}

#############################
---真实数据---
#############################
实体类型：{entity_types}
主图像文档OCR文本：{input_text}
额外版面检测图像：（图像直接追加在本提示词之后，主图像文档为第一张）
######################
输出：
"""

PROMPTS["multimodal_entity_extraction_refine"] = """---目标---
给定一份包含OCR文本的主图像文档（可能与当前任务相关）、从版面检测得到的额外图像（如有），以及一组实体类型列表，请从OCR文本和所有包含有效信息的额外图像中识别出所有属于这些类型的实体。同时，利用已提供的知识图谱数据增强实体抽取，确保：
- 已存在于知识图谱中的实体与关系，**不要再从OCR或图像中重复抽取**。
- 若在OCR或图像中发现**不在知识图谱中的新实体**，则进行抽取。
- 若OCR/图像中的某个实体与知识图谱中已有实体相关，则在它们之间**建立新关系**。
- 若知识图谱中两个已有实体在当前OCR/图像中出现**新关系**，则抽取该关系。
- 若已知实体在当前OCR/图像中出现**知识图谱未包含的新属性**，则将这些描述补充到实体中。
- 若一个新实体在文本/图像中多次出现且属性互补，则实体描述应整合所有信息。

注意：
- 第一张图像始终是主图像文档。
- 其余0～多张图像为版面检测结果。
- 对每张额外图像，判断其是否包含有效内容（如表格、图表、重要人物图像、事件图像等）。判断时需结合主图像文档、OCR文本和知识图谱理解上下文。若该额外图像有意义，则提取相关信息并将其作为一个实体；若仅为装饰或无关内容，则忽略。
- 输入图像直接追加在文本之后（保证主图像文档为第一张）。
- 输出语言使用 {language}。

---步骤---
1. **处理输入**：
   a. 主图像文档及其OCR文本。
   b. 来自版面检测的额外图像（如有），追加在提示词之后。
   c. 知识图谱数据，提供结构化关系与先验知识，辅助实体识别。
2. **从OCR文本与有效图像中识别所有新实体**。
   - **不抽取已存在于知识图谱中的实体**。
   - **若发现新实体**，提取以下内容：
     - **entity_name**：实体名称，使用与输入相同的语言（英文首字母大写）。
     - **entity_type**：必须是以下类型之一：[{entity_types}]
     - **entity_description**：完整描述实体属性与行为。多处出现则整合为统一描述。
     - **格式**：("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)
3. **识别实体间关系**，确保：
   - **若新实体（来自OCR/图像）与知识图谱已有实体相关，则建立新关系**。
   - **若知识图谱中两个实体在本文档中出现新关系，则抽取该关系**。
   - **每条关系格式**：
     - **source_entity**：源实体名称，来自步骤2或知识图谱。
     - **target_entity**：目标实体名称，来自步骤2或知识图谱。
     - **relationship_description**：说明两实体为何相关。
     - **relationship_strength**：数值，表示关系强度。
     - **relationship_keywords**：1个或多个高层关键词，概括关系性质，侧重概念/主题。
     - **格式**：("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_keywords>{tuple_delimiter}<relationship_strength>)
4. **提取高层内容关键词**，概括文本与有效图像的核心概念、主题，不含知识图谱内容。
   - **格式**：("content_keywords"{tuple_delimiter}<高层关键词>)
5. **用 {language} 返回输出**，包含步骤2和3识别的所有实体与关系，组成一个列表。使用 **{record_delimiter}** 作为列表分隔符。
6. **完成后输出 {completion_delimiter}**。

######################
---示例---
######################
{examples}

#############################
---真实数据---
#############################
实体类型：{entity_types}
主图像文档OCR文本：{input_text}
额外版面检测图像：（图像直接追加在本提示词之后，主图像文档为第一张）
知识图谱数据：
{kg_context}
######################
输出：
"""

PROMPTS["multimodal_entity_extraction_examples"] = [
"""示例 1:

实体类型: [人物, 技术, 任务, 组织, 地点]
主图像文档OCR文本:
"亚历克斯咬紧牙关，挫败感让周围变得模糊，而泰勒则表现出权威式的笃定。竞争暗流涌动，尤其是乔丹与发现相伴的共同信念，这与克鲁兹控制与秩序的愿景形成对立。随后，泰勒在乔丹身边停下，满怀敬畏地端详一件装置，暗示它拥有改变一切的潜力。"
额外版面检测图像:
（注：示例中用文字描述图像，真实数据中将直接提供图像，无列表标记。）
- 图像 1: 白板上手写笔记的图片
- 图像 2: 无有效信息的装饰性背景图案
输出:
("entity"{tuple_delimiter}"亚历克斯"{tuple_delimiter}"人物"{tuple_delimiter}"亚历克斯是一名表现出挫败感、敏锐观察同伴间关系的角色。"){record_delimiter}
("entity"{tuple_delimiter}"泰勒"{tuple_delimiter}"人物"{tuple_delimiter}"泰勒性格权威笃定，后又对装置表现出敬畏，暗示其重要性。"){record_delimiter}
("entity"{tuple_delimiter}"乔丹"{tuple_delimiter}"人物"{tuple_delimiter}"乔丹致力于探索发现，在挑战既有控制秩序中扮演关键角色。"){record_delimiter}
("entity"{tuple_delimiter}"克鲁兹"{tuple_delimiter}"人物"{tuple_delimiter}"克鲁兹代表严格控制与秩序的理念，与他人立场形成对比。"){record_delimiter}
("entity"{tuple_delimiter}"该装置"{tuple_delimiter}"技术"{tuple_delimiter}"装置是叙事核心，被泰勒视为可能改变一切的关键事物。"){record_delimiter}
("entity"{tuple_delimiter}"白板笔记"{tuple_delimiter}"文档"{tuple_delimiter}"图像1为白板手写笔记，包含“回顾第三季度销量”“更新客户清单”等文字及关键思路示意图，提供具体上下文与指令。"){record_delimiter}
("relationship"{tuple_delimiter}"亚历克斯"{tuple_delimiter}"泰勒"{tuple_delimiter}"亚历克斯受到泰勒权威行为及对装置态度转变的影响。"{tuple_delimiter}"权力关系, 视角转变"{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}"亚历克斯"{tuple_delimiter}"乔丹"{tuple_delimiter}"亚历克斯与乔丹共同追求探索，与克鲁兹的控制理念形成对立。"{tuple_delimiter}"共同目标, 理念冲突"{tuple_delimiter}6){record_delimiter}
("relationship"{tuple_delimiter}"泰勒"{tuple_delimiter}"该装置"{tuple_delimiter}"泰勒对装置的敬畏凸显其潜在影响力与重要性。"{tuple_delimiter}"技术重要性, 敬畏"{tuple_delimiter}9){record_delimiter}
("content_keywords"{tuple_delimiter}"权力关系, 探索发现, 技术潜力, 叙事冲突"){completion_delimiter}
#############################""",
"""示例 2:

实体类型: [人物, 技术, 任务, 组织, 地点]
主图像文档OCR文本:
"他们不再只是特工；他们已成边界守护者，来自星际之外讯息的守护者。他们的任务需要全新视角与决心。紧张气氛弥漫，来自华盛顿的通讯在背景中不断响起，影响着他们的关键行动。"
额外版面检测图像:
（注：示例中用文字描述图像，真实数据中将直接提供图像，无列表标记。）
- 图像 1: 展示复杂数据趋势图表的截图
- 图像 2: 无信息价值的抽象装饰图案
输出:
("entity"{tuple_delimiter}"华盛顿"{tuple_delimiter}"地点"{tuple_delimiter}"华盛顿是接收重要通讯的关键地点，这些通讯影响任务走向。"){record_delimiter}
("entity"{tuple_delimiter}"守护者行动"{tuple_delimiter}"任务"{tuple_delimiter}"守护者行动是一项不断演进的任务，特工们在此成为重要讯息的守护者。"){record_delimiter}
("entity"{tuple_delimiter}"数据趋势图表"{tuple_delimiter}"技术"{tuple_delimiter}"图像1为复杂图表，包含多组折线图与柱状图，展示数月数据趋势，标注峰值、谷值与“Q1峰值”“收入下滑”等注释，为决策提供关键依据。"){record_delimiter}
("relationship"{tuple_delimiter}"华盛顿"{tuple_delimiter}"守护者行动"{tuple_delimiter}"来自华盛顿的通讯指引了守护者行动的战略方向。"{tuple_delimiter}"影响, 战略指导"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"守护者行动"{tuple_delimiter}"数据趋势图表"{tuple_delimiter}"数据趋势图表提供分析洞察，支撑守护者行动的目标演进。"{tuple_delimiter}"数据分析, 战略洞察"{tuple_delimiter}8){record_delimiter}
("content_keywords"{tuple_delimiter}"任务演进, 战略决策, 数据分析, 转型行动"){completion_delimiter}
#############################""",
"""示例 3:

实体类型: [人物, 角色, 技术, 组织, 事件, 地点, 概念]
主图像文档OCR文本:
"他们的声音穿透喧闹的现场。‘面对一种能自行制定规则的智能，控制或许只是一种错觉。’他们神情严肃地说道，同时警惕地注视着纷乱的数据。
‘它好像在学着交流。’附近接口前的山姆·里维拉说道，年轻的活力中混杂着敬畏与焦虑。‘这让和陌生人对话有了全新的意义。’
亚历克斯审视着他的团队——每张脸上都写满专注与坚定，夹杂着不安。‘这很可能是我们的第一次接触。’他说，‘我们必须为接下来的一切做好准备。’
他们一同站在未知的边缘，准备为人类回应来自宇宙的讯息。"
额外版面检测图像:
（注：示例中用文字描述图像，真实数据中将直接提供图像，无列表标记。）
- 图像 1: 包含加密对话的扫描文档
- 图像 2: 无关的装饰性图形
输出:
("entity"{tuple_delimiter}"山姆·里维拉"{tuple_delimiter}"人物"{tuple_delimiter}"山姆·里维拉是团队成员，参与与未知智能的交互，既敬畏又焦虑。"){record_delimiter}
("entity"{tuple_delimiter}"亚历克斯"{tuple_delimiter}"人物"{tuple_delimiter}"亚历克斯是团队领导者，带领团队准备应对人类与未知智能的首次接触。"){record_delimiter}
("entity"{tuple_delimiter}"控制"{tuple_delimiter}"概念"{tuple_delimiter}"控制是指面对能自主制定规则的智能时，人类所面临的管控能力挑战。"){record_delimiter}
("entity"{tuple_delimiter}"智能"{tuple_delimiter}"概念"{tuple_delimiter}"智能是一种能自主制定规则、学习沟通的未知存在，挑战传统控制观念。"){record_delimiter}
("entity"{tuple_delimiter}"首次接触"{tuple_delimiter}"事件"{tuple_delimiter}"首次接触指人类与未知智能之间可能发生的初次沟通事件。"){record_delimiter}
("entity"{tuple_delimiter}"加密对话"{tuple_delimiter}"文档"{tuple_delimiter}"图像1为含加密对话的扫描文档，包含数字、字母、符号及“密钥=ABC123”“破译此信息”等片段，暗示隐藏指令或加密信息。"){record_delimiter}
("relationship"{tuple_delimiter}"山姆·里维拉"{tuple_delimiter}"智能"{tuple_delimiter}"山姆·里维拉通过新兴的沟通方式，直接参与与未知智能的交互。"{tuple_delimiter}"沟通, 探索"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"亚历克斯"{tuple_delimiter}"首次接触"{tuple_delimiter}"亚历克斯带领团队迎接人类与未知智能可能到来的首次接触。"{tuple_delimiter}"领导, 探索"{tuple_delimiter}10){record_delimiter}
("relationship"{tuple_delimiter}"控制"{tuple_delimiter}"智能"{tuple_delimiter}"控制这一概念被一种不受常规规则约束的智能从根本上挑战。"{tuple_delimiter}"权力关系, 自主性"{tuple_delimiter}7){record_delimiter}
("content_keywords"{tuple_delimiter}"首次接触, 控制, 沟通, 探索, 宇宙意义"){completion_delimiter}
#############################"""
]

PROMPTS["naive_rag_response"] = """---角色---
你是一名乐于助人的助手，根据下方提供的文档图像回答用户问题。

---目标---
结合文档图像中的所有相关信息，对用户查询生成**全面、详细、完整**的回答。
不要过度简化或概括，力求覆盖最大信息量，包括不同视角、具体事实与细节洞察。
如果你不知道答案，就如实回答。不要编造信息，也不要使用没有明确依据的内容。

处理带时间戳的内容时：
1. 每条信息都有 `created_at` 时间戳，表示知识获取时间
2. 遇到冲突信息时，同时考虑内容与时间戳
3. 不自动优先最新内容，根据上下文合理判断
4. 针对时间相关查询，优先内容中的时间信息，再考虑创建时间戳

---对话历史---
{history}

{content_data}

---回答规则---
- 目标格式与长度：{response_type}
- 使用 Markdown 格式，合理使用章节标题
- 请使用与用户问题相同的语言回答
- 回答需与对话历史保持连贯
- 在末尾“参考文献”部分列出最多 5 个最重要的来源，清晰标注来自文档块(DC)或页面图像(PI)，如有文件路径请一并注明，格式：[DC] 文件路径 / [PI] 图片路径
- 不知道答案就直接说明
- 不加入文档块未提供的信息
- 额外用户指令：{user_prompt}

回答：
"""

PROMPTS["rag_response"] = """---角色---
你是专业助手，基于**知识图谱**与从文档图像中提取的**视觉信息**（含扫描页、幻灯片、图表、表单等文本与视觉内容）回答问题。
你必须仔细融合知识图谱与文档图像的信息。若存在冲突或互补信息，需结合两者进行严谨推理。
若图谱中存在明确结构化知识，不要只依赖视觉内容。

---目标---
结合知识图谱与文档图像中的所有相关信息，生成**全面、详细、完整**的回答。
不要过度简化或概括，力求覆盖最大信息量，包括不同视角、具体事实与细节洞察。
如果你不知道答案，就如实回答。不要编造信息，也不要使用没有明确依据的内容。

处理带时间戳的关系时：
1. 每条关系都有 `created_at` 时间戳，表示知识获取时间
2. 遇到冲突关系时，同时考虑语义内容与时间戳
3. 不自动优先最新创建的关系，根据上下文合理判断
4. 针对时间相关查询，优先内容中的时间信息，再考虑创建时间戳

---对话历史---
{history}

---知识图谱、文档块与页面图像---
{context_data}

---回答规则---
- 目标格式与长度：{response_type}
- 使用 Markdown 格式，合理使用章节标题
- 请使用与用户问题相同的语言回答
- 回答需与对话历史保持连贯
- 在末尾“参考文献”部分列出最多 5 个最重要的来源，清晰标注来自知识图谱(KG)、文档块(DC)或页面图像(PI)，如有路径请一并注明，格式：[KG/DC/PI] 文件路径
- 不知道答案就直接说明
- 不要编造内容，不加入知识库未提供的信息
- 额外用户指令：{user_prompt}

回答：
"""

PROMPTS["rag_two_step_response"] = """---角色---
你是专业助手，基于**知识图谱**与从文档图像中提取的**视觉信息**（含文本、图表、表单等）回答问题。

你会收到用户查询和两份独立答案：
1. 基于知识图谱的回答
2. 基于文档图像的回答

你的任务是：分析用户查询，并将两份答案**融合为一份统一、完整的回答**。
不要遗漏任何一方的相关要点。当答案冲突或互补时，使用严谨推理进行调和。
如果知识图谱提供了明确事实，除非有强视觉证据矛盾，否则不要覆盖图谱结论。

请用中文回答。

---查询---
{query}

---输入答案---
- 来自知识图谱的回答：
{kg_answer}

- 来自文档图像的回答：
{image_answer}

---目标---
结合知识图谱与文档图像的所有相关信息，生成**全面、详细、完整**的回答。
不要过度简化或概括，力求覆盖最大信息量，包括不同视角、事实细节与双方的细节洞察。
如果你不知道答案，就如实回答。不要编造信息，也不要使用没有明确依据的内容。

处理带时间戳的信息时：
1. 每条信息（关系与内容）都有 `created_at` 时间戳，表示获取时间
2. 遇到冲突信息时，同时考虑内容/关系与时间戳
3. 不自动优先最新信息，根据上下文合理判断
4. 针对时间相关查询，优先内容中的时间信息，再考虑创建时间戳

---回答规则---
- 目标格式与长度：多段落
- 生成融合双方输入的最终答案
- 使用 Markdown 格式，合理使用章节标题
- 按要点分章节组织答案，每章聚焦一个核心内容或方面
- 在末尾“参考文献”部分列出最多 5 个最重要的来源，清晰标注来自知识图谱(KG)、文档块(DC)或页面图像(PI)，如有路径请一并注明，格式：[KG/DC/PI] 文件路径
- 回答需与对话历史保持连贯
- 不知道答案就直接说明，不要编造
- 不加入输入未提供的信息
"""

PROMPTS["entity_continue_extraction"] = """
在上一轮抽取中遗漏了大量实体和关系。

---记住步骤---

1. 识别所有实体。对每个识别出的实体，提取以下信息：
- entity_name：实体名称，使用与输入文本相同的语言。如果是英文，首字母大写。
- entity_type：必须是以下类型之一：[{entity_types}]
- entity_description：对实体属性与相关行为的完整描述
每个实体格式为：("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)

2. 在步骤1识别的实体中，找出所有**存在明确关联**的（源实体, 目标实体）对。
对每对相关实体，提取以下信息：
- source_entity：源实体名称，同步骤1
- target_entity：目标实体名称，同步骤1
- relationship_description：说明你认为源实体与目标实体为何相关
- relationship_strength：数值，表示源实体与目标实体之间的关系强度
- relationship_keywords：一个或多个高层关键词，概括关系的整体性质，侧重概念或主题而非具体细节
每条关系格式为：("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_keywords>{tuple_delimiter}<relationship_strength>)

3. 提取能概括全文核心概念、主题或内容的高层关键词。这些关键词应体现文档的整体思想。
内容级关键词格式为：("content_keywords"{tuple_delimiter}<高层关键词>)

4. 用 {language} 返回步骤1和步骤2识别的所有实体与关系，组成一个列表。使用 **{record_delimiter}** 作为列表分隔符。

5. 完成后输出 {completion_delimiter}

---输出---

请使用相同格式在下方补充遗漏内容：\n
"""

PROMPTS["entity_if_loop_extraction"] = """
---目标---

看起来可能仍有部分实体被遗漏。

---输出---

仅使用 `YES` 或 `NO` 回答是否仍有需要补充的实体。
"""

PROMPTS["naive_rag_response"] = """---角色---
你是一位乐于助人的助手，根据下方提供的**文档图片**信息回应用户查询。

---目标---
针对查询生成**全面、详细、完整**的回答，
并整合来自文档图片中的所有相关信息。
不要过度简化或粗暴概括，力求**信息覆盖最大化**，
包括不同视角、详细事实与细微洞察。
若无法回答，直接如实说明。
不要编造内容，不包含无证据支持的信息。

处理带时间戳的内容时：
1. 每条内容都带有 created_at 时间戳，表示该知识的获取时间
2. 遇到冲突信息时，需同时考虑内容本身与时间戳
3. 不要自动优先选择最新内容，需根据上下文合理判断
4. 针对与特定时间相关的查询，优先考虑内容中的时间信息，再考虑创建时间戳

---对话历史---
{history}

{content_data}

---回答规则---

- 目标格式与长度：{response_type}
- 使用 Markdown 格式，并添加合适的章节标题
- 请使用与用户问题相同的语言回答
- 确保回答与对话历史保持连贯
- 在回答末尾的「参考文献」部分列出最多 5 个最重要的参考来源，
  明确标注来源是文档分块（DC）还是页面图片（PI），
  并尽可能提供文件路径，格式如下：
  [DC] 文件路径 / [PI] 图片路径
- 若无法回答，直接如实说明
- 不要包含文档分块未提供的信息
- 额外用户指令：{user_prompt}

回答："""

PROMPTS["rag_response"] = """---角色---
你是一名专业助手，负责依据**知识图谱**与从文档图片中提取的**视觉信息**回答问题。
这些图片包含相关文本与视觉内容（如扫描页、幻灯片、图表、表单等）。
你必须仔细融合来自知识图谱与文档图片的信息。
若信息存在冲突或互补情况，需结合两类来源进行严谨推理。
若知识图谱中存在相关结构化知识，不要仅依赖视觉内容。

---目标---
针对用户查询，生成**全面、详细、完整**的回答，
并整合来自**知识图谱**与**文档图片**的所有相关信息。
不要过度简化或概括，力求**信息覆盖完整**，
包括来自两类来源的不同视角、详细事实与细节洞察。
若无法回答，直接如实说明。
不要编造内容，不包含无明确证据支持的信息。

处理带时间戳的关系时：
1. 每条关系均带有 created_at 时间戳，表示该知识的获取时间
2. 遇到冲突关系时，需同时考虑语义内容与时间戳
3. 不要自动优先选择最新创建的关系，需根据上下文合理判断
4. 针对与特定时间相关的查询，优先考虑内容中的时间信息，再考虑创建时间戳

---对话历史---
{history}

---知识图谱、文档分块与页面图片---
{context_data}

---回答规则---

- 目标格式与长度：{response_type}
- 使用 Markdown 格式，并添加合适的章节标题
- 请使用与用户问题相同的语言回答
- 确保回答与对话历史保持连贯
- 在回答末尾的「参考文献」部分列出最多 5 个最重要的参考来源，
  明确标注来源是知识图谱（KG）、文档分块（DC）还是页面图片（PI），
  并尽可能提供文件路径，格式如下：[KG/DC/PI] 文件路径
- 若无法回答，直接如实说明
- 不要编造信息，不包含知识库未提供的内容
- 额外用户指令：{user_prompt}

回答："""

PROMPTS["rag_two_step_response"] = """---角色---
你是一名专业助手，负责依据**知识图谱**与从文档图片中提取的**视觉信息**回答问题。
这些图片包含相关文本与视觉内容（如扫描页、幻灯片、图表、表单等）。

你将收到用户查询以及两份独立的回答：
1. 基于知识图谱生成的回答
2. 基于文档图片生成的回答

你的任务是：分析用户查询，并将两份回答**融合为一份全面的最终回答**。
不得遗漏任一来源中的任何相关要点。
当两份答案存在冲突或互补信息时，使用严谨推理进行调和。
若知识图谱提供了明确事实，除非有强有力的视觉证据与之矛盾，否则不要覆盖图谱事实。

请使用英文回复。

---查询---
{query}

---输入答案---
- 来自知识图谱的回答：
{kg_answer}

- 来自文档图片的回答：
{image_answer}

---目标---
针对查询生成**全面、详细、完整**的回答，
融合来自知识图谱与文档图片的所有相关信息。
不要过度简化或粗暴概括，力求**信息覆盖最大化**，
包括来自两个来源的不同视角、详细事实与细节洞察。
若无法回答，直接如实说明。
不要编造内容，不包含无证据支持的信息。

处理带时间戳的信息时：
1. 每条信息（包括关系与内容）均带有 created_at 时间戳，表示知识的获取时间。
2. 遇到冲突信息时，需同时考虑内容/关系与时间戳。
3. 不要自动优先选择最新信息，需根据上下文合理判断。
4. 针对与特定时间相关的查询，优先考虑内容中的时间信息，再考虑创建时间戳。

---回答规则---
- 目标格式与长度：多段落
- 生成融合两份输入的最终答案
- 使用 Markdown 格式，并使用合适的章节标题
- 将答案按章节组织，每个章节聚焦一个核心要点或方面
- 在回答末尾的「参考文献」部分列出最多 5 个最重要的参考来源，
  明确标注来源是知识图谱（KG）、文档分块（DC）还是页面图片（PI），
  并尽可能提供文件路径，格式如下：[KG/DC/PI] 文件路径
- 确保回答与对话历史保持连贯
- 若无法回答，直接如实说明，不要编造内容
- 不包含输入未提供的信息
"""
