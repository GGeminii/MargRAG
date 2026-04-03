"""
功能说明：
    统一维护实验评估提示词模板（Prompt）。

说明：
    - 评估方法1：全局 QA 两两对比（无标准答案）。
    - 评估方法2：本地 QA 正确性判定（有标准答案）。
"""

# 评估提示词字典：按方法名管理，便于在 evaluation.py 中按 key 调用
EVAL_PROMPTS: dict[str, str] = {}


EVAL_PROMPTS["global_pairwise"] = """You will evaluate two answers to the same question based on three criteria: Comprehensiveness, Diversity, and Empowerment.

- Comprehensiveness: How much detail does the answer provide to cover all aspects and details of the question?
- Diversity: How varied and rich is the answer in providing different perspectives and insights on the question?
- Empowerment: How well does the answer help the reader understand and make informed judgments about the topic?

For each criterion, choose the better answer (either Answer 1 or Answer 2) and explain why. Then, select an overall winner based on these three categories.

Here is the question:
{query}

Here are the two answers:
Answer 1 Start:
{answer1}
Answer 1 End

Answer 2 Start:
{answer2}
Answer 2 End

Evaluate both answers using the three criteria listed above and provide detailed explanations for each criterion.

Output your evaluation in the following JSON format:
{{
  "Comprehensiveness": {{
    "Winner": "[Answer 1 or Answer 2]",
    "Explanation": "[Provide explanation here]"
  }},
  "Diversity": {{
    "Winner": "[Answer 1 or Answer 2]",
    "Explanation": "[Provide explanation here]"
  }},
  "Empowerment": {{
    "Winner": "[Answer 1 or Answer 2]",
    "Explanation": "[Provide explanation here]"
  }},
  "Overall Winner": {{
    "Winner": "[Answer 1 or Answer 2]",
    "Explanation": "[Summarize why this answer is the overall winner based on the three criteria]"
  }}
}}

Important:
1. Return JSON only.
2. Do not add markdown code fences.
3. Do not include extra keys.
"""


EVAL_PROMPTS["local_correctness"] = """You are given a question, the model's response, and the correct answer.

Your task is to evaluate whether the model's response correctly answers the question based on the correct answer provided.

Please follow this format in your output:
{{
  "is_correct": "yes" or "no",
  "reason": "Your explanation of why the response is correct or incorrect."
}}

Make sure your judgment is based only on the given answer, and explain your reasoning clearly and concisely.

Here is the input:
Question: {query}
Model's Response: {result}
Correct Answer: {answer}

Important:
1. Return JSON only.
2. Do not add markdown code fences.
3. Do not include extra keys.
"""

