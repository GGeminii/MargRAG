#!/usr/bin/env bash
set -euo pipefail

# 始终以脚本所在目录为基准，避免相对路径在不同 cwd 下失效
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

# 加载环境变量（OPENAI API 等）
source "${SCRIPT_DIR}/../../env.sh"

# 必填环境变量检查：评估阶段需要调用 LLM 裁判
: "${OPENAI_API_KEY:?Set OPENAI_API_KEY in env.sh}"

# 进入 world_history 目录，保证配置中的相对路径可用
cd "${SCRIPT_DIR}"

# 评估方法2：本地 correctness 评估
# - result-file：模型输出结果
# - gt-file：标准答案文件
python3 ../utils/query_evaluation.py \
    --config-file ./conf/addon_params_evaluation.yaml \
    --mode local_correctness \
    --result-file ./exp/World_History_Volume_1/results/results_mmkg_debate.json \
    --gt-file ./data/local_qa_gt.json \
    --output-file ./exp/World_History_Volume_1/evaluation/local_correctness_eval.json

# 针对本地评估结果生成图表分析
python3 ../utils/visualize_evaluation.py \
    --local-eval-file ./exp/World_History_Volume_1/evaluation/local_correctness_eval.json \
    --output-dir ./exp/World_History_Volume_1/evaluation/figures \
    --summary-file ./exp/World_History_Volume_1/evaluation/figures/eval_visual_summary_local.json

