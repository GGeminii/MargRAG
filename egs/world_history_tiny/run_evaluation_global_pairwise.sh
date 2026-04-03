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

# 评估方法1：全局 pairwise 对比评估
# - Answer 1: results_mmkg_debate.json
# - Answer 2: results.json
python3 ../utils/query_evaluation.py \
    --config-file ./conf/addon_params_evaluation.yaml \
    --mode global_pairwise \
    --ours-result-file ./exp/World_History_Volume_1/results/results_mmkg_debate.json \
    --baseline-result-file ./exp/World_History_Volume_1/results/results.json \
    --output-file ./exp/World_History_Volume_1/evaluation/global_pairwise_eval.json

# 针对全局评估结果生成图表分析
python3 ../utils/visualize_evaluation.py \
    --global-eval-file ./exp/World_History_Volume_1/evaluation/global_pairwise_eval.json \
    --output-dir ./exp/World_History_Volume_1/evaluation/figures \
    --summary-file ./exp/World_History_Volume_1/evaluation/figures/eval_visual_summary_global.json

