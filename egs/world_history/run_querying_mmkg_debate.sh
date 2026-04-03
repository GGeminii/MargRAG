#!/usr/bin/env bash
set -euo pipefail

# 始终以脚本所在目录为基准，避免从其他目录执行时相对路径失效
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

# 加载环境变量（OPENAI_API_KEY、MINERU_PATH 等）
source "${SCRIPT_DIR}/../../env.sh"

# 必填环境变量检查：若未设置则立即报错退出
: "${OPENAI_API_KEY:?Set OPENAI_API_KEY in env.sh}"

# 进入 world_history 示例目录，确保下面的相对路径正确
cd "${SCRIPT_DIR}"

# 执行 MMKG Debate 查询示例：
python3 ../utils/query_mmkg_debate.py \
    --config-file ./conf/addon_params_mmkg_debate.yaml \
    --working-dir ./exp/World_History_Volume_1 \
    --input-queries './data/queries.txt' \
    --query-mode mmkg_debate \
    --output-file './exp/World_History_Volume_1/results/results_mmkg_debate.json' \
    --concurrency 12

