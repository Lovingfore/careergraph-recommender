#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

TRAIN_MODEL=0
START_WEB=0
SKIP_INSTALL=0
REFRESH_FROM_RAW=0
LISTEN_ADDRESS="127.0.0.1"
PORT=8000

usage() {
  cat <<'EOF'
用法：bash deploy.sh [选项]

选项：
  --train-model              安装可选模型依赖并训练 TemporalGAT
  --start-web                完成流水线后启动 Django Web 服务
  --skip-install             跳过 pip 安装，直接复用当前虚拟环境
  --refresh-from-raw         使用 data/raw 中的 O*NET 原始文件重建 clean 数据
  --host ADDRESS             Web 监听地址（默认 127.0.0.1）
  --port PORT                Web 端口（默认 8000）
  -h, --help                显示帮助
EOF
}

while (($# > 0)); do
  case "$1" in
    --train-model) TRAIN_MODEL=1 ;;
    --start-web) START_WEB=1 ;;
    --skip-install) SKIP_INSTALL=1 ;;
    --refresh-from-raw) REFRESH_FROM_RAW=1 ;;
    --host) shift; LISTEN_ADDRESS="${1:?--host 需要地址}" ;;
    --port) shift; PORT="${1:?--port 需要端口}" ;;
    -h|--help) usage; exit 0 ;;
    *) echo "未知参数：$1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

if [[ ! "$PORT" =~ ^[0-9]+$ ]] || ((PORT < 1 || PORT > 65535)); then
  echo "端口必须是 1-65535 的整数：$PORT" >&2
  exit 2
fi

if [[ -x ".venv/bin/python" ]]; then
  PYTHON="$(cd .venv/bin && pwd)/python"
  if ! "$PYTHON" -c 'import sys; assert sys.version_info >= (3, 10)' >/dev/null 2>&1; then
    echo "The existing .venv Python must be version 3.10 or newer." >&2
    exit 1
  fi
else
  if ((SKIP_INSTALL == 1)); then
    echo "未找到 .venv，不能跳过依赖安装。请移除 --skip-install 以创建环境并安装依赖。" >&2
    exit 1
  fi
  SYSTEM_PYTHON=""
  for candidate in python3 python; do
    if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c 'import sys; assert sys.version_info >= (3, 10)' >/dev/null 2>&1; then
      SYSTEM_PYTHON="$candidate"
      break
    fi
  done
  if [[ -z "$SYSTEM_PYTHON" ]]; then
    echo "未找到 Python 3.10 或更高版本，请先安装 Python 并确保其在 PATH 中。" >&2
    exit 1
  fi
  echo "==> 创建虚拟环境：.venv"
  "$SYSTEM_PYTHON" -m venv .venv
  PYTHON="$(cd .venv/bin && pwd)/python"
fi

run_step() {
  local description="$1"
  shift
  echo
  echo "==> $description"
  "$PYTHON" "$@"
}

if ((SKIP_INSTALL == 0)); then
  run_step "安装基础依赖" -m pip install --disable-pip-version-check -r requirements.txt
  if ((TRAIN_MODEL == 1)); then
    run_step "安装可选模型依赖" -m pip install --disable-pip-version-check -r requirements-optional-models.txt
  fi
fi

if ((REFRESH_FROM_RAW == 1)) || [[ ! -f "data/clean/occupations.csv" ]] || [[ ! -f "data/clean/skills.csv" ]] || [[ ! -f "data/clean/occupation_skill.csv" ]] || [[ ! -f "data/clean/resumes_zh.csv" ]] || [[ ! -f "data/clean/job_transitions.csv" ]] || [[ ! -f "data/clean/user_profiles.csv" ]] || [[ ! -f "data/clean/user_skill_events.csv" ]]; then
  run_step "准备数据集" src/prepare_dataset.py
else
  echo
  echo "==> Reuse checked-in clean dataset (use --refresh-from-raw to rebuild)"
fi
run_step "生成技能预测 baseline" src/forecast_skills.py
run_step "构建特征" src/build_features.py
run_step "评估推荐结果" src/evaluate_recommendation.py
run_step "初始化 SQLite 数据库" src/init_database.py --db-path artifacts/topic17.sqlite3 --data-dir data/clean
run_step "构建职位-技能二部图" src/bipartite_graph.py --data-dir data/clean --out-dir data/processed/bipartite --min-demand-weight 0.30
if ((TRAIN_MODEL == 1)); then
  run_step "训练 TemporalGAT" src/train_temporal_gat.py --data-dir data/clean --artifact-dir artifacts/models --epochs 30 --seed 42
fi
run_step "执行阶段验收" src/verify_stage1.py --db-path artifacts/topic17.sqlite3

echo
echo "部署准备完成。"
if ((START_WEB == 1)); then
  if [[ "$LISTEN_ADDRESS" != "127.0.0.1" && "$LISTEN_ADDRESS" != "localhost" ]]; then
    export CAREERGRAPH_ALLOW_NETWORK=1
    echo "Network binding enabled for this process (Django ALLOWED_HOSTS=*)"
  fi
  echo "启动 Web：http://${LISTEN_ADDRESS}:${PORT}/"
  exec "$PYTHON" web/manage.py runserver "${LISTEN_ADDRESS}:${PORT}"
else
  if [[ "$LISTEN_ADDRESS" != "127.0.0.1" && "$LISTEN_ADDRESS" != "localhost" ]]; then
    printf '启动 Web：CAREERGRAPH_ALLOW_NETWORK=1 "%s" web/manage.py runserver %s:%s\n' "$PYTHON" "$LISTEN_ADDRESS" "$PORT"
  else
    printf '启动 Web："%s" web/manage.py runserver %s:%s\n' "$PYTHON" "$LISTEN_ADDRESS" "$PORT"
  fi
  echo "如需直接启动，请添加参数：--start-web"
fi
