#!/bin/bash
# 프로덕션 배포 스크립트 (Gunicorn + Uvicorn)

set -e

echo "📦 KBP Retrosynthesis Backend - Production Server"
echo "=================================================="

CONDA_ENV_NAME="${CONDA_ENV_NAME:-kbp-backend}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# conda 환경 활성화 (graphviz 'dot' 등 env 바이너리가 PATH에 잡히도록)
CONDA_BASE="$(conda info --base 2>/dev/null)"
if [ -z "$CONDA_BASE" ]; then
    echo "❌ conda를 찾을 수 없습니다."
    exit 1
fi
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_NAME"
echo "✅ conda env activated: $CONDA_ENV_NAME ($(python --version))"

# 환경 변수 로드 (주석/빈 줄 안전하게 처리)
if [ -f ".env" ]; then
    set -a
    source .env
    set +a
    export PYTHONUNBUFFERED=1
else
    echo "⚠️  .env 파일을 찾을 수 없습니다."
    exit 1
fi

# 로그 디렉토리 생성
mkdir -p ./logs

# Gunicorn 실행
echo "🚀 Starting production server with Gunicorn..."

# workers=1 필수: planner 모델과 job 상태가 프로세스 메모리에만 있어서,
# 워커가 여러 개면 각자 모델을 따로 로드하고(메모리/시간 낭비) job_id도
# 등록한 워커에서만 조회 가능해짐(로드밸런서가 다른 워커로 보내면 404).
gunicorn \
    --workers 1 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000 \
    --access-logfile ./logs/access.log \
    --error-logfile ./logs/error.log \
    --log-level info \
    --timeout 1800 \
    "app.main:app"
