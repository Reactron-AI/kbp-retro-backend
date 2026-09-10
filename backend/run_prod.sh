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

# 단일 uvicorn 프로세스로 직접 실행 (gunicorn 미사용)
echo "🚀 Starting production server with Uvicorn..."

# workers=1이 항상 필수이므로(플래너/job 상태가 프로세스 메모리에만 있음 — 위 주석 참고,
# 이제는 DEPLOYMENT.md 참고) gunicorn의 다중 워커 관리가 애초에 필요 없다. GPU_ID를 설정한
# 경우(GPU_ID != -1) gunicorn의 fork로 생성된 워커 프로세스 안에서 CUDA 초기화가
# "No CUDA GPUs are available"로 실패하는 문제가 있어(WSL2 CUDA 패스스루 환경에서 재현,
# 순수 uvicorn/단일 프로세스에서는 재현 안 됨) gunicorn 자체를 제거했다.
exec python -m uvicorn app.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --timeout-keep-alive 1800
