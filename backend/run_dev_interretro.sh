#!/bin/bash
# InterRetro 전용 인스턴스 시작 스크립트 (포트 8002)
# kbp-backend와 별도 conda env(kbp-interretro, torch==2.2.1)를 사용한다.
# requirements-interretro.txt는 설치 기록용이며 여기서 pip install -r을
# 돌리지 않는다 — torch/dgl이 특수 --index-url/-f로 설치돼 있어 일반
# pip install -r을 돌리면 인덱스가 안 맞아 깨질 수 있다.

set -e

echo "📦 KBP Retrosynthesis Backend - InterRetro Instance"
echo "===================================================="

CONDA_ENV_NAME="kbp-interretro"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -f ".env.interretro" ]; then
    echo "❌ .env.interretro 파일이 없습니다."
    exit 1
fi

CONDA_BASE="$(conda info --base 2>/dev/null)"
if [ -z "$CONDA_BASE" ]; then
    echo "❌ conda를 찾을 수 없습니다."
    exit 1
fi
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_NAME"
echo "✅ conda env activated: $CONDA_ENV_NAME ($(python --version))"

echo ""
echo "🚀 Starting FastAPI server..."
echo "📍 API will be available at: http://localhost:8002"
echo "📚 Documentation at: http://localhost:8002/docs"
echo "❌ Press Ctrl+C to stop"
echo ""

set -a
source .env.interretro
set +a
export PYTHONUNBUFFERED=1

uvicorn app.main:app --host 0.0.0.0 --port 8002 --reload
