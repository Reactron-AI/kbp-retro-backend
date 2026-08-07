#!/bin/bash
# 로컬 개발 환경 시작 스크립트
# 사용자가 미리 만들어둔 conda 환경(kbp-backend)을 사용한다.
# (conda run으로 띄우면 stdout이 버퍼링돼서 로그가 안 보이고, PATH도 활성화되지
#  않아 graphviz의 'dot' 바이너리를 못 찾는 문제가 있어 conda activate를 직접 사용)

set -e  # 에러 발생 시 종료

echo "📦 KBP Retrosynthesis Backend - Development Server"
echo "=================================================="

CONDA_ENV_NAME="${CONDA_ENV_NAME:-kbp-backend}"

# 디렉토리 설정
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 환경 변수 파일 확인
if [ ! -f ".env" ]; then
    echo "⚠️  .env 파일이 없습니다. .env.example에서 복사합니다..."
    cp .env.example .env
    echo "✅ .env 파일이 생성되었습니다. 필요시 수정하세요."
fi

# conda 환경 활성화 (PATH에 graphviz 'dot' 등 env 바이너리가 잡히도록 activate 사용)
CONDA_BASE="$(conda info --base 2>/dev/null)"
if [ -z "$CONDA_BASE" ]; then
    echo "❌ conda를 찾을 수 없습니다."
    exit 1
fi
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_NAME"
echo "✅ conda env activated: $CONDA_ENV_NAME ($(python --version))"

# 의존성 확인 (이미 설치돼 있으면 빠르게 통과)
echo "📥 Checking dependencies..."
pip install -q -r requirements.txt

# 서버 시작
echo ""
echo "🚀 Starting FastAPI server..."
echo "📍 API will be available at: http://localhost:8000"
echo "📚 Documentation at: http://localhost:8000/docs"
echo "❌ Press Ctrl+C to stop"
echo ""

# 환경 변수 로드 (주석/빈 줄 안전하게 처리)
set -a
source .env
set +a
export PYTHONUNBUFFERED=1

# FastAPI 실행
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
