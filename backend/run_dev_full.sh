#!/bin/bash
# KBP Retrosynthesis Backend — 완전 초기화 스크립트
# 기존 conda 환경(기본 이름: kbp-backend)이 없으면 새로 만들고,
# retro_star/LocalRetro 의존성을 전부 설치한 뒤 FastAPI 백엔드를 시작한다.
# 이미 환경이 갖춰져 있다면 run_dev.sh만 써도 된다.

set -e

echo "📦 KBP Retrosynthesis Backend — Full Setup"
echo "=================================================="

CONDA_ENV_NAME="${CONDA_ENV_NAME:-kbp-backend}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "📁 Project root: $PROJECT_ROOT"
cd "$PROJECT_ROOT"

# ─────────────────────────────────────────────────────
# 1. conda 환경 생성/활성화
# ─────────────────────────────────────────────────────

CONDA_BASE="$(conda info --base 2>/dev/null)"
if [ -z "$CONDA_BASE" ]; then
    echo "❌ conda를 찾을 수 없습니다. miniconda/anaconda를 먼저 설치하세요."
    exit 1
fi
source "$CONDA_BASE/etc/profile.d/conda.sh"

if ! conda env list | grep -q "^$CONDA_ENV_NAME "; then
    echo "📝 Creating conda env: $CONDA_ENV_NAME (python 3.9)..."
    conda create -y -n "$CONDA_ENV_NAME" python=3.9
fi
conda activate "$CONDA_ENV_NAME"
echo "✅ conda env activated: $CONDA_ENV_NAME ($(python --version))"

# ─────────────────────────────────────────────────────
# 2. graphviz 바이너리 (sudo 없는 서버 대비 conda-forge로 설치)
# ─────────────────────────────────────────────────────

if ! command -v dot &> /dev/null; then
    echo "📥 Installing graphviz binary via conda-forge (no sudo needed)..."
    conda install -y -c conda-forge graphviz
fi

# ─────────────────────────────────────────────────────
# 3. retro_star 의존성 설치 (editable mode)
# ─────────────────────────────────────────────────────

echo "📥 Installing retro_star dependencies..."

if [ -d "$PROJECT_ROOT/KBP-main/retro_star/retro_star/packages/mlp_retrosyn" ]; then
    echo "  Installing mlp_retrosyn..."
    pip install -e "$PROJECT_ROOT/KBP-main/retro_star/retro_star/packages/mlp_retrosyn"
else
    echo "  ⚠️  mlp_retrosyn not found (optional)"
fi

if [ -d "$PROJECT_ROOT/KBP-main/retro_star/retro_star/packages/rdchiral" ]; then
    echo "  Installing rdchiral..."
    pip install -e "$PROJECT_ROOT/KBP-main/retro_star/retro_star/packages/rdchiral"
else
    echo "  ⚠️  rdchiral not found (optional)"
fi

if [ -d "$PROJECT_ROOT/KBP-main/retro_star" ]; then
    echo "  Installing retro_star..."
    pip install -e "$PROJECT_ROOT/KBP-main/retro_star"
else
    echo "  ❌ retro_star not found!"
    exit 1
fi

# ─────────────────────────────────────────────────────
# 4. chemprop 설치 (선택)
# ─────────────────────────────────────────────────────

if [ -d "$PROJECT_ROOT/KBP-main/chemprop" ]; then
    echo "📥 Installing chemprop..."
    pip install -e "$PROJECT_ROOT/KBP-main/chemprop"
    echo "✅ chemprop installed"
else
    echo "ℹ️  chemprop not found (optional)"
fi

# ─────────────────────────────────────────────────────
# 5. FastAPI 백엔드 의존성 설치
# ─────────────────────────────────────────────────────
# requirements.txt에 dgl==0.9.1/dgllife==0.2.8이 포함돼 있음. 이 버전은 torch==1.9.0
# 전용으로 검증된 조합이라, 다른 버전을 pip가 멋대로 끌어오지(특히 torch 업그레이드)
# 않도록 torch를 먼저 깔고 나머지를 설치한다.

echo "📥 Installing FastAPI backend dependencies..."
pip install --upgrade pip
pip install "torch==1.9.0"
pip install -r "$SCRIPT_DIR/requirements.txt"
echo "✅ Backend dependencies installed"

# ─────────────────────────────────────────────────────
# 6. 환경 변수 파일 확인
# ─────────────────────────────────────────────────────

if [ ! -f "$SCRIPT_DIR/.env" ]; then
    echo "⚠️  .env file not found. Creating from template..."
    cp "$SCRIPT_DIR/.env.example" "$SCRIPT_DIR/.env"
    echo "✅ .env file created. Edit if needed."
fi

# ─────────────────────────────────────────────────────
# 7. 의존성 확인
# ─────────────────────────────────────────────────────

echo "🔍 Checking dependencies..."

python -c "import fastapi; print('  ✅ fastapi')" || echo "  ❌ fastapi"
python -c "import torch; print('  ✅ torch', torch.__version__)" || echo "  ❌ torch"
python -c "import rdkit; print('  ✅ rdkit')" || echo "  ❌ rdkit"
python -c "import dgl; print('  ✅ dgl')" || echo "  ⚠️  dgl (localretro/interretro will fail without it)"
python -c "import retro_star; print('  ✅ retro_star')" || echo "  ⚠️  retro_star (will be imported at runtime)"

# ─────────────────────────────────────────────────────
# 8. FastAPI 서버 시작
# ─────────────────────────────────────────────────────

echo ""
echo "🚀 Starting FastAPI server..."
echo "📍 API will be available at: http://localhost:8000"
echo "📚 Documentation at: http://localhost:8000/docs"
echo "❌ Press Ctrl+C to stop"
echo ""

# 환경 변수 로드 (주석/빈 줄 안전하게 처리)
set -a
source "$SCRIPT_DIR/.env"
set +a
export PYTHONUNBUFFERED=1

# FastAPI 실행
cd "$SCRIPT_DIR"
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
