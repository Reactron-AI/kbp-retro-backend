#!/bin/bash
# InterRetro 전용 인스턴스 상시 실행 스크립트 (systemd 서비스에서 사용, 포트 8002).
# run_dev_interretro.sh와 동일하지만 --reload를 뺐다 (run_prod.sh와 같은 이유:
# 상시 서비스에는 파일 감시가 불필요하고, /mnt/d의 DrvFs mtime 오탐지로 인한
# 주기적 재시작을 피하기 위함 — KBP-condition/inference/run_prod.sh 참고) 그리고
# gunicorn 대신 uvicorn을 직접 실행한다 (run_prod.sh와 같은 이유: gunicorn이 fork한
# 워커 프로세스 안에서 CUDA 초기화가 "No CUDA GPUs are available"로 실패하는 문제가
# GPU_ID를 켰을 때 재현됨 — backend/README.md 참고).

set -e

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

# dgl의 graphbolt 확장이 libnvrtc.so.12/libcublas.so.12 등을 시스템 전역
# ldconfig 캐시에서 못 찾으면 "Cannot load Graphbolt C++ library"로 깨진다.
# conda-forge로 설치한 cuda-nvrtc/cuda-cudart/libcublas가 env 안에는 있지만
# (kbp-backend 때의 cudatoolkit=11.1과 달리) activate.d에서 자동으로
# LD_LIBRARY_PATH에 잡히지 않아서 명시적으로 붙여줘야 한다.
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

set -a
source .env.interretro
set +a
export PYTHONUNBUFFERED=1

exec python -m uvicorn app.main:app \
    --host 0.0.0.0 \
    --port 8002 \
    --timeout-keep-alive 1800
