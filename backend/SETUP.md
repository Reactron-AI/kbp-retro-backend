# 🔧 KBP Backend 초기화 — 정확한 단계별 가이드

> ⚠️ **중요:** 사용자가 제시한 설치 명령어들을 모두 포함하는 가이드입니다.

## 📋 사전 요구사항

### 시스템 패키지

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y python3.9 python3-pip python3-venv graphviz

# 또는 사용자가 제시한 명령어
apt-get update
apt-get install graphviz hdbscan
```

### Python 버전

```bash
python3 --version  # 3.7 이상 필요
# 예: Python 3.9.x
```

---

## 🚀 초기화 (3가지 방법)

### 방법 1️⃣: 자동 스크립트 (권장)

```bash
cd /mnt/vast/skbp/nhkim/KBP/KBP-main/backend

# 전체 초기화 + 서버 시작
bash run_dev_full.sh
```

**자동 수행 항목:**
- ✅ 시스템 패키지 확인/설치
- ✅ 가상 환경 생성
- ✅ retro_star 패키지들 설치 (editable mode)
- ✅ chemprop 설치
- ✅ FastAPI 백엔드 의존성 설치
- ✅ 서버 시작

---

### 방법 2️⃣: 수동 단계별 (디버깅용)

#### Step 1: 시스템 패키지 설치

```bash
# 사용자 제시 명령어 실행
sudo apt-get update
sudo apt-get install graphviz hdbscan

# 또는 개별적으로
sudo apt-get install -y python3-dev build-essential
```

#### Step 2: 프로젝트 루트로 이동

```bash
cd /mnt/vast/skbp/nhkim/KBP/KBP-main
ls -la
# 다음이 모두 보여야 함:
# - retro_star/          (retro_star 코드)
# - chemprop/            (chemprop 코드)
# - backend/             (FastAPI 백엔드)
```

#### Step 3: retro_star 패키지 설치 (editable mode)

```bash
# 1. mlp_retrosyn
pip install -e retro_star/packages/mlp_retrosyn

# 2. rdchiral
pip install -e retro_star/packages/rdchiral

# 3. retro_star 자체
pip install -e retro_star

# 검증
python3 -c "import retro_star; print('✅ retro_star imported')"
```

#### Step 4: 추가 머신러닝 라이브러리 설치

```bash
# 사용자 제시 명령어
pip install rdkit umap-learn more-itertools torch-geometric lightning

# 또는 필요한 것만
pip install rdkit umap-learn
```

#### Step 5: chemprop 설치

```bash
cd chemprop
pip install -e .
cd ..

# 검증
python3 -c "import chemprop; print('✅ chemprop imported')" || echo "Note: chemprop optional"
```

#### Step 6: FastAPI 백엔드 설정

```bash
cd backend

# 가상 환경 생성 (선택)
python3 -m venv venv
source venv/bin/activate

# 환경 변수 파일 생성
cp .env.example .env

# 의존성 설치
pip install -r requirements.txt
```

#### Step 7: 서버 시작

```bash
# 개발 모드
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# 또는 프로덕션 모드
gunicorn \
  --workers 2 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  app.main:app
```

---

### 방법 3️⃣: Docker (프로덕션용)

```dockerfile
FROM python:3.9-slim

# 시스템 패키지
RUN apt-get update && apt-get install -y \
    graphviz hdbscan \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# retro_star 설치
COPY retro_star/packages ./retro_star/packages
COPY retro_star ./retro_star
RUN pip install -e retro_star/packages/mlp_retrosyn && \
    pip install -e retro_star/packages/rdchiral && \
    pip install -e retro_star

# chemprop 설치
COPY chemprop ./chemprop
RUN pip install -e chemprop

# 백엔드 설치
COPY backend ./backend
RUN pip install -r backend/requirements.txt

WORKDIR /app/backend
CMD ["gunicorn", "--workers", "2", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "app.main:app"]
```

---

## 📊 초기화 순서도

```
┌─────────────────────────┐
│ 시스템 패키지 설치      │ (apt-get)
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│ retro_star 의존성       │ (pip install -e)
├─ mlp_retrosyn          │
├─ rdchiral              │
└─ retro_star
             ▼
┌─────────────────────────┐
│ 추가 ML 라이브러리      │ (pip install)
├─ rdkit                 │
├─ umap-learn            │
├─ torch-geometric       │
└─ lightning
             ▼
┌─────────────────────────┐
│ chemprop 설치           │ (pip install -e)
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│ FastAPI 백엔드 설정     │
├─ pip install -r requirements.txt
├─ .env 설정
└─ uvicorn 시작
             ▼
┌─────────────────────────┐
│ ✅ 서버 준비 완료       │
│ http://localhost:8000   │
└─────────────────────────┘
```

---

## ✅ 검증 체크리스트

### 1. 모든 패키지 import 가능?

```bash
python3 << 'EOF'
imports = {
    'fastapi': 'FastAPI',
    'uvicorn': 'Uvicorn',
    'pydantic': 'Pydantic',
    'torch': 'PyTorch',
    'rdkit': 'RDKit',
    'retro_star': 'RetroStar',
    'chemprop': 'ChemProp',
    'networkx': 'NetworkX',
    'graphviz': 'Graphviz'
}

for module, name in imports.items():
    try:
        __import__(module)
        print(f'✅ {name}')
    except ImportError as e:
        print(f'❌ {name}: {e}')
EOF
```

### 2. 포트 사용 가능?

```bash
# 8000 포트 확인
netstat -an | grep 8000
# (아무것도 표시 안 되면 사용 가능)

# 또는
lsof -i :8000
# (아무것도 표시 안 되면 사용 가능)
```

### 3. API 응답 확인?

```bash
# 터미널에서 서버 시작 후, 다른 터미널에서:
curl http://localhost:8000/health

# 예상 응답
# {"status":"ok","version":"1.0.0","planner_ready":true,"gpu_available":false}
```

---

## 🐛 문제 해결

### "No module named retro_star"

**원인:** retro_star 설치 누락

**해결:**
```bash
pip install -e retro_star/
# 또는
pip install -e /mnt/vast/skbp/nhkim/KBP/KBP-main/retro_star/
```

### "graphviz command not found"

**원인:** 시스템 패키지 미설치

**해결:**
```bash
sudo apt-get install graphviz
# 또는 Mac
brew install graphviz
```

### "CUDA out of memory"

**원인:** GPU 메모리 부족

**해결:** `.env` 파일에서
```env
GPU_ID=-1  # CPU 사용으로 변경
```

### "Port 8000 already in use"

**원인:** 다른 프로세스가 포트 사용 중

**해결:**
```bash
# 기존 프로세스 종료
kill -9 $(lsof -t -i :8000)

# 또는 다른 포트 사용
uvicorn app.main:app --port 8001
```

---

## 📚 요약: 사용자 제시 명령어 ↔ 내 구현

| 사용자 명령어 | 내 구현 위치 | 상태 |
|--------------|-----------|------|
| `pip install -e retro_star/packages/mlp_retrosyn` | `run_dev_full.sh` | ✅ 포함 |
| `pip install -e retro_star/packages/rdchiral` | `run_dev_full.sh` | ✅ 포함 |
| `pip install -e .` | `run_dev_full.sh` | ✅ 포함 |
| `pip install rdkit umap-learn ...` | `requirements.txt` | ✅ 주석 처리 |
| `apt-get update` | `run_dev_full.sh` | ✅ 포함 |
| `apt-get install graphviz hdbscan` | `run_dev_full.sh` | ✅ 포함 |
| `cd chemprop` | `run_dev_full.sh` | ✅ 포함 |
| `pip install -e .` | `run_dev_full.sh` | ✅ 포함 |

---

## 🎯 권장 실행 방법

### 개발 (로컬)

```bash
# 전체 자동 설치 + 서버 시작
bash run_dev_full.sh
```

### 프로덕션 (서버)

```bash
# 수동으로 각 단계 수행 후
gunicorn --workers 2 --worker-class uvicorn.workers.UvicornWorker app.main:app
```

### CI/CD (Docker)

```bash
docker build -t kbp-backend .
docker run -p 8000:8000 kbp-backend
```

---

**이제 사용자가 제시한 명령어들이 모두 포함되었습니다!** ✅
