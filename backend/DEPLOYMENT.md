# KBP Retrosynthesis API — 설정 및 배포 가이드

## 📋 목차

1. [프로젝트 구조](#프로젝트-구조)
2. [로컬 개발 환경](#로컬-개발-환경)
3. [API 명세](#api-명세)
4. [프론트엔드 통합](#프론트엔드-통합)
5. [SSH 서버 배포](#ssh-서버-배포)
6. [문제 해결](#문제-해결)

---

## 프로젝트 구조

```
KBP-main/
├── backend/                          # FastAPI 백엔드
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                  # FastAPI 애플리케이션
│   │   └── models.py                # Pydantic 모델
│   ├── retro_wrapper/
│   │   ├── __init__.py
│   │   └── planner.py               # Retro* 래퍼
│   ├── requirements.txt
│   ├── .env.example
│   ├── run_dev.sh                   # 개발 서버 실행
│   └── run_prod.sh                  # 프로덕션 서버 실행
│
├── retro_star/                       # 기존 Retro* 코드 (수정 없음)
│   ├── example.py
│   ├── retro_star/
│   └── ...
│
└── site/                             # 프론트엔드
    ├── retrosynthesis-api.html      # API 통합 버전 (NEW)
    ├── api-client.js                # API 클라이언트 (NEW)
    ├── sidebar.js
    └── ...
```

---

## 로컬 개발 환경

### 1️⃣ 준비 작업

```bash
# 1. 프로젝트 디렉토리로 이동
cd /mnt/vast/skbp/nhkim/KBP/KBP-main

# 2. Conda 환경 확인 (선택)
# retro_star 환경이 있다면:
conda activate retro_star_env
```

### 2️⃣ 백엔드 시작

```bash
# 1. 백엔드 디렉토리로 이동
cd backend

# 2. 환경 변수 파일 생성
cp .env.example .env

# 3. .env 파일 편집 (필요시)
# - PLANNER_MODEL: 'interretro' (기본값, 권장)
# - GPU_ID: -1 (CPU) 또는 GPU 번호 (0, 1, ...)
# - ITERATIONS: 500 (기본값)
# - ALLOWED_ORIGINS: 프론트엔드 URL 추가

# 4. 개발 서버 실행 (자동으로 pip install 실행)
bash run_dev.sh

# 또는 수동으로:
# python3 -m venv venv
# source venv/bin/activate
# pip install -r requirements.txt
# uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

**예상 출력:**
```
[Planner] Initializing RSPlanner with interretro...
[Planner] GPU: -1, iterations: 500, topk: 50
[Planner] RSPlanner initialized successfully!
✅ Planner initialized successfully!
🚀 Uvicorn running on http://0.0.0.0:8000
```

### 3️⃣ API 확인

```bash
# 헬스 체크
curl http://localhost:8000/health

# API 문서 (웹 브라우저)
http://localhost:8000/docs

# API 정보
curl http://localhost:8000/api/v1/info
```

### 4️⃣ 프론트엔드 실행

**옵션 A: Live Server (VS Code)**
```
1. VS Code에서 site/ 폴더 열기
2. retrosynthesis-api.html 우클릭
3. "Open with Live Server" 선택
4. http://localhost:5500 에서 접근
```

**옵션 B: Python 간단한 HTTP 서버**
```bash
cd site/
python3 -m http.server 5500
# http://localhost:5500/retrosynthesis-api.html
```

**옵션 C: 직접 파일 열기**
```bash
# 브라우저에서:
file:///mnt/vast/skbp/nhkim/KBP/site/retrosynthesis-api.html
```

### 5️⃣ 첫 계획 실행

1. **웹페이지 접속** → `http://localhost:5500/retrosynthesis-api.html`
2. **SMILES 입력** (또는 기본값 사용)
3. **"Find Synthetic Pathways" 클릭**
4. **결과 확인** (5-10분 소요, 설정에 따라)

---

## API 명세

### Base URL
```
http://localhost:8000  (개발)
http://your-server.com:8000  (프로덕션)
```

### 엔드포인트

#### `POST /api/v1/plan` - 합성 계획 요청

**요청:**
```json
{
  "target_mol": "CN1C=C(C2=NC(NC3=CC(NC(C=C)=O)=C(N(CCN(C)C)C)C=C3OC)=NC=C2)C4=CC=CC=C41",
  "planner_model": "interretro",
  "iterations": 500,
  "expansion_topk": 50,
  "include_bb": null,
  "exclude_bb": null
}
```

**응답 (성공):**
```json
{
  "success": true,
  "target_mol": "CN1C=C(...)",
  "time": 125.5,
  "iterations": 342,
  "routes": [
    {
      "steps": [
        {
          "step": 0,
          "product": "CN1C=C(...)",
          "reactants": ["A", "B"],
          "cost": 1.2
        }
      ]
    }
  ],
  "route_cost": 5.2,
  "route_length": 4
}
```

**응답 (실패):**
```json
{
  "success": false,
  "target_mol": "CN1C=C(...)",
  "time": 300.0,
  "error": "Could not find synthesis route..."
}
```

#### `GET /health` - 서버 상태 확인

**응답:**
```json
{
  "status": "ok",
  "version": "1.0.0",
  "planner_ready": true,
  "gpu_available": false
}
```

#### `GET /api/v1/info` - API 정보

**응답:**
```json
{
  "api_version": "1.0.0",
  "service": "KBP Retrosynthesis Planner",
  "supported_planners": ["mlp", "localretro", "interretro"],
  "endpoints": {
    "plan": "/api/v1/plan",
    "health": "/health",
    "docs": "/docs"
  }
}
```

---

## 프론트엔드 통합

### 기본 사용법 (JavaScript)

```javascript
// 1. API 클라이언트 초기화
initializeAPIClient('http://localhost:8000');

// 2. 합성 계획 요청
const result = await apiClient.planRetrosynthesis(
  'CC(=O)Oc1ccccc1C(=O)O',  // SMILES
  {
    planner_model: 'interretro',
    iterations: 500,
    expansion_topk: 50
  }
);

// 3. 결과 처리
if (result.success) {
  console.log('경로 개수:', result.routes.length);
  console.log('비용:', result.route_cost);
  console.log('깊이:', result.route_length);
} else {
  console.error('계획 실패:', result.error);
}
```

### API 클라이언트 (`api-client.js`)

```javascript
// API URL 변경
setAPIUrl('http://new-server:8000');

// 현재 API URL 확인
const url = getAPIUrl();

// 서버 상태 확인
const health = await apiClient.checkHealth();
console.log(health.planner_ready);
```

### 기존 HTML을 새 버전으로 업그레이드

**원본:** `site/retrosynthesis.html` (localStorage 기반)
**새 버전:** `site/retrosynthesis-api.html` (API 기반)

두 파일을 모두 유지하거나, 기존 파일을 대체 가능:

```bash
# 백업 (선택)
cp site/retrosynthesis.html site/retrosynthesis-legacy.html

# 새 버전으로 대체
cp site/retrosynthesis-api.html site/retrosynthesis.html
```

---

## SSH 서버 배포

### 1️⃣ 서버에 파일 업로드

```bash
# 로컬에서 서버로 파일 복사
rsync -avz ./backend/ user@server:/path/to/KBP/backend/

# 또는 scp 사용
scp -r ./backend user@server:/path/to/KBP/
```

### 2️⃣ 서버에서 환경 설정

```bash
# 서버 접속
ssh user@server

# 작업 디렉토리로 이동
cd /path/to/KBP/KBP-main/backend

# Python 버전 확인
python3 --version  # 3.8 이상 필요

# 가상 환경 생성
python3 -m venv venv
source venv/bin/activate

# 의존성 설치
pip install --upgrade pip
pip install -r requirements.txt
```

### 3️⃣ 환경 변수 설정

```bash
# .env 파일 생성
cp .env.example .env
nano .env  # 편집

# 주요 설정:
# ALLOWED_ORIGINS=http://your-frontend-domain:port,http://your-ip:port
# GPU_ID=0  (또는 -1 for CPU)
# PLANNER_MODEL=interretro
```

### 4️⃣ Systemd 서비스 등록 (자동 시작)

**파일 생성:** `/etc/systemd/system/kbp-backend.service`

```ini
[Unit]
Description=KBP Retrosynthesis Backend
After=network.target

[Service]
Type=notify
User=your-username
WorkingDirectory=/path/to/KBP/KBP-main/backend
Environment="PATH=/path/to/KBP/KBP-main/backend/venv/bin"
EnvironmentFile=/path/to/KBP/KBP-main/backend/.env
ExecStart=/path/to/KBP/KBP-main/backend/venv/bin/gunicorn \
    --workers 2 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000 \
    --access-logfile /var/log/kbp/access.log \
    --error-logfile /var/log/kbp/error.log \
    app.main:app

Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

**서비스 활성화:**
```bash
# 로그 디렉토리 생성
sudo mkdir -p /var/log/kbp
sudo chown your-username:your-username /var/log/kbp

# 서비스 로드
sudo systemctl daemon-reload

# 자동 시작 설정
sudo systemctl enable kbp-backend

# 서비스 시작
sudo systemctl start kbp-backend

# 상태 확인
sudo systemctl status kbp-backend

# 로그 보기
sudo journalctl -u kbp-backend -f
```

### 5️⃣ Nginx 역프록시 설정 (선택)

**파일:** `/etc/nginx/sites-available/kbp-backend`

```nginx
server {
    listen 80;
    server_name your-domain.com;

    # API
    location /api {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # 타임아웃 (합성 계획은 오래 걸림)
        proxy_connect_timeout 60s;
        proxy_send_timeout 600s;
        proxy_read_timeout 600s;
    }

    # 헬스 체크
    location /health {
        proxy_pass http://localhost:8000/health;
    }

    # 프론트엔드
    location / {
        root /path/to/KBP/site;
        try_files $uri $uri/ /retrosynthesis-api.html;
    }
}
```

**활성화:**
```bash
sudo ln -s /etc/nginx/sites-available/kbp-backend /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

### 6️⃣ SSL 설정 (Let's Encrypt)

```bash
sudo apt-get install certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

---

## 문제 해결

### ❌ "Cannot import retro_star"

**원인:** Python path에 retro_star가 없음

**해결:**
```python
# retro_wrapper/planner.py 에서
RETRO_STAR_PATH = Path(__file__).parent.parent.parent / 'retro_star'
# 이 경로가 맞는지 확인
```

### ❌ "Planner not initialized"

**원인:** 서버 시작 후 API 호출 직후에 요청

**해결:** 로그를 보고 플래너 초기화 완료를 기다림
```bash
# 로그 확인
tail -f logs/error.log
```

### ❌ CORS 오류

**원인:** 프론트엔드 URL이 ALLOWED_ORIGINS에 없음

**해결:** `.env` 파일 수정
```env
ALLOWED_ORIGINS=http://localhost:5500,http://localhost:3000,http://your-domain.com
```

### ❌ GPU 메모리 부족

**원인:** GPU에 모델이 로드되지 않음

**해결:** CPU 사용으로 변경
```env
GPU_ID=-1
```

### ❌ 요청 타임아웃

**원인:** 합성 계획이 너무 오래 걸림

**해결:**
- Iterations 줄이기 (기본값 500 → 100)
- Expansion_topk 줄이기 (기본값 50 → 25)
- API 타임아웃 증가: `app.main.py`에서 `timeout` 설정

---

## 성능 최적화

### 모델 선택

| 모델 | 속도 | 정확도 | 권장 상황 |
|------|------|--------|----------|
| **MLP** | 빠름 ⚡ | 낮음 | 빠른 스크린 |
| **LocalRetro** | 중간 | 중간 | 균형 |
| **InterRetro** | 느림 | 높음 | 정밀한 결과 |

### 하드웨어 권장사양

| 환경 | CPU | 메모리 | GPU | 시간/쿼리 |
|------|-----|--------|-----|----------|
| 로컬 (CPU) | 8+ cores | 16GB+ | - | 5-15분 |
| 로컬 (GPU) | 8+ cores | 16GB+ | NVIDIA | 1-5분 |
| 서버 (CPU) | 32+ cores | 64GB+ | - | 2-5분 |
| 서버 (GPU) | 16+ cores | 32GB+ | NVIDIA | 30s-2min |

---

## 문의 및 지원

문제가 발생하면:
1. 로그 확인: `tail -f logs/error.log`
2. API 상태 확인: `curl http://localhost:8000/health`
3. 문서 참고: `/docs` endpoint 방문
