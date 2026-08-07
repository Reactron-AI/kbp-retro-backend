# KBP Backend — Retrosynthesis Planning API

**FastAPI 기반 역합성 경로 계획 서비스.** `retro_star`의 `RSPlanner*`를 subprocess가
아니라 함수 호출로 직접 사용한다. 다중 unique route 탐색은 수 분~십수 분 걸릴 수 있어
요청-응답이 아니라 **job 등록 + polling** 방식으로 동작한다.

```
POST /api/v1/plan
{ "target_mol": "CC(=O)Oc1ccccc1C(=O)O", "max_routes": 20 }

→ { "job_id": "a1b2c3", "status": "queued", "poll_url": "/api/v1/jobs/a1b2c3" }

GET /api/v1/jobs/a1b2c3
→ { "status": "done", "result": { "routes": [...], ... } }
```

---

## 🚀 빠른 시작

이미 만들어둔 conda 환경(`kbp-backend`)을 그대로 사용한다.

```bash
cd backend
cp .env.example .env   # 필요시 편집 (PLANNER_MODEL, OUTPUT_DIR 등)
bash run_dev.sh         # conda activate kbp-backend 후 uvicorn --reload 실행

# 브라우저에서 확인: http://localhost:8000/docs
```

`kbp-backend` 환경이 아직 없거나 처음부터 새로 만들어야 한다면 `bash run_dev_full.sh`를
쓴다 (conda env 생성 + retro_star/LocalRetro 의존성 설치까지 한 번에 처리).

---

## 📁 구조

```
backend/
├── app/
│   ├── main.py          # FastAPI 앱: /health, /api/v1/plan, /api/v1/jobs/{id}
│   ├── jobs.py           # 인메모리 job store + 단일 워커 스레드 실행기
│   └── models.py        # Pydantic 요청/응답 모델
├── retro_wrapper/
│   ├── __init__.py
│   └── planner.py       # retro_star.case_study.collect_unique_depth0_routes 래퍼
├── requirements.txt     # 의존성 (dgl/dgllife 버전 핀 포함)
├── .env.example         # 환경 변수 템플릿
├── run_dev.sh           # 개발 서버 (conda activate kbp-backend)
├── run_dev_full.sh       # conda env부터 새로 만드는 완전 셋업
├── run_prod.sh           # gunicorn 프로덕션 서버 (workers=1 고정)
└── DEPLOYMENT.md
```

`retro_wrapper/planner.py`는 `retro_star/example.py`와 동일한 탐색 로직
(`retro_star/retro_star/case_study.py`의 `collect_unique_depth0_routes`)을 그대로
가져다 쓴다 — 첫 반응(depth-0)이 서로 다른 unique route를 `max_routes`개까지 찾고,
각 route마다 `route.txt`/`route.png`를 `OUTPUT_DIR/routes/<job_id>/`에 남긴다.

---

## 🔌 API 엔드포인트

### `POST /api/v1/plan` — 계획 요청 등록 (202 Accepted)

**요청:**
```json
{
  "target_mol": "SMILES_STRING",
  "planner_model": "localretro",
  "max_routes": 20
}
```
`planner_model`은 서버가 시작 시 로드한 모델과 같아야 한다 (다르면 400). 서버가
실제로 어떤 모델을 로드했는지는 `GET /api/v1/info`의 `loaded_planner`로 확인한다.

**응답:**
```json
{ "job_id": "a1b2c3", "status": "queued", "poll_url": "/api/v1/jobs/a1b2c3" }
```

### `GET /api/v1/jobs/{job_id}` — 상태/결과 조회

```json
{
  "job_id": "a1b2c3",
  "status": "done",
  "target_mol": "...",
  "created_at": 1234567890.1,
  "started_at": 1234567891.2,
  "finished_at": 1234567905.6,
  "result": {
    "routes": [
      {
        "route_index": 0,
        "route_cost": 7.6,
        "route_len": 5,
        "steps": [{"depth": 0, "product": "...", "reactants": ["..."]}],
        "route_viz": "/static/routes/a1b2c3/viz/mol_0_route.png",
        "route_file": "/static/routes/a1b2c3/routes/mol_0_route.txt"
      }
    ],
    "search_count": 2,
    "stop_reason": "max_routes"
  },
  "error": null
}
```
`status`는 `queued` → `running` → `done`(성공) / `error`(예외) 순서로 바뀐다.
`route_viz`/`route_file`은 `app.mount("/static/routes", ...)`로 그대로 서빙되므로
API base URL만 붙이면 `<img src>`로 바로 쓸 수 있다.

### `GET /health`, `GET /api/v1/info`
서버/플래너 상태, 로드된 모델 이름 확인.

---

## 📚 JavaScript 클라이언트 (`site/api-client.js`)

```javascript
initializeAPIClient('http://localhost:8000');

const job = await apiClient.planRetrosynthesis(
  'CC(=O)Oc1ccccc1C(=O)O',
  { planner_model: 'localretro', max_routes: 20 },
  (update) => console.log('progress:', update.status)  // polling마다 호출
);

if (job.status === 'done') {
  console.log('routes:', job.result.routes);
}
```

---

## 📊 환경 변수 (.env) 핵심 항목

```env
PLANNER_MODEL=localretro   # mlp | localretro | interretro
GPU_ID=-1
ITERATIONS=500
EXPANSION_TOPK=50
OUTPUT_DIR=./output         # route.txt/route.png 저장 위치, /static/routes로 서빙됨
ALLOWED_ORIGINS=http://localhost:5500,http://localhost:3000

# localretro 체크포인트가 기본 경로(KBP-main/LocalRetro/...)와 다르면 여기서 지정
LOCALRETRO_CONFIG_PATH=../LocalRetro/data/config/default_config.json
```

각 모델이 실제로 무엇을 요구하는지:

| 모델 | 체크포인트 | 추가 의존성 |
|---|---|---|
| `mlp` | 레포에 포함됨, 바로 동작 | 없음 |
| `localretro` | `LocalRetro/models/*.pth` + `data/USPTO_50K/` + config json | `dgl==0.9.1`, `dgllife==0.2.8` (torch==1.9.0 전용 핀, requirements.txt에 포함) |
| `interretro` | LocalRetro 체크포인트 + `InterRetro/.../*.pth` | `torch==2.2.1` 요구 — 현재 환경(torch==1.9.0)과 충돌해서 보류 상태 |

---

## 🔧 개발 vs 프로덕션

| 환경 | 명령어 | 비고 |
|------|--------|------|
| **개발** | `bash run_dev.sh` | `--reload`, conda activate만 함 |
| **프로덕션** | `bash run_prod.sh` | gunicorn, **`--workers 1` 고정** |

`--workers 1`이 필수인 이유: planner 모델(수 GB)과 job 상태가 프로세스 메모리에만
있다. 워커가 여러 개면 각자 모델을 따로 로드(메모리·시간 낭비)하고, job_id도 등록된
워커에서만 조회 가능해져서 로드밸런서가 다른 워커로 보내면 404가 난다.

---

## ⚙️ graphviz (`dot`) 바이너리

route.png 생성에는 파이썬 `graphviz` 패키지뿐 아니라 시스템 `dot` 실행 파일이
필요하다. `dot`이 없으면 `route_viz`가 조용히 `null`로 빠진다(에러 없음). sudo
권한이 없는 서버에서는 conda로 설치하면 된다 (가상환경 안에만 들어가서 PATH도
`conda activate`로 자동으로 잡힌다):

```bash
conda install -n kbp-backend -c conda-forge graphviz
```

`run_dev.sh`/`run_prod.sh`는 `conda activate`를 직접 쓰기 때문에(​`conda run`이
아님) PATH가 정상적으로 잡힌다. `conda run -n kbp-backend uvicorn ...`처럼 직접
실행하면 stdout이 버퍼링되어 로그가 안 보이고 PATH도 활성화되지 않아 `dot`을 못
찾는 문제가 있으니 피한다.

---

## 📝 예제

### cURL
```bash
JOB=$(curl -s -X POST http://localhost:8000/api/v1/plan \
  -H "Content-Type: application/json" \
  -d '{"target_mol": "CC(=O)Oc1ccccc1C(=O)O", "max_routes": 5}')
JOB_ID=$(echo "$JOB" | python3 -c "import sys,json;print(json.load(sys.stdin)['job_id'])")

# 완료될 때까지 polling
watch -n 3 curl -s "http://localhost:8000/api/v1/jobs/$JOB_ID"
```

### Python
```python
import requests, time

job = requests.post(
    'http://localhost:8000/api/v1/plan',
    json={'target_mol': 'CC(=O)Oc1ccccc1C(=O)O', 'max_routes': 5},
).json()

while True:
    status = requests.get(f"http://localhost:8000/api/v1/jobs/{job['job_id']}").json()
    if status['status'] in ('done', 'error'):
        break
    time.sleep(4)

print(status['result']['routes'] if status['status'] == 'done' else status['error'])
```

---

## 🐛 문제 해결

| 문제 | 원인 | 해결 |
|------|------|------|
| "Planner not initialized" | 초기화 중/실패 | server.log에서 startup 에러 확인 |
| `route_viz`가 항상 null | `dot` 바이너리 없음 | 위 graphviz 섹션 참고 |
| `planner_model` 요청에 400 | 서버가 다른 모델을 로드함 | `/api/v1/info`의 `loaded_planner` 확인 후 맞춰서 요청 |
| CORS 오류 | URL 미등록 | `.env`의 `ALLOWED_ORIGINS` 확인 |
| job이 영원히 `running` | 탐색 자체가 오래 걸림(분 단위 정상) | `max_routes`/`ITERATIONS` 줄이기 |
| localretro 켰는데 dgl import 에러 | dgl 버전이 torch와 안 맞음 | `requirements.txt`의 `dgl==0.9.1`/`dgllife==0.2.8` 핀 확인 (최신 dgl 2.x는 torch==1.9.0과 호환 안 됨) |

더 자세한 정보는 [DEPLOYMENT.md](DEPLOYMENT.md)를 참고하세요.
