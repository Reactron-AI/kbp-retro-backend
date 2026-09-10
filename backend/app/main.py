"""
FastAPI 백엔드 - Retrosynthesis Planning API
"""

import os
import sys
import logging
import torch
from typing import Optional
from pathlib import Path
from contextlib import asynccontextmanager

# 경로 설정 (상대 import 문제 해결)
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException, Request, Depends, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security.api_key import APIKeyHeader
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from .jobs import create_job, get_job, submit_job, update_progress
from . import feedback_store
from . import saved_store
from .models import (
    RetrosynthesisPlanRequest,
    JobStatusResponse,
    JobSubmitResponse,
    StatusResponse,
    BBPriceRequest,
    BBPriceResponse,
    ReactionFeedbackItem,
    RouteFeedbackPatch,
    ReagentFeedbackPatch,
    FeedbackStoreResponse,
    SavedPathwayItem,
    SavedStoreResponse,
)
from retro_wrapper import RetroStarPlanner, init_planner, get_planner, lookup_prices

# ─────────────────────────────────────────────────────────
# 로깅 설정
# ─────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────
# 시작/종료 이벤트
# ─────────────────────────────────────────────────────────

planner_instance: Optional[RetroStarPlanner] = None


async def startup_event():
    """서버 시작 시 planner 초기화 (Lazy init)"""
    global planner_instance
    
    try:
        logger.info("🚀 Initializing RetroStar planner...")
        
        # 환경 변수에서 설정 읽기 (mlp만 체크포인트가 설치되어 있어 기본값으로 사용.
        # interretro/localretro 체크포인트가 준비되면 PLANNER_MODEL과
        # LOCALRETRO_*/INTERRETRO_* 경로를 .env에 채우면 그대로 전환된다.)
        planner_model = os.getenv("PLANNER_MODEL", "mlp")
        gpu_id = int(os.getenv("GPU_ID", "-1"))
        iterations = int(os.getenv("ITERATIONS", "500"))
        expansion_topk = int(os.getenv("EXPANSION_TOPK", "50"))
        output_dir = os.getenv("OUTPUT_DIR", "./output")

        logger.info(f"Config: model={planner_model}, gpu={gpu_id}, iterations={iterations}")

        planner_instance = init_planner(
            planner_model=planner_model,
            gpu=gpu_id,
            iterations=iterations,
            expansion_topk=expansion_topk,
            use_value_fn=True,
            output_dir=output_dir,
            localretro_root=os.getenv("LOCALRETRO_ROOT"),
            localretro_model_path=os.getenv("LOCALRETRO_MODEL_PATH"),
            localretro_config_path=os.getenv("LOCALRETRO_CONFIG_PATH"),
            localretro_data_dir=os.getenv("LOCALRETRO_DATA_DIR"),
            interretro_root=os.getenv("INTERRETRO_ROOT"),
            interretro_checkpoint=os.getenv("INTERRETRO_CHECKPOINT"),
        )
        logger.info("✅ Planner initialized successfully!")

        routes_static_dir = Path(output_dir) / "routes"
        routes_static_dir.mkdir(parents=True, exist_ok=True)
        app.mount(
            "/static/routes",
            StaticFiles(directory=str(routes_static_dir)),
            name="routes",
        )
        
    except Exception as e:
        logger.warning(f"⚠️  Lazy init: Planner initialization deferred. First request will trigger init. Error: {e}")
        # 첫 API 요청 시 다시 초기화됨


async def shutdown_event():
    """서버 종료 시 정리"""
    logger.info("🛑 Shutting down server...")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI 생명주기 관리"""
    # 시작
    await startup_event()
    yield
    # 종료
    await shutdown_event()


# ─────────────────────────────────────────────────────────
# FastAPI 앱 생성
# ─────────────────────────────────────────────────────────

app = FastAPI(
    title="KBP Retrosynthesis API",
    description="Retro* 기반 역합성 계획 서비스",
    version="1.0.0",
    docs_url="/docs",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# ─────────────────────────────────────────────────────────
# CORS 설정 (프론트엔드 접근 허용)
# ─────────────────────────────────────────────────────────

allowed_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:5500").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger.info(f"✅ CORS enabled for: {allowed_origins}")


# ─────────────────────────────────────────────────────────
# API 키 인증 + rate limiting (공개 노출 대비)
# ─────────────────────────────────────────────────────────
# API_KEY가 .env에 설정되어 있지 않으면 인증을 건너뛴다 (로컬 개발 편의용).
# 외부에 공개할 때는 .env에 API_KEY를 반드시 채워야 한다.

API_KEY = os.getenv("API_KEY") or None
if not API_KEY:
    logger.warning("⚠️  API_KEY not set in .env — /api/v1/plan is unauthenticated. Set API_KEY before exposing this server publicly.")

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(key: Optional[str] = Security(_api_key_header)) -> None:
    if API_KEY and key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded. Try again later."})


# ─────────────────────────────────────────────────────────
# 헬스 체크 엔드포인트
# ─────────────────────────────────────────────────────────

@app.get(
    "/health",
    response_model=StatusResponse,
    tags=["System"],
    summary="서버 상태 확인"
)
async def health_check() -> StatusResponse:
    """
    서버 건강 상태 및 플래너 상태 확인
    """
    gpu_available = torch.cuda.is_available()
    planner_ready = planner_instance is not None
    
    return StatusResponse(
        status="ok",
        version="1.0.0",
        planner_ready=planner_ready,
        gpu_available=gpu_available,
    )


# ─────────────────────────────────────────────────────────
# 메인 API 엔드포인트
# ─────────────────────────────────────────────────────────

@app.post(
    "/api/v1/plan",
    response_model=JobSubmitResponse,
    status_code=202,
    tags=["Planning"],
    summary="합성 경로 계획 요청 (비동기, job_id 즉시 반환)",
    dependencies=[Depends(require_api_key)],
)
@limiter.limit("5/minute")
async def plan_retrosynthesis(request: Request, body: RetrosynthesisPlanRequest) -> JobSubmitResponse:
    """
    타겟 분자에 대한 합성 경로 계획을 큐에 등록한다.

    example.py와 동일하게 첫 반응(depth-0)이 서로 다른 unique route를
    최대 max_routes개까지 탐색하며, 검색은 수 분~십수 분 걸릴 수 있어
    바로 결과를 반환하지 않고 job_id를 반환한다. 진행 상황/결과는
    `GET /api/v1/jobs/{job_id}`로 조회한다.

    **요청 예시:**
    ```json
    { "target_mol": "CC(=O)Oc1ccccc1C(=O)O", "max_routes": 20 }
    ```
    **응답:**
    ```json
    { "job_id": "a1b2c3", "status": "queued", "poll_url": "/api/v1/jobs/a1b2c3" }
    ```
    """
    if planner_instance is None:
        raise HTTPException(
            status_code=503,
            detail="Planner not initialized. Server not ready."
        )

    if body.planner_model != planner_instance.planner_model:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Server is configured with planner_model="
                f"'{planner_instance.planner_model}'. Requested "
                f"'{body.planner_model}' is not loaded."
            )
        )

    job_id = create_job(body.target_mol, body.model_dump())
    logger.info(f"📋 Queued job {job_id}: target={body.target_mol[:40]}...")

    submit_job(
        job_id,
        planner_instance.plan,
        job_id=job_id,
        target_mol=body.target_mol,
        max_routes=body.max_routes,
        exclude_smiles=body.exclude_smiles,
        exclude_smiles_strict=body.exclude_smiles_strict,
        include_smiles=body.include_smiles,
        progress_cb=lambda found, total, _jid=job_id: update_progress(_jid, found, total),
    )

    return JobSubmitResponse(
        job_id=job_id,
        status="queued",
        poll_url=f"/api/v1/jobs/{job_id}",
    )


@app.get(
    "/api/v1/jobs/{job_id}",
    response_model=JobStatusResponse,
    tags=["Planning"],
    summary="작업 상태/결과 조회",
)
async def get_job_status(job_id: str) -> JobStatusResponse:
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobStatusResponse(**job)


# ─────────────────────────────────────────────────────────
# 빌딩블록 가격 조회 엔드포인트
# ─────────────────────────────────────────────────────────

@app.post(
    "/api/v1/bb-price",
    response_model=BBPriceResponse,
    tags=["Building Blocks"],
    summary="빌딩블록 SMILES의 $/g 가격 조회 (bb_curation.csv 기반)",
)
@limiter.limit("30/minute")
async def get_bb_prices(request: Request, body: BBPriceRequest) -> BBPriceResponse:
    """
    입력 SMILES마다 bb_curation.csv에서 조회한 $/g 가격을 반환한다.
    목록에 없는 분자(구매처에서 안 파는 빌딩블록)는 null.
    """
    prices = lookup_prices(body.smiles)
    return BBPriceResponse(prices=prices)


# ─────────────────────────────────────────────────────────
# 피드백 엔드포인트 (실현 가능성/선호도, 개별 반응, 시약/촉매/용매 평가)
# ─────────────────────────────────────────────────────────
# 브라우저 localStorage에만 있던 피드백은 캐시를 지우거나 다른 기기로 옮기면
# 사라지는 문제가 있어서, 서버 파일(feedback_store.py, JSON)에도 실제로 저장한다.
# localStorage는 오프라인/즉시 반영용 캐시로 남고, 여기가 여러 사람 것을 합치는
# 실제 저장소다. GET은 전체를 합쳐서 돌려주고, 페이지 로드 시 그걸로
# localStorage를 보강한다(route-feedback.js의 syncFromServer 참고).

@app.get(
    "/api/v1/feedback",
    response_model=FeedbackStoreResponse,
    tags=["Feedback"],
    summary="전체 피드백 조회 (여러 브라우저/사용자 것을 합친 서버 저장본)",
    dependencies=[Depends(require_api_key)],
)
async def get_feedback() -> FeedbackStoreResponse:
    return FeedbackStoreResponse(**feedback_store.get_all())


@app.post(
    "/api/v1/feedback/reaction",
    tags=["Feedback"],
    summary="개별 반응(스텝) 평가 upsert",
    dependencies=[Depends(require_api_key)],
)
@limiter.limit("120/minute")
async def upsert_reaction_feedback(request: Request, body: ReactionFeedbackItem):
    return feedback_store.upsert_reaction(body.model_dump())


@app.delete(
    "/api/v1/feedback/reaction/{item_id}",
    tags=["Feedback"],
    summary="개별 반응 평가 삭제",
    dependencies=[Depends(require_api_key)],
)
async def delete_reaction_feedback(item_id: str):
    if not feedback_store.delete_reaction(item_id):
        raise HTTPException(status_code=404, detail="Feedback entry not found")
    return {"deleted": item_id}


@app.post(
    "/api/v1/feedback/route",
    tags=["Feedback"],
    summary="경로 단위 실현 가능성/선호도 upsert (보낸 필드만 반영)",
    dependencies=[Depends(require_api_key)],
)
@limiter.limit("120/minute")
async def upsert_route_feedback(request: Request, body: RouteFeedbackPatch):
    patch = body.model_dump(exclude={"pwId"}, exclude_unset=True)
    return feedback_store.upsert_route(body.pwId, patch)


@app.post(
    "/api/v1/feedback/reagent",
    tags=["Feedback"],
    summary="시약/촉매/용매 후보 평가 upsert (보낸 필드만 반영)",
    dependencies=[Depends(require_api_key)],
)
@limiter.limit("120/minute")
async def upsert_reagent_feedback(request: Request, body: ReagentFeedbackPatch):
    patch = body.model_dump(exclude={"pwId", "ck"}, exclude_unset=True)
    return feedback_store.upsert_reagent(body.pwId, body.ck, patch)


@app.delete(
    "/api/v1/feedback",
    tags=["Feedback"],
    summary="전체 피드백 삭제 — 이 서버에 쌓인 모든 사람의 피드백이 지워진다 (주의)",
    dependencies=[Depends(require_api_key)],
)
async def clear_all_feedback():
    feedback_store.clear_all()
    return {"cleared": True}


# ─────────────────────────────────────────────────────────
# 즐겨찾기(⭐ Saved Pathways) 엔드포인트
# ─────────────────────────────────────────────────────────
# feedback과 같은 이유로 서버 파일에도 저장한다: 브라우저 localStorage만 쓰면
# 사이트의 Cloudflare Tunnel 주소가 바뀔 때(=origin 변경, 예: PC 재부팅) 이전에
# 저장해둔 즐겨찾기가 안 보이게 된다. GET은 전체를 합쳐서 돌려주고, 페이지 로드
# 시 그걸로 localStorage를 보강한다(route-feedback.js의 syncSavedFromServer 참고).

@app.get(
    "/api/v1/saved",
    response_model=SavedStoreResponse,
    tags=["Saved"],
    summary="전체 즐겨찾기 조회 (여러 브라우저/사용자 것을 합친 서버 저장본)",
    dependencies=[Depends(require_api_key)],
)
async def get_saved() -> SavedStoreResponse:
    return SavedStoreResponse(saved=saved_store.get_all())


@app.post(
    "/api/v1/saved",
    tags=["Saved"],
    summary="즐겨찾기 경로 저장/갱신 (전체 스냅샷 upsert)",
    dependencies=[Depends(require_api_key)],
)
@limiter.limit("60/minute")
async def upsert_saved(request: Request, body: SavedPathwayItem):
    return saved_store.upsert(body.id, body.data)


@app.delete(
    "/api/v1/saved/{pw_id}",
    tags=["Saved"],
    summary="즐겨찾기 경로 삭제",
    dependencies=[Depends(require_api_key)],
)
async def delete_saved(pw_id: str):
    if not saved_store.delete(pw_id):
        raise HTTPException(status_code=404, detail="Saved pathway not found")
    return {"deleted": pw_id}


@app.delete(
    "/api/v1/saved",
    tags=["Saved"],
    summary="전체 즐겨찾기 삭제 — 이 서버에 쌓인 모든 사람의 즐겨찾기가 지워진다 (주의)",
    dependencies=[Depends(require_api_key)],
)
async def clear_all_saved():
    saved_store.clear_all()
    return {"cleared": True}


# ─────────────────────────────────────────────────────────
# 정보 엔드포인트
# ─────────────────────────────────────────────────────────

@app.get(
    "/api/v1/info",
    tags=["System"],
    summary="API 정보"
)
async def api_info():
    """
    API 버전 및 지원 모델 정보
    """
    return {
        "api_version": "1.0.0",
        "service": "KBP Retrosynthesis Planner",
        "loaded_planner": planner_instance.planner_model if planner_instance else None,
        "endpoints": {
            "plan": "POST /api/v1/plan",
            "job_status": "GET /api/v1/jobs/{job_id}",
            "health": "/health",
            "info": "/api/v1/info",
            "docs": "/docs",
        }
    }


# ─────────────────────────────────────────────────────────
# 루트 경로 (리다이렉트)
# ─────────────────────────────────────────────────────────

@app.get("/", tags=["System"], summary="루트 경로")
async def root():
    """API 설명서는 /docs 에서 확인"""
    return {
        "message": "KBP Retrosynthesis Planning API",
        "docs": "/docs",
        "health": "/health",
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
