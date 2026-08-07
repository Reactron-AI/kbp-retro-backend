"""FastAPI 요청/응답 데이터 모델"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class RetrosynthesisPlanRequest(BaseModel):
    """합성 계획 요청 — 큐에 등록되고 job_id를 즉시 반환한다"""
    target_mol: str = Field(
        ...,
        description="타겟 분자 (SMILES 형식)",
        example="CC(=O)Oc1ccccc1C(=O)O"
    )
    planner_model: str = Field(
        default="mlp",
        description="계획자 모델 (interretro/localretro는 체크포인트가 설치된 경우만 사용 가능)",
        pattern="^(mlp|localretro|interretro)$"
    )
    max_routes: int = Field(
        default=20,
        ge=1,
        le=100,
        description="탐색할 unique route 최대 개수 (첫 반응 기준 서로 다른 경로)"
    )
    exclude_smiles: List[str] = Field(
        default_factory=list,
        description="이번 탐색에서 '이미 확보된 물질'로 취급하지 않을 SMILES 목록"
    )
    include_smiles: List[str] = Field(
        default_factory=list,
        description="결과 route 중 이 물질들을 포함하는 route를 우선 정렬하기 위한 SMILES 목록 (탐색을 강제하지는 않음)"
    )


class JobSubmitResponse(BaseModel):
    """계획 요청 접수 응답"""
    job_id: str
    status: str = Field(default="queued", description="queued|running|done|error")
    poll_url: str = Field(..., description="상태 조회용 URL")


class JobStatusResponse(BaseModel):
    """작업 상태/결과 조회 응답"""
    job_id: str
    status: str = Field(..., description="queued|running|done|error")
    target_mol: str
    created_at: float
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    result: Optional[Dict[str, Any]] = Field(
        default=None,
        description="완료 시 collect_unique_depth0_routes() 결과 (routes, route_signatures 등)"
    )
    error: Optional[str] = None


class StatusResponse(BaseModel):
    """서버 상태 응답"""
    status: str = Field(default="ok", description="서버 상태")
    version: str = Field(default="1.0.0", description="API 버전")
    planner_ready: bool = Field(description="계획자 초기화 여부")
    gpu_available: bool = Field(description="GPU 사용 가능 여부")
