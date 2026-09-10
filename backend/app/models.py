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
        description="이번 탐색에서 '이미 확보된 물질'(building block)로 취급하지 않을 SMILES 목록. "
                    "재고에 없는 중간체에는 효과가 없다."
    )
    exclude_smiles_strict: List[str] = Field(
        default_factory=list,
        description="building block 여부와 무관하게, route의 어느 위치(중간체 포함)에도 "
                    "이 SMILES들이 등장하지 못하도록 완전히 배제한다"
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
    progress: Optional[Dict[str, int]] = Field(
        default=None,
        description="탐색 중 근사 진행률: {routes_found, max_routes}. 아직 없으면 null."
    )


class BBPriceRequest(BaseModel):
    """빌딩블록 가격($/g) 조회 요청"""
    smiles: List[str] = Field(
        ...,
        description="가격을 조회할 SMILES 목록",
        max_length=500,
    )


class BBPriceResponse(BaseModel):
    """빌딩블록 가격($/g) 조회 응답"""
    prices: Dict[str, Optional[float]] = Field(
        ...,
        description="입력 SMILES -> $/g (bb_curation.csv에 없으면 null)"
    )


class StatusResponse(BaseModel):
    """서버 상태 응답"""
    status: str = Field(default="ok", description="서버 상태")
    version: str = Field(default="1.0.0", description="API 버전")
    planner_ready: bool = Field(description="계획자 초기화 여부")
    gpu_available: bool = Field(description="GPU 사용 가능 여부")


class ReactionFeedbackItem(BaseModel):
    """개별 반응(스텝) 평가 — site/results.html·saved.html의 kbp_feedback 항목과 동일 구조"""
    id: str = Field(..., description="`${stepKey}@${pwId}` 형태의 고유 id")
    target: Optional[str] = None
    pwId: str
    pwLabel: Optional[str] = None
    typeKey: Optional[str] = None
    stepKey: str
    precursors: List[str] = Field(default_factory=list)
    product: Optional[str] = None
    flag: Optional[str] = Field(default=None, pattern="^(plausible|implausible|unsure)$")
    comment: str = ""
    reasons: List[str] = Field(default_factory=list, description="FEASIBILITY_REASON_OPTIONS codes, meaningful when flag='implausible'")
    reasonCustom: str = ""
    ts: Optional[float] = None


class RouteFeedbackPatch(BaseModel):
    """경로 단위 실현 가능성/선호도 — 보낸 필드만 merge된다"""
    pwId: str
    feasibility: Optional[str] = Field(default=None, pattern="^(yes|no|unsure)$")
    feasibilityReason: Optional[str] = None
    preference: Optional[str] = Field(default=None, pattern="^(yes|no|unsure)$")
    preferenceReason: Optional[str] = None


class ReagentFeedbackPatch(BaseModel):
    """시약/촉매/용매 후보 하나에 대한 평가 — 보낸 필드만 merge된다"""
    pwId: str
    ck: str = Field(..., description="`${stepKey}::${candidateIdx}` 형태의 키")
    flag: Optional[str] = Field(default=None, pattern="^(plausible|implausible)$")
    comment: Optional[str] = None


class FeedbackStoreResponse(BaseModel):
    """서버에 쌓인 전체 피드백 (여러 브라우저/사용자의 것을 합친 것)"""
    reactionFeedback: List[Dict[str, Any]] = Field(default_factory=list)
    routeFeedback: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    reagentFeedback: Dict[str, Dict[str, Any]] = Field(default_factory=dict)


class SavedPathwayItem(BaseModel):
    """즐겨찾기(⭐ Saved Pathways) 항목 하나 — site의 kbp_pw_<id> localStorage 항목과 동일 구조"""
    id: str = Field(..., description="pathway id (예: '7-5_apixaban_route_0')")
    data: Dict[str, Any] = Field(
        ..., description="pathway 전체 스냅샷 (nodes/edges/stepConditions/typeKey/score 등)"
    )


class SavedStoreResponse(BaseModel):
    """서버에 쌓인 전체 즐겨찾기 (여러 브라우저/사용자 것을 합친 것) — pwId -> 스냅샷"""
    saved: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
