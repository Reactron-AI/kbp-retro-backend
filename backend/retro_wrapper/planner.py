"""
Retro* Planning Wrapper for Web API
- 초기화 시 planner 인스턴스 생성 (expensive operation)
- plan() 메서드로 타겟 분자 처리 (example.py와 동일한 다중 unique-route 탐색)
"""

import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# Add retro_star to path
RETRO_STAR_PATH = Path(__file__).parent.parent.parent / 'retro_star'
sys.path.insert(0, str(RETRO_STAR_PATH))

try:
    from retro_star.case_study import collect_unique_depth0_routes
except ImportError as e:
    raise ImportError(f"Cannot import retro_star. Check RETRO_STAR_PATH: {RETRO_STAR_PATH}") from e


def _route_mols(route: Dict[str, Any]) -> set:
    """route['steps']에 등장하는 모든 product/reactant SMILES 집합."""
    mols = set()
    for step in route.get('steps', []):
        mols.add(step.get('product'))
        mols.update(step.get('reactants', []))
    mols.discard(None)
    return mols


def _annotate_include_matches(result: Dict[str, Any], include_smiles: Optional[List[str]]) -> None:
    """include_smiles가 주어지면 각 route에 'contains_included'를 달고, 그
    플래그가 True인 route를 앞쪽으로 정렬한다 (탐색 자체를 그 분자 경유로
    강제하지는 못하므로, 이미 나온 결과에 대한 후처리 우선순위/태깅일 뿐).
    한 번도 등장하지 않은 include 분자는 result['include_smiles_not_found']에
    담아 프론트가 "이번 결과에 없음"을 안내할 수 있게 한다."""
    include_set = set(include_smiles or [])
    if not include_set:
        return

    found = set()
    for route in result.get('routes', []):
        mols = _route_mols(route)
        matched = mols & include_set
        route['contains_included'] = bool(matched)
        found |= matched

    result['routes'].sort(key=lambda r: not r.get('contains_included', False))
    result['include_smiles_not_found'] = sorted(include_set - found)


class RetroStarPlanner:
    """
    Retro* 계획자 래퍼
    - 초기화 시 무거운 모델 로드 (한 번만)
    - plan() 호출로 각 쿼리 처리. example.py의 _collect_unique_depth0_routes와
      동일한 retro_star.case_study.collect_unique_depth0_routes를 사용해
      여러 개의 서로 다른(첫 반응 기준) 합성 경로 + route.txt/route.png를 생성한다.
    """

    def __init__(
        self,
        planner_model: str = 'mlp',
        gpu: int = -1,
        expansion_topk: int = 50,
        iterations: int = 500,
        use_value_fn: bool = True,
        localretro_root: Optional[str] = None,
        localretro_model_path: Optional[str] = None,
        localretro_config_path: Optional[str] = None,
        localretro_data_dir: Optional[str] = None,
        interretro_root: Optional[str] = None,
        interretro_checkpoint: Optional[str] = None,
        output_dir: Optional[str] = None,
        include_bb: Optional[List[str]] = None,
        exclude_bb: Optional[List[str]] = None,
    ):
        """
        Args:
            planner_model: 'mlp', 'localretro', or 'interretro'
            gpu: GPU ID (-1 for CPU)
            expansion_topk: 상위 K개의 반응 선택
            iterations: 계획 최대 반복 횟수
            use_value_fn: 가치 함수 사용 여부
            localretro_root: LocalRetro 루트 경로
            localretro_model_path: LocalRetro 모델 경로
            localretro_config_path: LocalRetro 설정 경로
            localretro_data_dir: LocalRetro 데이터 디렉토리
            interretro_root: InterRetro 루트 경로
            interretro_checkpoint: InterRetro 체크포인트 경로
            output_dir: 결과(route.txt/route.png) 저장 디렉토리 (기본: ./output)
            include_bb: 포함할 빌딩 블록 목록
            exclude_bb: 제외할 빌딩 블록 목록
        """
        self.planner_model = planner_model
        self.gpu = gpu
        self.expansion_topk = expansion_topk
        self.iterations = iterations
        self.use_value_fn = use_value_fn
        self.include_bb = include_bb or []
        self.exclude_bb = exclude_bb or []
        self.output_dir = Path(output_dir or './output')

        # 작업(job)별 route.txt/route.png 저장용 디렉토리: output_dir/routes/<job_id>
        self.route_save_dir = self.output_dir / 'routes'
        self.route_save_dir.mkdir(parents=True, exist_ok=True)

        print(f"[Planner] Initializing RSPlanner with {planner_model}...")
        print(f"[Planner] GPU: {gpu}, iterations: {iterations}, topk: {expansion_topk}")

        # retro_star.common이 import 시점에 sys.argv를 파싱하므로 비워둔다.
        sys.argv = [sys.argv[0]]
        from retro_star.api import RSPlanner, RSPlannerInterRetro, RSPlannerLocalRetro

        # retro_star.common.parse_args가 위 import의 부작용으로 argparse 기본값
        # (--gpu 기본값 -1)을 기준으로 os.environ['CUDA_VISIBLE_DEVICES']를 이미
        # 덮어썼다 (sys.argv를 비웠으니 항상 기본값 -1). 이 시점까지 CUDA는 아직
        # 한 번도 실제로 초기화되지 않았으므로(지연 초기화) 여기서 우리가 받은
        # gpu 값으로 다시 설정하면 이후의 실제 모델 로딩(.to(device))에 반영된다.
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu) if gpu is not None and gpu >= 0 else ''

        # example.py의 _build_planner와 동일한 옵션: 첫 반응(depth-0)을 매 호출마다
        # 다르게 탐색해서 collect_unique_depth0_routes가 중복 없는 경로를 모을 수 있게 한다.
        common_kwargs = {
            'gpu': gpu,
            'expansion_topk': expansion_topk,
            'root_expansion_topk': 'rank_after_banned',
            'root_keep_first_reaction': True,
            'iterations': iterations,
            'include_bb': self.include_bb,
            'exclude_bb': self.exclude_bb,
            'viz': False,
        }

        if planner_model == 'localretro':
            self.planner = RSPlannerLocalRetro(
                **common_kwargs,
                use_value_fn=False,
                localretro_root=localretro_root,
                localretro_model_path=localretro_model_path,
                localretro_config_path=localretro_config_path,
                localretro_data_dir=localretro_data_dir,
            )
        elif planner_model == 'interretro':
            self.planner = RSPlannerInterRetro(
                **common_kwargs,
                use_value_fn=use_value_fn,
                interretro_root=interretro_root,
                interretro_checkpoint=interretro_checkpoint,
                localretro_root=localretro_root,
                localretro_model_path=localretro_model_path,
                localretro_config_path=localretro_config_path,
                localretro_data_dir=localretro_data_dir,
            )
        else:
            self.planner = RSPlanner(**common_kwargs, use_value_fn=use_value_fn)

        print("[Planner] RSPlanner initialized successfully!")

    def plan(
        self,
        job_id: str,
        target_mol: str,
        max_routes: int = 20,
        exclude_smiles: Optional[List[str]] = None,
        exclude_smiles_strict: Optional[List[str]] = None,
        include_smiles: Optional[List[str]] = None,
        progress_cb: Optional[Callable[[int, int], None]] = None,
    ) -> Dict[str, Any]:
        """
        타겟 분자에 대해 example.py와 동일한 다중 unique-route 탐색을 실행한다.

        Args:
            job_id: 결과(route.txt/route.png)를 저장할 하위 디렉토리 이름
            target_mol: SMILES 형식의 타겟 분자
            max_routes: 탐색을 멈추기까지 모을 최대 unique route 수
            exclude_smiles: 이번 탐색에서만 "이미 확보된 물질" 취급을 하지 않을
                SMILES 목록 — 검색이 이 분자들을 더 분해하거나(가능하면) 아예
                실패하도록 만든다 (banned_reactions와 동일하게 매 호출마다
                새로 전달되는 파라미터). 단, 이 분자가 재고에 없는 진짜
                중간체라면 원래도 "이미 확보된 물질"이 아니었으므로 아무
                효과가 없다 — building block 전용 exclude.
            exclude_smiles_strict: exclude_smiles와 달리, 이 분자들을 재고
                취급하지 않는 것은 물론 route의 어느 위치(중간체 포함)에도
                등장하지 못하도록 그 분자를 만들어내는 반응 후보 자체를
                차단한다. building block이든 중간체든 상관없이 "이 물질은
                절대 이번 route에 쓰지 않는다"는 제약.
            include_smiles: 이번 탐색 결과 중 이 분자들을 포함하는 route를
                앞쪽으로 정렬해주는, 탐색 이후 후처리 힌트. 실제 탐색이 이
                분자를 반드시 지나가도록 강제하지는 않는다 (single-target
                best-first MCTS에는 그런 lookahead 메커니즘이 없음).
            progress_cb: collect_unique_depth0_routes()가 매 depth-0 탐색
                시도 시작 시 (지금까지 확보한 unique route 수, max_routes)로
                호출하는 콜백. job 상태에 진행률을 반영하는 용도 — 실제 탐색
                총량(중복/실패로 버려지는 시도 수)은 미리 알 수 없으므로
                정확한 %가 아니라 "확보한 route 수 / 목표 route 수" 기준의
                근사치다.

        Returns:
            collect_unique_depth0_routes()의 결과 dict + 'success'/'time' 필드.
            각 route의 'route_viz'/'route_file'은 이 job의 case_study_dir 기준
            절대 경로이며, /static/routes/<job_id>/... 로 그대로 서빙된다.
        """
        start_time = time.time()
        case_study_dir = self.route_save_dir / job_id

        try:
            print(f"[Planner] Planning for: {target_mol[:50]}... (job={job_id})")

            result = collect_unique_depth0_routes(
                self.planner,
                target_mol,
                str(case_study_dir),
                planner_name=self.planner_model,
                max_routes=max_routes,
                exclude_smiles=exclude_smiles,
                exclude_smiles_strict=exclude_smiles_strict,
                progress_callback=progress_cb,
            )
            result['success'] = len(result['routes']) > 0
            result['time'] = time.time() - start_time
            self._rewrite_artifact_urls(result)
            _annotate_include_matches(result, include_smiles)
            return result

        except Exception as e:
            print(f"[Planner] Error during planning: {str(e)}")
            return {
                'success': False,
                'target_mol': target_mol,
                'time': time.time() - start_time,
                'routes': [],
                'error': str(e),
            }

    def plan_batch(
        self,
        target_mols: List[str],
        max_routes: int = 20,
    ) -> List[Dict[str, Any]]:
        """여러 분자 배치 처리 (각각 새 job_id 하위 디렉토리에 결과 저장)"""
        return [
            self.plan(job_id=f'batch_{idx}', target_mol=mol, max_routes=max_routes)
            for idx, mol in enumerate(target_mols)
        ]

    def _rewrite_artifact_urls(self, result: Dict[str, Any]) -> None:
        """route_viz/route_file 파일시스템 경로를 main.py가 마운트한
        /static/routes/<job_id>/... URL로 바꿔서, 프론트엔드가 API base URL과
        이어붙이기만 하면 바로 fetch/<img src>로 쓸 수 있게 한다."""
        routes_root = self.route_save_dir.resolve()
        for route in result.get('routes', []):
            for key in ('route_viz', 'route_file'):
                path = route.get(key)
                if not path:
                    continue
                try:
                    rel = Path(path).resolve().relative_to(routes_root)
                except ValueError:
                    continue
                route[key] = f'/static/routes/{rel.as_posix()}'


# 전역 planner 인스턴스 (FastAPI 시작시 초기화)
_planner_instance: Optional[RetroStarPlanner] = None


def init_planner(**kwargs) -> RetroStarPlanner:
    """전역 planner 인스턴스 초기화"""
    global _planner_instance
    _planner_instance = RetroStarPlanner(**kwargs)
    return _planner_instance


def get_planner() -> RetroStarPlanner:
    """전역 planner 인스턴스 반환"""
    if _planner_instance is None:
        raise RuntimeError("Planner not initialized. Call init_planner() first.")
    return _planner_instance
