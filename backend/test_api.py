#!/usr/bin/env python3
"""
KBP Retrosynthesis API — 테스트 및 예제 스크립트
사용법: python3 test_api.py [SMILES] [API_URL]
"""

import sys
import time
import json
import requests
from typing import Optional


class APITester:
    """API 테스트 클래스"""

    def __init__(self, api_url: str = 'http://localhost:8000'):
        self.api_url = api_url.rstrip('/')
        self.session = requests.Session()

    def test_health(self) -> bool:
        """서버 상태 확인"""
        print("🔍 Testing health endpoint...")
        try:
            response = self.session.get(f'{self.api_url}/health', timeout=5)
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Health check passed")
                print(f"   Status: {data.get('status')}")
                print(f"   Version: {data.get('version')}")
                print(f"   Planner ready: {data.get('planner_ready')}")
                print(f"   GPU available: {data.get('gpu_available')}")
                return True
            else:
                print(f"❌ Health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Cannot connect to API: {e}")
            return False

    def test_info(self) -> Optional[str]:
        """API 정보 조회. 서버에 로드된 planner 이름을 반환한다."""
        print("\n🔍 Testing info endpoint...")
        try:
            response = self.session.get(f'{self.api_url}/api/v1/info', timeout=5)
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Info retrieved")
                print(f"   API Version: {data.get('api_version')}")
                print(f"   Service: {data.get('service')}")
                print(f"   Loaded planner: {data.get('loaded_planner')}")
                return data.get('loaded_planner')
            else:
                print(f"❌ Failed: {response.status_code}")
                return None
        except Exception as e:
            print(f"❌ Error: {e}")
            return None

    def plan_retrosynthesis(
        self,
        target_mol: str,
        planner_model: str = 'mlp',
        max_routes: int = 5,
        timeout: float = 600,
        poll_interval: float = 4.0,
    ) -> Optional[dict]:
        """합성 계획 요청을 등록하고 완료될 때까지 polling한다."""
        print(f"\n📋 Planning for: {target_mol[:50]}...")
        print(f"   Model: {planner_model}")
        print(f"   Max routes: {max_routes}")
        print(f"   Timeout: {timeout}s")

        payload = {
            'target_mol': target_mol,
            'planner_model': planner_model,
            'max_routes': max_routes,
        }

        try:
            response = self.session.post(
                f'{self.api_url}/api/v1/plan',
                json=payload,
                timeout=10,
                headers={'Content-Type': 'application/json'}
            )
        except Exception as e:
            print(f"❌ Error submitting job: {e}")
            return None

        if response.status_code != 202:
            print(f"❌ API error: {response.status_code}")
            print(f"   Response: {response.text}")
            return None

        job_id = response.json()['job_id']
        print(f"✅ Job queued: {job_id}")

        start_time = time.time()
        while time.time() - start_time < timeout:
            status_resp = self.session.get(f'{self.api_url}/api/v1/jobs/{job_id}', timeout=10)
            if status_resp.status_code != 200:
                print(f"❌ Failed to poll job: {status_resp.status_code}")
                return None

            job = status_resp.json()
            elapsed = time.time() - start_time
            print(f"   [{elapsed:.0f}s] status={job['status']}")

            if job['status'] == 'done':
                result = job['result']
                print(f"\n✅ Planning completed in {elapsed:.1f}s")
                print(f"   Routes found: {len(result.get('routes', []))}")
                print(f"   Search count: {result.get('search_count')}")
                print(f"   Stop reason: {result.get('stop_reason')}")

                for route in result.get('routes', [])[:3]:
                    print(f"\n   🔗 Route {route['route_index']} "
                          f"(cost={route['route_cost']:.2f}, len={route['route_len']}):")
                    for step in route.get('steps', [])[:3]:
                        product = step.get('product', '')[:40]
                        reactants = ' + '.join(r[:30] for r in step.get('reactants', []))
                        print(f"      depth {step['depth']}: {product} <- {reactants}")

                return job
            elif job['status'] == 'error':
                print(f"❌ Planning failed: {job.get('error')}")
                return None

            time.sleep(poll_interval)

        print(f"❌ Timed out after {timeout}s waiting for job {job_id}")
        return None

    def print_full_result(self, job: dict):
        """전체 결과 출력"""
        print("\n" + "=" * 60)
        print("FULL RESULT")
        print("=" * 60)
        print(json.dumps(job, indent=2, default=str))


def main():
    """메인 함수"""
    import argparse

    parser = argparse.ArgumentParser(
        description='KBP Retrosynthesis API 테스트',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
예제:
  # 기본 테스트 (서버에 로드된 모델 자동 사용)
  python3 test_api.py

  # 특정 분자로 테스트
  python3 test_api.py "CC(=O)Oc1ccccc1C(=O)O"

  # 원격 서버에서 테스트
  python3 test_api.py "CN1C=C(...)" "http://your-server:8000"

  # 적은 route 수로 빠르게
  python3 test_api.py "CC(=O)Oc1ccccc1C(=O)O" --max-routes 2
        '''
    )

    parser.add_argument(
        'smiles',
        nargs='?',
        default='CC(=O)Oc1ccccc1C(=O)O',
        help='타겟 분자 SMILES (기본: aspirin)'
    )

    parser.add_argument(
        'api_url',
        nargs='?',
        default='http://localhost:8000',
        help='API 서버 URL (기본: http://localhost:8000)'
    )

    parser.add_argument(
        '--model', '-m',
        choices=['mlp', 'localretro', 'interretro'],
        default=None,
        help='계획 모델 (기본: 서버에 로드된 모델, /api/v1/info로 확인)'
    )

    parser.add_argument(
        '--max-routes', '-r',
        type=int,
        default=5,
        help='탐색할 unique route 수 (기본: 5)'
    )

    parser.add_argument(
        '--timeout', '-t',
        type=float,
        default=600,
        help='타임아웃 (초, 기본: 600)'
    )

    parser.add_argument(
        '--full', '-f',
        action='store_true',
        help='전체 JSON 결과 출력'
    )

    args = parser.parse_args()

    print("🧪 KBP Retrosynthesis API Tester")
    print("=" * 60)
    print(f"API URL: {args.api_url}")
    print()

    tester = APITester(args.api_url)

    if not tester.test_health():
        print("\n❌ API 서버에 연결할 수 없습니다.")
        print(f"   서버가 {args.api_url}에서 실행 중인지 확인하세요.")
        sys.exit(1)

    loaded_planner = tester.test_info()
    planner_model = args.model or loaded_planner or 'mlp'

    job = tester.plan_retrosynthesis(
        target_mol=args.smiles,
        planner_model=planner_model,
        max_routes=args.max_routes,
        timeout=args.timeout,
    )

    if job:
        if args.full:
            tester.print_full_result(job)
        print("\n✅ 테스트 성공!")
        return 0
    else:
        print("\n❌ 테스트 실패!")
        return 1


if __name__ == '__main__':
    sys.exit(main())
