"""
In-memory job store + single-worker executor.

Route planning can take minutes, so requests are handled as background jobs
instead of blocking the HTTP request. A single worker thread serializes
access to the shared RSPlanner instance (its expand/value functions are not
guaranteed thread-safe under concurrent searches).

In-memory only: jobs are lost on process restart, and a job created on one
worker process is invisible to another. Run the API with a single process
(see backend/run_prod.sh) — this is fine for a single-researcher / low-traffic
deployment, not for horizontal scaling.
"""

import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

_executor = ThreadPoolExecutor(max_workers=1)
_jobs: Dict[str, Dict[str, Any]] = {}


def create_job(target_mol: str, params: Dict[str, Any]) -> str:
    job_id = uuid.uuid4().hex[:12]
    _jobs[job_id] = {
        'job_id': job_id,
        'status': 'queued',
        'target_mol': target_mol,
        'params': params,
        'created_at': time.time(),
        'started_at': None,
        'finished_at': None,
        'result': None,
        'error': None,
        'progress': None,
    }
    return job_id


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    return _jobs.get(job_id)


def update_progress(job_id: str, routes_found: int, max_routes: int) -> None:
    """collect_unique_depth0_routes()의 progress_callback에서 호출된다.
    확보한 unique route 수 / 목표 route 수 기준 근사 진행률 — job이 이미
    끝났거나(다른 스레드가 finished_at을 찍은 뒤) 사라졌을 수 있으므로
    조용히 무시한다."""
    job = _jobs.get(job_id)
    if job is None:
        return
    job['progress'] = {'routes_found': routes_found, 'max_routes': max_routes}


def submit_job(job_id: str, run_fn: Callable[..., Any], /, *args, **kwargs) -> None:
    """Run run_fn(*args, **kwargs) on the single-worker executor, updating job state."""

    def _run():
        job = _jobs[job_id]
        job['status'] = 'running'
        job['started_at'] = time.time()
        try:
            job['result'] = run_fn(*args, **kwargs)
            job['status'] = 'done'
        except Exception as exc:
            logger.exception('Job %s failed', job_id)
            job['error'] = str(exc)
            job['status'] = 'error'
        finally:
            job['finished_at'] = time.time()

    _executor.submit(_run)
