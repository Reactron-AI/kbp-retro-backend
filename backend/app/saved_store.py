"""
파일 기반 즐겨찾기(Saved Pathways) 저장소.

feedback_store.py와 똑같은 이유·패턴이다 (단일 프로세스 전제, 프로세스 재시작에도
남아있어야 함, JSON 파일 + Lock으로 직렬화 + 임시 파일 원자적 교체). 별도 파일로
분리한 이유는 즐겨찾기 데이터(경로 전체 스냅샷)가 피드백보다 훨씬 커서 같이 두면
피드백 읽기/쓰기까지 매번 느려지기 때문이다.

브라우저 localStorage(kbp_saved + kbp_pw_<id>)에만 있으면 사이트의 Cloudflare
Tunnel 주소가 바뀔 때(=origin이 바뀌어 localStorage가 안 보이게 됨, PC 재부팅 등)
즐겨찾기가 사라진 것처럼 보이는 문제가 있어서, 조건/반응 피드백과 동일하게
서버에도 보관해 여러 브라우저·재접속에 걸쳐 남아있게 한다.
"""

import json
import os
from pathlib import Path
from threading import Lock
from typing import Any, Dict

DATA_DIR = Path(os.getenv("FEEDBACK_DATA_DIR", Path(__file__).parent.parent / "data"))
DATA_FILE = DATA_DIR / "saved_pathways.json"

_lock = Lock()


def _load() -> Dict[str, Dict[str, Any]]:
    if not DATA_FILE.exists():
        return {}
    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _save(data: Dict[str, Dict[str, Any]]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = DATA_FILE.with_suffix(".json.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, DATA_FILE)


def get_all() -> Dict[str, Dict[str, Any]]:
    with _lock:
        return _load()


def upsert(pw_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
    with _lock:
        store = _load()
        store[pw_id] = data
        _save(store)
        return data


def delete(pw_id: str) -> bool:
    with _lock:
        store = _load()
        if pw_id not in store:
            return False
        del store[pw_id]
        _save(store)
        return True


def clear_all() -> None:
    with _lock:
        _save({})
