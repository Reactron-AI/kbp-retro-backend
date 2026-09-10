"""
파일 기반 피드백 저장소 (실현 가능성/선호도, 개별 반응 평가, 시약/촉매/용매 평가).

jobs.py와 같은 "단일 프로세스, 저트래픽" 전제를 따르지만, job과 달리 피드백은
프로세스 재시작에도 남아있어야 하므로(브라우저 localStorage만으로는 캐시를 지우면
사라짐) JSON 파일에 저장한다. 여러 브라우저/사용자가 동시에 쓸 수 있으므로
읽기-수정-쓰기 구간을 asyncio.Lock으로 직렬화하고, 쓰기는 임시 파일 + os.replace로
원자적으로 반영한다(쓰는 도중 프로세스가 죽어도 파일이 깨지지 않도록).
"""

import json
import logging
import os
import time
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DATA_DIR = Path(os.getenv("FEEDBACK_DATA_DIR", Path(__file__).parent.parent / "data"))
DATA_FILE = DATA_DIR / "feedback.json"

_lock = Lock()


def _empty_store() -> Dict[str, Any]:
    return {"reactionFeedback": [], "routeFeedback": {}, "reagentFeedback": {}}


def _load() -> Dict[str, Any]:
    if not DATA_FILE.exists():
        return _empty_store()
    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Backfill in case the file predates one of these keys.
        for key in ("reactionFeedback", "routeFeedback", "reagentFeedback"):
            data.setdefault(key, [] if key == "reactionFeedback" else {})
        return data
    except (json.JSONDecodeError, OSError) as exc:
        logger.error("Failed to read feedback store, starting fresh: %s", exc)
        return _empty_store()


def _save(data: Dict[str, Any]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = DATA_FILE.with_suffix(".json.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, DATA_FILE)


def get_all() -> Dict[str, Any]:
    with _lock:
        return _load()


def upsert_reaction(item: Dict[str, Any]) -> Dict[str, Any]:
    """entry.id로 upsert — 클라이언트의 kbp_feedback 항목과 동일한 필드 구조."""
    with _lock:
        data = _load()
        entries: List[Dict[str, Any]] = data["reactionFeedback"]
        idx = next((i for i, e in enumerate(entries) if e.get("id") == item.get("id")), None)
        item = {**item, "ts": item.get("ts") or time.time() * 1000}
        if idx is not None:
            entries[idx] = item
        else:
            entries.append(item)
        _save(data)
        return item


def delete_reaction(item_id: str) -> bool:
    with _lock:
        data = _load()
        entries: List[Dict[str, Any]] = data["reactionFeedback"]
        before = len(entries)
        data["reactionFeedback"] = [e for e in entries if e.get("id") != item_id]
        if len(data["reactionFeedback"]) == before:
            return False
        _save(data)
        return True


def upsert_route(pw_id: str, patch: Dict[str, Optional[Any]]) -> Dict[str, Any]:
    """pwId 하나에 대해 필드 단위로 merge (예: {"feasibility": "yes"}만 보내도 됨)."""
    with _lock:
        data = _load()
        current = data["routeFeedback"].get(pw_id, {})
        merged = {**current, **patch, "ts": time.time() * 1000}
        data["routeFeedback"][pw_id] = merged
        _save(data)
        return merged


def clear_all() -> None:
    with _lock:
        _save(_empty_store())


def upsert_reagent(pw_id: str, ck: str, patch: Dict[str, Optional[Any]]) -> Dict[str, Any]:
    """pwId + candidateKey(stepKey::idx) 단위로 merge."""
    with _lock:
        data = _load()
        pw_bucket = data["reagentFeedback"].setdefault(pw_id, {})
        current = pw_bucket.get(ck, {})
        merged = {**current, **patch}
        pw_bucket[ck] = merged
        _save(data)
        return merged
