import json
import os
import threading
import uuid
from typing import Literal

from fastapi import HTTPException
from pydantic import BaseModel, Field

from case_generator import generate_case, CASE_CATEGORIES
from case_pool import put_case, take_case, public_catalog, ready_count


class CaseGenerationJobRequest(BaseModel):
    category_id: str
    difficulty: Literal["easy", "standard"] = "standard"

    # 예전 요청은 기존대로 새 사건 생성으로 처리합니다.
    start_mode: Literal["quick", "new"] = "new"

    # 반복 사건을 줄이기 위한 기기 식별값입니다.
    # 로그인 인증이나 접근 권한을 의미하지 않습니다.
    player_id: str = Field(
        default="local",
        min_length=1,
        max_length=100,
    )


_jobs = {}
_jobs_lock = threading.RLock()
_active_job_id = None


def _log(prefix, value):
    text = json.dumps(
        value,
        ensure_ascii=False,
        default=str,
    )

    for name in (
        "LLM_API_KEY",
        "OPENAI_API_KEY",
        "IMAGE_API_KEY",
        "VOICE_API_KEY",
    ):
        secret = os.getenv(name, "").strip()

        if secret:
            text = text.replace(secret, "[키 숨김]")

    print(prefix, text, flush=True)


def _trim_jobs():
    while len(_jobs) >= 50:
        removable = next(
            (
                key for key, job in _jobs.items()
                if job["status"] in ("ready", "failed")
            ),
            None,
        )

        if removable is None:
            break

        del _jobs[removable]


def _new_job(req):
    return {
        "job_id": uuid.uuid4().hex,
        "category_id": req.category_id,
        "difficulty": req.difficulty,
        "start_mode": req.start_mode,
        "player_id": req.player_id,
        "status": "queued",
        "file": "",
        "title": "",
        "message": "사건 준비를 접수했습니다.",
        "review_warning": False,
        "reused": False,
    }


def _worker(job_id, category_id, difficulty):
    global _active_job_id

    final = {
        "status": "failed",
        "message": "사건 생성에 실패했습니다. 서버 로그를 확인하세요.",
    }

    try:
        with _jobs_lock:
            _jobs[job_id].update({
                "status": "running",
                "message": "핵심 사건을 생성하고 검사하는 중입니다.",
            })

        _log(
            "[핵심 사건 생성]",
            {
                "job_id": job_id,
                "category_id": category_id,
                "difficulty": difficulty,
            },
        )

        result = generate_case(
            category_id=category_id,
            difficulty=difficulty,
            compact=True,
        )

        attempts = result.get("attempts") or []

        if not result.get("success"):
            _log("[사건 생성 작업 실패]", {
                "job_id": job_id,
                "attempts": attempts,
            })
            return

        case = result.get("case") or {}

        if (
            case.get("category_id") != category_id
            or case.get("difficulty") != difficulty
        ):
            raise ValueError("생성된 사건의 카테고리·난이도가 다릅니다.")

        # 기존 생성기는 검토 실패·검토 메모가 있는 예비 사건을
        # 성공으로 반환할 수 있으므로, 이번 보관함 등록에서는 구분합니다.
        latest_review = (
            attempts[-1].get("review")
            if attempts
            else None
        )

        if (
            latest_review != "통과"
            or case.get("review_notes")
        ):
            final["message"] = (
                "AI 검토까지 통과한 사건을 준비하지 못했습니다. "
                "저장 사건 빠른 시작을 선택하거나 나중에 다시 시도하세요."
            )

            _log("[사건 보관 보류]", {
                "job_id": job_id,
                "review": latest_review,
            })
            return

        case_id = put_case(
            case,
            source=result.get("file", ""),
            approved=True,
        )

        final = {
            "status": "ready",
            "file": f"pool_{case_id}.json",
            "title": case.get("title", ""),
            "message": (
                "새 사건의 핵심 준비가 완료되었습니다. "
                "사진과 음성은 플레이하면서 준비합니다."
            ),
            "review_warning": False,
        }

        _log("[사건 생성 작업 완료]", {
            "job_id": job_id,
            "file": final["file"],
        })

    except Exception as error:
        _log("[사건 생성 작업 오류]", {
            "job_id": job_id,
            "error_type": type(error).__name__,
            "message": str(error),
        })

    finally:
        with _jobs_lock:
            if job_id in _jobs:
                _jobs[job_id].update(final)

            if _active_job_id == job_id:
                _active_job_id = None


def _start_new(req):
    global _active_job_id

    with _jobs_lock:
        if _active_job_id:
            existing = _jobs.get(_active_job_id)

            if existing and existing["status"] in ("queued", "running"):
                same_request = (
                    existing["category_id"] == req.category_id
                    and existing["difficulty"] == req.difficulty
                    and existing["player_id"] == req.player_id
                    and existing["start_mode"] == req.start_mode
                )

                if same_request:
                    return dict(existing)

                raise HTTPException(
                    409,
                    "다른 새 사건을 생성 중입니다. "
                    "준비된 사건으로 빠른 시작하거나 기다려 주세요.",
                )

        _trim_jobs()

        job = _new_job(req)
        job_id = job["job_id"]

        _jobs[job_id] = job
        _active_job_id = job_id

        try:
            threading.Thread(
                target=_worker,
                args=(
                    job_id,
                    req.category_id,
                    req.difficulty,
                ),
                daemon=True,
            ).start()

        except Exception:
            _jobs.pop(job_id, None)
            _active_job_id = None
            raise

        return dict(job)


def request_prefill(category_id, difficulty):
    """
    선택 기능:
    이번 게임과 같은 분류의 다음 사건을 한 개만 미리 생성합니다.
    무한 반복해서 채우지는 않습니다.
    """
    enabled = os.getenv(
        "CASE_POOL_PREFETCH",
        "off",
    ).strip().lower() == "on"

    if not enabled:
        return

    if (
        category_id not in CASE_CATEGORIES
        or difficulty not in ("easy", "standard")
    ):
        return

    try:
        target = max(
            1,
            int(os.getenv("CASE_POOL_TARGET", "3")),
        )

        if ready_count(category_id, difficulty) >= target:
            return

        with _jobs_lock:
            if _active_job_id:
                return

            req = CaseGenerationJobRequest(
                category_id=category_id,
                difficulty=difficulty,
                start_mode="new",
                player_id="__server_prefill__",
            )

            _start_new(req)

        print(
            "[다음 사건 준비] 플레이 중 다음 사건 한 개를 준비합니다.",
            flush=True,
        )

    except Exception as error:
        # 다음 사건 준비 실패가 현재 게임을 막지는 않습니다.
        _log("[다음 사건 준비 보류]", {
            "error_type": type(error).__name__,
        })


def register_case_generation_jobs(app):
    @app.get("/case/pool")
    def get_pool():
        # 사건의 정답·단서 역할·비공개 정보는 반환하지 않습니다.
        entries = [
            entry for entry in public_catalog()
            if entry["approved"]
        ]

        return {
            "count": len(entries),
            "cases": entries,
        }

    @app.post("/case/jobs", status_code=202)
    def create_job(req: CaseGenerationJobRequest):
        req.category_id = req.category_id.strip()

        if req.category_id not in CASE_CATEGORIES:
            raise HTTPException(
                400,
                "지원하지 않는 카테고리입니다.",
            )

        if req.start_mode == "new":
            return _start_new(req)

        # 빠른 시작은 생성 중인 AI 작업과 별개로
        # 준비된 사건을 선택할 수 있습니다.
        job = _new_job(req)

        try:
            chosen = take_case(
                req.category_id,
                req.difficulty,
                req.player_id,
            )

            job.update({
                **chosen,
                "status": "ready",
                "message": (
                    "준비된 사건을 다시 사용합니다."
                    if chosen["reused"]
                    else "저장된 AI 사건으로 빠르게 시작합니다."
                ),
            })

        except Exception as error:
            job.update({
                "status": "failed",
                "message": str(error),
            })

        with _jobs_lock:
            _trim_jobs()
            _jobs[job["job_id"]] = job

        return dict(job)

    @app.get("/case/jobs/{job_id}")
    def get_job(job_id: str):
        with _jobs_lock:
            job = _jobs.get(job_id)

            if job is None:
                raise HTTPException(
                    404,
                    "작업 기록이 없습니다. "
                    "서버를 재시작했다면 Unity의 Play도 다시 실행하세요.",
                )

            # 상태 확인은 AI를 호출하지 않습니다.
            return dict(job)