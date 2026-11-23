import copy
import uuid
from typing import Literal

from fastapi import HTTPException
from pydantic import BaseModel

from validator import validate_case
from game_rules import (
    new_accuse_state,
    new_investigation_state,
    public_areas,
)
from question_bank import question_list, clue_questions


class FinishRoundRequest(BaseModel):
    game_id: str
    reason: Literal["timeout", "failed"]


class RoundActionRequest(BaseModel):
    game_id: str


def register_case_round_flow(
    app,
    active_games,
    public_evidences,
    build_briefing,
):
    def get_session(game_id):
        session = active_games.get(game_id)

        if session is None:
            raise HTTPException(
                404,
                "게임 기록을 찾을 수 없습니다.",
            )

        return session

    def require_failed_round(session):
        accuse = session.get("accuse", {})

        if not accuse.get("finished"):
            raise HTTPException(
                409,
                "진행 중인 게임에서는 이 기능을 사용할 수 없습니다.",
            )

        if (accuse.get("result") or {}).get("success"):
            raise HTTPException(
                409,
                "이 기능은 실패한 수사의 재도전용입니다.",
            )

    def make_start_response(game_id, session):
        game = session["truth"]
        investigation = session["investigation"]

        return {
            "game_id": game_id,
            "case_title": game.get("title", ""),
            "case_summary": game.get("case_summary", ""),
            "category_id": game.get("category_id", ""),
            "category_name": game.get("category_name", ""),
            "difficulty": game.get("difficulty", "standard"),
            "suspects": [
                {
                    "suspect_id": suspect_id,
                    "name": suspect["name"],
                    "description": suspect.get("description", ""),
                }
                for suspect_id, suspect in game["suspects"].items()
            ],
            "evidences": public_evidences(
                game,
                investigation["found"],
            ),
            "investigation": investigation["enabled"],
            "areas": public_areas(investigation),
            "briefing": build_briefing(game),
            "questions": [
                {
                    "suspect_id": suspect_id,
                    "categories": question_list(game, suspect_id),
                }
                for suspect_id in game["suspects"]
            ],
            "clue_questions": clue_questions(
                game,
                investigation["found"],
            ),
        }

    @app.post("/game/finish")
    def finish_round(req: FinishRoundRequest):
        session = get_session(req.game_id)
        accuse = session["accuse"]

        # 이미 종료된 게임의 판정 상태는 그대로 유지합니다.
        if accuse.get("finished"):
            return {
                "finished": True,
                "message": "종료된 게임입니다.",
            }

        # 답 종료는 기존 지목 판정이 확정해야 합니다.
        # 이 API로 남은 지목 기회를 생략하지 않습니다.
        if req.reason == "failed":
            raise HTTPException(
                409,
                "아직 최종 실패로 판정되지 않은 게임입니다.",
            )

        # 시간 종료를 서버에도 반영합니다.
        accuse["finished"] = True
        accuse["result"] = {
            "success": False,
            "game_over": True,
            "reason": "timeout",
            "message": "제한시간이 끝났습니다.",
        }

        return {
            "finished": True,
            "message": "시간 초과로 수사를 종료했습니다.",
        }

    @app.post("/game/retry")
    def retry_round(req: RoundActionRequest):
        old_session = get_session(req.game_id)
        require_failed_round(old_session)

        if old_session.get("solution_revealed"):
            raise HTTPException(
                409,
                "정답을 확인한 사건은 재도전할 수 없습니다.",
            )

        # 통신 재시도로 같은 요청이 다시 와도
        # 새 세션을 여러 개 만들지 않습니다.
        existing_retry_id = old_session.get("retry_game_id")

        if existing_retry_id:
            retry_session = get_session(existing_retry_id)

            if retry_session.get("solution_revealed"):
                raise HTTPException(
                    409,
                    "재도전한 사건의 정답이 이미 공개되었습니다.",
                )

            return make_start_response(
                existing_retry_id,
                retry_session,
            )

        original = old_session.get("original_case")

        if original is None:
            raise HTTPException(
                409,
                "이 게임에는 재도전용 사건 원본이 없습니다. "
                "수정된 서버에서 새 게임을 시작하세요.",
            )

        game = copy.deepcopy(original)

        for suspect in game["suspects"].values():
            suspect["pressure_level"] = 0
            suspect["confessed"] = False
            suspect["presented_evidence"] = []
            suspect["asked_questions"] = []
            suspect["answer_cache"] = {}
            suspect["chat_log"] = []
            suspect["statements"] = []

        errors = validate_case(game)

        if errors:
            raise HTTPException(
                500,
                "재도전 사건의 검증에 실패했습니다.",
            )

        new_game_id = uuid.uuid4().hex[:8]

        while new_game_id in active_games:
            new_game_id = uuid.uuid4().hex[:8]

        new_session = {
            "truth": game,
            "case_source": old_session.get("case_source", ""),
            "accuse": new_accuse_state(),
            "investigation": new_investigation_state(game),
            "original_case": copy.deepcopy(original),
            "solution_revealed": False,
        }

        active_games[new_game_id] = new_session
        old_session["retry_game_id"] = new_game_id

        # AI 사건 생성은 호출하지 않습니다.
        return make_start_response(new_game_id, new_session)

    @app.post("/game/solution")
    def reveal_solution(req: RoundActionRequest):
        session = get_session(req.game_id)
        require_failed_round(session)

        # 이전 세션을 이용해 재도전 중인 사건의
        # 정답을 미리 열람하는 것을 막습니다.
        if session.get("retry_game_id"):
            raise HTTPException(
                409,
                "이 게임은 재도전으로 이어졌습니다. "
                "현재 게임을 종료한 뒤 정답을 확인하세요.",
            )

        if session.get("final_solution"):
            return session["final_solution"]

        game = session["truth"]
        culprit = game["suspects"].get(game["culprit_id"])

        if culprit is None:
            raise HTTPException(
                500,
                "사건 정답의 인물 정보를 확인하지 못했습니다.",
            )

        evidence_id = game.get("decisive_evidence_id", "")
        evidence = game["evidence_db"].get(evidence_id, {})

        result = {
            "case_title": game.get("title", ""),
            "culprit_name": culprit.get("name", ""),
            "method": (
                game.get("truth")
                or "이 사건에는 상세한 범행 경위가 등록되지 않았습니다."
            ),
            "decisive_evidence": {
                "evidence_id": evidence_id,
                "name": evidence.get("name", ""),
                "description": evidence.get("desc", ""),
            },
        }

        session["solution_revealed"] = True
        session["final_solution"] = result

        # 새 AI 해설이 아니라 저장된 정답을 반환합니다.
        return result