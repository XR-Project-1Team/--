import os
import sys
import json
import random
import shutil
from datetime import datetime
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from validator import validate_generated_case, CLUE_ROLE_LIMITS
from case_quality import check_case_quality, split_warnings
from usage_counter import counted_call

# .env 파일의 API 키를 불러옴
load_dotenv()

# 심문 AI와 같은 OpenRouter 키를 쓰되, 사건 생성 전용 모델을 따로 지정할 수 있음
client = OpenAI(
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv(
        "LLM_BASE_URL",
        "https://monogpt.kr/api/monorouter/v1/openai/v1",
    ),
    timeout=120,
    max_retries=0,
)

CASE_MODEL_NAME = os.getenv(
    "CASE_LLM_MODEL",
    os.getenv("LLM_MODEL", "gpt-5.5"),
)

# 검증을 통과한 사건이 저장되는 폴더
CASES_DIR = Path(__file__).parent / "cases"
# 개수 제한을 넘은 오래된 사건을 옮겨 두는 폴더 (게임에서는 쓰지 않음, 지우지 않고 보관)
OLD_CASES_DIR = CASES_DIR / "old"
MAX_ACTIVE_CASES = 10

MAX_ATTEMPTS = 4

# 매번 다른 사건이 나오도록 무작위로 섞어 줄 재료 (레고 블록)
CASE_TYPES = ["살인", "독살", "실종 후 사망", "추락사로 위장된 살인"]
PLACES = ["호텔", "산장", "미술관", "대학 연구실", "유람선", "고급 레스토랑", "작은 출판사", "시골 펜션"]

CASE_CATEGORIES = {
    "locked_room": {
        "name": "밀실 미스터리",
        "places": ["호텔", "산장", "저택", "시골 펜션"],
        "instruction": (
            "출입이 제한된 장소에서 벌어진 사건을 만듭니다. "
            "잠금 상태, 출입 방법, 인물의 이동 시간 사이에 "
            "플레이어가 증거로 풀 수 있는 핵심 수수께끼를 만듭니다. "
            "밀실이 성립한 이유와 범행 후 이동 방법을 "
            "비공개 truth에 논리적으로 설명합니다."
        ),
    },
    "research": {
        "name": "학교·연구소",
        "places": ["대학 연구실", "공동 실험실", "민간 연구소", "대학 연구동"],
        "instruction": (
            "연구 성과, 실험 기록, 연구비 또는 논문을 둘러싼 "
            "갈등이 사건의 중요한 배경이 되게 합니다. "
            "출입 기록과 연구 자료를 비교해 추리할 수 있게 합니다. "
            "전문지식이 없어도 공개된 단서로 해결할 수 있어야 합니다."
        ),
    },
    "corporate": {
        "name": "기업·비밀 거래",
        "places": ["기업 본사", "스타트업 사무실", "출판사", "물류회사 사무실"],
        "instruction": (
            "계약, 금전 거래, 회사 정보 또는 업무상 이해관계를 "
            "사건의 중요한 배경으로 사용합니다. "
            "이메일, 거래 내역, 업무 기록 등을 조합해 추리하게 합니다. "
            "비리를 숨기는 행동과 살인에 관한 거짓말을 구분합니다."
        ),
    },
    "psychology": {
        "name": "심리·기억 미스터리",
        "places": ["개인 상담실", "심리상담센터", "기억 연구시설", "심리검사센터"],
        "instruction": (
            "기억의 불일치, 오해, 의도적인 거짓말 또는 "
            "상담 기록의 차이가 사건의 중요한 요소가 되게 합니다. "
            "정신질환이나 트라우마가 있다는 이유만으로 "
            "인물을 범인으로 설정하지 않습니다. "
            "매번 기억을 잃은 사람이 범인인 구조를 반복하지 않습니다. "
            "사건은 물리적인 증거와 시간 관계로 해결 가능해야 하며, "
            "플레이어가 반드시 범인의 자백을 받아야만 "
            "해결되는 구조로 만들지 않습니다."
        ),
    },
}

ROLE_RULES = "\n".join(
    f"    - {role}: {low}~{high}개" for role, (low, high) in CLUE_ROLE_LIMITS.items()
)

SYSTEM_PROMPT = f"""
당신은 추리게임의 '사건 설계 AI'입니다. 논리적으로 모순이 없는 사건 하나를 만듭니다.
반드시 한국어로 작성하고, 아래 JSON 구조로만 답하십시오. 설명 문장은 쓰지 마십시오.

[규칙]
1. 용의자는 정확히 3명이고 suspect_id는 1, 2, 3 입니다. 그중 1명만 범인입니다.
2. 단서는 9~12개이고 evidence_id는 "E01"부터 차례대로 붙입니다.
3. 단서마다 role(역할 태그)을 하나 붙입니다. 개수 규칙:
{ROLE_RULES}
   - 결정적: 범인만 가리키는 단서 (related_suspect_id = 범인)
   - 정황: 범인 쪽으로 기울게 하는 단서 (related_suspect_id = 범인)
   - 알리바이: 무고한 용의자의 결백 증명. 무고한 2명에게 1개씩 (related_suspect_id = 그 무고한 사람)
   - 모순: 누군가의 거짓 진술을 반박. 최소 1개는 범인의 알리바이를 깨야 함
   - 미스리드: 무고한 사람을 의심하게 만드는 함정 (related_suspect_id = 무고한 사람)
4. decisive_evidence_id는 role이 "결정적"인 단서 중 하나입니다.
5. 범인의 evidence_effects에서 decisive_evidence_id의 값은 30~45입니다.
6. evidence_effects는 그 용의자를 압박하는 단서 ID와 압박도 증가량(5~45)입니다.
7. 범인의 forbidden에는 반드시 "자신이 범인이라는 사실"을 넣습니다.
   무고한 용의자의 forbidden에는 범인 관련 문장을 넣지 말고, 그 사람이 숨기고 싶은 개인 비밀을 넣습니다.
8. description은 플레이어에게 공개되는 짧은 소개(직업 + 진술)이며 범인을 암시하지 않습니다.
9. 단서의 desc는 플레이어가 보는 설명이며 role 이름을 직접 쓰지 않습니다.
10. 단서의 name과 desc에는 "결정적", "정황", "알리바이", "모순", "미스리드", "범인" 같은 단어를 쓰지 않습니다.
    단서 이름은 물건이나 기록의 이름만 씁니다. (나쁜 예: "지민재의 알리바이" / 좋은 예: "주방 출입 기록")
11. 단서 설명에 범인의 이름을 직접 쓰지 않습니다. 플레이어가 심문과 증거 조합으로 범인을 추리해야 합니다.
12. 단서의 desc는 role과 맞아야 합니다. 정황·결정적 단서가 오히려 범인의 알리바이를 뒷받침하면 안 됩니다.
13. 자연스럽고 맞춤법이 맞는 한국어로 씁니다.
14. culprit_id는 사용자가 지정한 번호를 그대로 씁니다. (예시 JSON의 번호를 따라 하지 않음)
15. 모든 용의자의 pressure_level은 0입니다.
16. 단서 이름에는 사람 이름을 넣지 않습니다. 결정적·정황 단서 설명에도 범인 이름을 직접 쓰지 않고 직업·소지품·습관으로 간접적으로 가리킵니다.
17. 범인을 가리키는 결정적·정황·모순 단서는 모두 범인의 evidence_effects에 넣고,
    무고한 사람을 의심하게 만드는 미스리드·모순 단서는 그 사람의 evidence_effects에 넣습니다.
18. 세 용의자의 speech_style은 서로 확실히 다르게 만듭니다. (예: 격식체 냉정형 / 반말 섞인 다혈질형 / 말줄임표 많은 소심형)
19. 단서마다 area(플레이어가 조사하는 구역)를 붙입니다. 구역은 3~5개이고 한 구역에 단서 1~4개를 넣습니다.
    found_at이 비슷한 단서는 같은 구역으로 묶습니다. (예: '5층 테라스', '피해자 연구실', '보안 기록', '검시 결과')
    구역 이름에는 사람 이름을 쓰지 않고, 결정적 단서를 모두 한 구역에 몰아넣지 않습니다.
20. briefing은 게임을 시작할 때 형사(플레이어)에게 보여 주는 수사 브리핑 3~5문장입니다.
    발견 경위, 현장 상황, 형사가 할 일(현장 조사와 심문)을 씁니다.
    용의자 이름, 숨긴 비밀, 단서 역할, 범인을 짐작하게 하는 내용은 쓰지 않습니다.
21. 용의자마다 questions 4~6개를 만듭니다. 형사가 그 사람에게 물어볼 만한 맞춤 질문입니다.
    category는 "관계", "알리바이", "감정" 중 하나이고, 세 카테고리를 골고루 씁니다.
    질문은 공개 정보(description, alibi, public)만 근거로 하고, 단서 이름·숨긴 비밀·범인 여부를 직접 묻지 않습니다.
22. 단서마다 question 1개를 만듭니다. 플레이어가 그 단서를 찾은 뒤 실물을 보여 주지 않고 말로 떠보는 질문입니다.
    (예: '문서 파쇄실 출입 기록표' → "그날 밤 1층 문서 파쇄실 세단기를 누가 쓰고 있었는지 압니까?")
    세 용의자 누구에게 물어도 어색하지 않게 쓰고, 범인 이름과 역할 이름은 쓰지 않습니다.

[자료형 및 공개 문구 추가 규칙]
- public, private, forbidden은 반드시 문자열 배열로 작성합니다.
  항목이 한 개여도 배열로 작성하며, 빈 문자열을 넣지 않습니다.
- private와 forbidden에 같은 문장을 중복해서 넣지 않습니다.
  private에는 질문에 따라 드러날 수 있는 개인 정보를,
  forbidden에는 자백 조건 전까지 숨길 정보를 구분해 작성합니다.
- briefing에는 아래 단어와 표현을 쓰지 않습니다:
  결정적, 정황, 알리바이, 모순, 미스리드, 범인, 진범,
  무고한, 함정, 용의자 1, 용의자 2, 용의자 3.
- briefing은 발견 상황, 장소, 조사할 일과 심문할 일만 설명합니다.
- 위 briefing 문구 제한을 비공개 truth나 role 필드에 적용하지 않습니다.
[JSON 구조]
{{
  "title": "사건 제목",
  "case_summary": "플레이어에게 보여 줄 사건 개요 1~2문장",
  "briefing": "게임 시작 때 보여 줄 수사 브리핑 3~5문장",
  "victim": "피해자 이름과 신분",
  "location": "사건 장소",
  "time_of_death": "사망 추정 시각",
  "truth": "실제로 무슨 일이 있었는지 (플레이어에게 비공개)",
  "culprit_id": 1,
  "decisive_evidence_id": "E01",
  "suspects": [
    {{
      "suspect_id": 1,
      "name": "이름",
      "job": "직업 또는 피해자와의 관계",
      "personality": "성격",
      "speech_style": {{
        "honorific": "격식체 존댓말 / 해요체 / 반말 섞임 중 하나와 설명",
        "sentence": "문장 길이와 리듬",
        "habit": "자주 쓰는 말버릇",
        "evasion": "곤란한 질문을 피하는 방식",
        "under_pressure": "압박받을 때 말투 변화",
        "examples": ["이 사람이 할 법한 대사 1", "대사 2"]
      }},
      "description": "공개 소개",
      "alibi": "본인이 주장하는 알리바이",
      "public": ["물어보면 말해 주는 정보"],
      "private": ["숨기는 비밀"],
      "forbidden": ["절대 말하면 안 되는 정보"],
      "evidence_effects": {{"E03": 20}},
      "pressure_level": 0,
      "questions": [
        {{"category": "알리바이", "text": "이 사람에게 물어볼 맞춤 질문"}}
      ]
    }}
  ],
  "evidences": [
    {{
      "evidence_id": "E01",
      "name": "단서 이름",
      "desc": "단서 설명",
      "found_at": "발견 장소",
      "area": "조사 구역",
      "question": "단서를 찾은 뒤 말로 떠보는 질문",
      "role": "결정적",
      "related_suspect_id": 1
    }}
  ]
}}
"""


# 규칙 검사기가 못 잡는 '이야기 품질'을 보는 검토 AI의 지시문
REVIEW_PROMPT = """
당신은 추리게임의 '사건 검토 AI'입니다. 다른 AI가 만든 사건을 읽고 게임을 망칠 만한 심각한 문제만 찾습니다.
입력은 두 부분입니다.
- player_view: 플레이어가 실제로 보는 공개 정보 (사건 개요, 용의자 소개, 단서 이름·설명)
- answer_sheet: 플레이어에게 숨겨진 정답지 (범인, 진실, 단서 역할, 용의자 비밀). 정답지에 범인 정보가 있는 것은 정상입니다.

[검토 기준]
1. 정답 노출: player_view만 읽고도 추리 없이 범인을 확신할 수 있는가? (예: 단서 이름·설명에 범인 이름과 범행이 함께 적힘)
2. 역할 불일치: 단서 설명이 answer_sheet의 role과 맞지 않는가?
   (예: 정황·결정적 단서가 오히려 범인의 결백을 뒷받침함, 알리바이 단서가 그 사람을 의심하게 만듦)
   ※ 모순 단서가 누군가의 알리바이와 부딪히는 것은 의도된 설계이므로 문제가 아닙니다.
3. 사실 모순: 의도되지 않은 시간·장소·인물 관계 충돌이 있는가?
4. 한국어: 뜻이 통하지 않는 단어, 외국어가 섞인 단어가 있는가?
5. 요청 주제: requested_category가 비어 있지 않다면,
   사건의 동기·단서·공개 정보에 해당 주제가 실질적으로 반영되어 있는가?
   장소 이름만 주제에 맞고 사건 내용은 무관하면 문제로 지적합니다.
   requested_category가 비어 있다면 이 기준은 적용하지 않습니다.

사소한 취향 문제는 적지 마십시오. 문제는 최대 5개, 단서 ID나 용의자 번호를 붙여 한 문장씩 씁니다.
반드시 아래 JSON으로만 답하십시오. problems는 문자열 목록입니다.
{"pass": true, "problems": []}
"""


def _extract_json(text: str) -> dict:
    """AI 응답에서 JSON 부분만 꺼냅니다. (```json 같은 표시가 섞여 와도 처리)"""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("응답에서 JSON을 찾을 수 없습니다.")
    return json.loads(text[start:end + 1])

def _to_text_list(value, field_name: str) -> list[str]:
    """
    문장 하나가 문자열로 오면 한 항목짜리 목록으로 바꿉니다.
    문장 내용이나 private/forbidden 분류는 바꾸지 않습니다.
    """
    if value is None:
        return []

    if isinstance(value, str):
        text = value.strip()

        if not text:
            raise ValueError(
                f"{field_name}에 빈 문자열이 들어 있습니다."
            )

        print(
            f"[사건 형식 보정] {field_name}: "
            "문자열을 한 문장짜리 목록으로 변환했습니다.",
            flush=True,
        )

        return [text]

    if not isinstance(value, list):
        raise ValueError(
            f"{field_name}은 문자열 목록이어야 합니다. "
            f"현재 자료형: {type(value).__name__}"
        )

    result = []

    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise ValueError(
                f"{field_name}[{index}]는 "
                "비어 있지 않은 문자열이어야 합니다."
            )

        result.append(item.strip())

    return result

def to_server_truth(raw: dict) -> dict:
    """AI가 만든 목록 형태의 사건을 서버가 쓰는 딕셔너리 형태로 바꿉니다."""
    suspects = {}
    for s in raw.get("suspects", []):
        suspect_id = int(s["suspect_id"])
        suspects[suspect_id] = {
            "name": s.get("name", ""),
            "job": s.get("job", ""),
            "personality": s.get("personality", ""),
            "speech_style": s.get("speech_style") or {},
            "description": s.get("description", ""),
            "alibi": s.get("alibi", ""),
            "knowledge": {
                "public": _to_text_list(
                    s.get("public", []),
                    f"용의자 {suspect_id}.public",
                ),
                "private": _to_text_list(
                    s.get("private", []),
                    f"용의자 {suspect_id}.private",
                ),
                "forbidden": _to_text_list(
                    s.get("forbidden", []),
                    f"용의자 {suspect_id}.forbidden",
                ),
            },
            "evidence_effects": {
                str(e_id): v for e_id, v in (s.get("evidence_effects") or {}).items()
            },
            "pressure_level": s.get("pressure_level", 0),
            "questions": [
                {"category": str(q.get("category", "")).strip(), "text": str(q.get("text", "")).strip()}
                for q in (s.get("questions") or []) if isinstance(q, dict)
            ],
        }

    evidence_db = {}
    for e in raw.get("evidences", []):
        related = e.get("related_suspect_id")
        evidence_db[str(e["evidence_id"])] = {
            "name": e.get("name", ""),
            "desc": e.get("desc", ""),
            "found_at": e.get("found_at", ""),
            "area": e.get("area", ""),
            "question": e.get("question", ""),
            "role": e.get("role", ""),
            "related_suspect_id": int(related) if related is not None else None,
        }

    return {
        "title": raw.get("title", ""),
        "case_summary": raw.get("case_summary", ""),
        "briefing": raw.get("briefing", ""),
        "victim": raw.get("victim", ""),
        "location": raw.get("location", ""),
        "time_of_death": raw.get("time_of_death", ""),
        "truth": raw.get("truth", ""),
        "culprit_id": int(raw.get("culprit_id", 0)),
        "decisive_evidence_id": str(raw.get("decisive_evidence_id", "")),
        "suspects": suspects,
        "evidence_db": evidence_db,
    }
class CaseRequestStopped(RuntimeError):
    """반복 생성으로 해결하지 않고 현재 작업을 중단할 오류입니다."""
    pass


def _call_ai(
    messages: list,
    temperature: float = 0.9,
    kind: str = "사건 생성",
) -> str:
    import time
    from openai import APITimeoutError

    # 기존 호출부와의 호환성을 위해 인수는 유지합니다.
    # temperature는 이번 요청에 전달하지 않습니다.
    max_tokens = int(
        os.getenv("CASE_MAX_TOKENS", "16000")
    )

    timeout_seconds = float(
        os.getenv("CASE_API_TIMEOUT_SECONDS", "180")
    )

    request_client = client.with_options(
        timeout=timeout_seconds,
        max_retries=0,
    )

    def receive_text():
        pieces = []
        finish_reason = None
        refused = False
        actual_model = CASE_MODEL_NAME

        started = time.monotonic()
        received_first_text = False

        print(
            f"[사건 AI 요청] 종류={kind}, "
            f"모델={CASE_MODEL_NAME}, 스트리밍=True",
            flush=True,
        )

        with request_client.chat.completions.create(
            model=CASE_MODEL_NAME,
            messages=messages,
            response_format={"type": "json_object"},
            max_completion_tokens=max_tokens,
            stream=True,
        ) as stream:
            for chunk in stream:
                actual_model = (
                    getattr(chunk, "model", None)
                    or actual_model
                )

                for choice in chunk.choices or []:
                    delta = choice.delta

                    if getattr(delta, "refusal", None):
                        refused = True

                    content = getattr(delta, "content", None)

                    if isinstance(content, str) and content:
                        pieces.append(content)

                        if not received_first_text:
                            received_first_text = True

                            print(
                                f"[사건 AI 수신 시작] 종류={kind}, "
                                f"경과={time.monotonic() - started:.1f}초",
                                flush=True,
                            )

                    if choice.finish_reason is not None:
                        finish_reason = choice.finish_reason

        text = "".join(pieces)

        # 사건 원문·API 키는 로그에 출력하지 않습니다.
        print(
            f"[사건 AI 응답] 종류={kind}, "
            f"모델={actual_model}, "
            f"종료={finish_reason}, "
            f"문자수={len(text)}, "
            f"경과={time.monotonic() - started:.1f}초",
            flush=True,
        )

        # 필터로 제한된 응답을 JSON 파싱 오류로 넘기지 않습니다.

        if finish_reason == "content_filter" or refused:
            from datetime import datetime
            from pathlib import Path

            # GitHub 저장소 밖에 진단용 응답만 보관합니다.
            # API 키, 요청 헤더, 전체 프롬프트는 저장하지 않습니다.
            safe_text = text

            for variable_name in (
                "LLM_API_KEY",
                "OPENAI_API_KEY",
                "IMAGE_API_KEY",
                "VOICE_API_KEY",
            ):
                secret = os.getenv(variable_name, "").strip()

                if secret:
                    safe_text = safe_text.replace(
                        secret,
                        "[키 숨김]",
                    )

            try:
                debug_dir = (
                    Path.home()
                    / ".vr_detective_private"
                )

                debug_dir.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                stamp = datetime.now().strftime(
                    "%Y%m%d_%H%M%S_%f"
                )

                debug_path = debug_dir / (
                    f"filtered_response_{stamp}.txt"
                )

                debug_path.write_text(
                    f"요청 종류: {kind}\n"
                    f"모델: {actual_model}\n"
                    f"종료 이유: {finish_reason}\n"
                    f"거절 필드 감지: {refused}\n\n"
                    "수신한 응답:\n"
                    + safe_text,
                    encoding="utf-8",
                )

                print(
                    "[필터 응답 진단 파일]",
                    debug_path,
                    flush=True,
                )

            except OSError:
                print(
                    "[필터 응답 진단] 파일을 저장하지 못했습니다.",
                    flush=True,
                )

            raise CaseRequestStopped(
                "AI 응답이 내용 필터로 종료되었습니다. "
                "자동 재요청을 중단했습니다. "
                "서버의 진단 파일을 확인하세요."
            )

        if finish_reason == "length":
            raise ValueError(
                "AI 출력이 한도에 도달해 잘렸습니다. "
                f"요청 출력 한도={max_tokens}, "
                f"수신 문자수={len(text)}"
            )

        # 스트림이 중간에 끝나면 불완전한 사건을 사용하지 않습니다.
        if finish_reason != "stop":
            raise CaseRequestStopped(
                "AI 응답이 정상적으로 끝나지 않았습니다. "
                f"종료 이유={finish_reason}"
            )

        if not text.strip():
            raise CaseRequestStopped(
                "AI가 텍스트를 반환하지 않았습니다."
            )

        if "{" not in text or "}" not in text:
            raise ValueError(
                "AI 응답에서 JSON 객체를 찾지 못했습니다. "
                f"수신 문자수={len(text)}"
            )

        return text

    try:
        # 스트림 전체를 받는 과정까지 기존 사용량 집계로 감쌉니다.
        return counted_call(kind, receive_text)

    except CaseRequestStopped:
        raise

    except APITimeoutError as error:
        raise CaseRequestStopped(
            "AI 응답 대기시간이 초과되었습니다. "
            "외부 서비스에서는 처리가 진행됐을 수 있으므로 "
            "자동 재요청을 중단했습니다."
        ) from error

    except Exception as error:
        status_code = getattr(error, "status_code", None)

        if status_code in (401, 402, 403, 429, 524):
            raise CaseRequestStopped(
                f"외부 AI 서비스 HTTP {status_code} 오류로 "
                "현재 생성을 중단했습니다. "
                "사용 통계와 서비스 상태를 확인하세요."
            ) from error

        raise


def review_case(case: dict) -> list[str] | None:
    """
    검토 AI가 사건의 이야기 문제를 찾습니다.
    문제 목록을 반환하고, 문제가 없으면 빈 리스트,
    검토 AI 호출 자체가 실패하면 None을 반환합니다.
    """
    # 플레이어가 보는 정보와 숨겨진 정답지를 나눠서 보여 줌
    review_input = {
        "requested_category": case.get("category_name", ""),
        "player_view": {
            "case_summary": case["case_summary"],
            "suspects": {s_id: {"name": s["name"], "description": s["description"]}
                         for s_id, s in case["suspects"].items()},
            "clues": {e_id: {"name": e["name"], "desc": e["desc"]}
                      for e_id, e in case["evidence_db"].items()},
        },
        "answer_sheet": {
            "culprit_id": case["culprit_id"],
            "truth": case.get("truth", ""),
            "suspects": {s_id: {"alibi": s["alibi"], "private": s["knowledge"]["private"]}
                         for s_id, s in case["suspects"].items()},
            "clue_roles": {e_id: {"role": e["role"], "related_suspect_id": e["related_suspect_id"]}
                           for e_id, e in case["evidence_db"].items()},
        },
    }
    case_text = json.dumps(review_input, ensure_ascii=False, default=str)
    try:
        content = _call_ai(
            [
                {"role": "system", "content": REVIEW_PROMPT},
                {"role": "user", "content": case_text},
            ],
            temperature=0.2,
            kind="사건 검토",
        )
        result = _extract_json(content)
    except CaseRequestStopped:
        raise

    except Exception:
        return None

    problems = [str(p) for p in result.get("problems", []) if str(p).strip()]
    if result.get("pass") is True and not problems:
        return []
    return problems or ["검토 AI가 통과시키지 않았지만 이유를 적지 않았습니다."]


def generate_case(
    case_type: str | None = None,
    place: str | None = None,
    culprit_id: int | None = None,
    category_id: str | None = None,
    difficulty: str = "standard",
    compact: bool = False,
) -> dict:
    """
    AI로 사건을 만들고 두 단계로 확인합니다.
      1단계 규칙 검사기: 용의자 수, 단서 역할 태그 등 (실패하면 반드시 다시 생성)
      2단계 검토 AI: 정답 노출, 역할 불일치, 모순, 어색한 한국어
    문제가 있으면 오류 목록을 생성 AI에게 돌려주고 고치게 합니다. (최대 MAX_ATTEMPTS번)
    끝까지 검토를 통과하지 못하면, 규칙은 통과한 가장 최근 사건에 검토 지적을 review_notes로 붙여 저장합니다.
    결과: {"success", "case", "file", "attempts"} 형태의 딕셔너리
    """
    category = None

    if category_id is not None:
        category = CASE_CATEGORIES.get(category_id)

        if category is None:
            raise ValueError(
                f"지원하지 않는 사건 카테고리입니다: {category_id}"
            )

    case_type = case_type or random.choice(CASE_TYPES)

    available_places = (
        category["places"] if category is not None else PLACES
    )

    place = place or random.choice(available_places)
    culprit_id = culprit_id or random.choice([1, 2, 3])

    category_text = ""

    if category is not None:
        category_text = (
            f"선택한 사건 카테고리: {category['name']}\n"
            f"주제별 추가 조건: {category['instruction']}\n"
            "장소 이름만 바꾸지 말고, 동기·단서·공개 질문에도 "
            "선택한 주제가 반영되게 하십시오.\n"
        )

        # 범인은 기존 코드에서 정한 값을 그대로 사용합니다.
    innocent_ids = [
        suspect_id
        for suspect_id in (1, 2, 3)
        if suspect_id != culprit_id
    ]

    clue_plan = [
        {
            "evidence_id": "E01",
            "role": "결정적",
            "related_suspect_id": culprit_id,
        },
        {
            "evidence_id": "E02",
            "role": "결정적",
            "related_suspect_id": culprit_id,
        },
        {
            "evidence_id": "E03",
            "role": "정황",
            "related_suspect_id": culprit_id,
        },
        {
            "evidence_id": "E04",
            "role": "정황",
            "related_suspect_id": culprit_id,
        },
        {
            "evidence_id": "E05",
            "role": "알리바이",
            "related_suspect_id": innocent_ids[0],
        },
        {
            "evidence_id": "E06",
            "role": "알리바이",
            "related_suspect_id": innocent_ids[1],
        },
        {
            "evidence_id": "E07",
            "role": "모순",
            "related_suspect_id": culprit_id,
        },
        {
            "evidence_id": "E08",
            "role": "모순",
            "related_suspect_id": innocent_ids[0],
        },
        {
            "evidence_id": "E09",
            "role": "미스리드",
            "related_suspect_id": innocent_ids[1],
        },
    ]

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": (
                f"사건 유형: {case_type}\n"
                f"장소: {place}\n"
                f"범인: suspect_id {culprit_id}번 "
                f"(culprit_id = {culprit_id})\n"
                + category_text
                + "\n[이번 사건의 증거 설계표]\n"
                + json.dumps(clue_plan, ensure_ascii=False)
                + "\n\n[추가 작성 지침]\n"
                "1. 이번 사건의 단서는 정확히 9개로 작성합니다. "
                "E01~E09를 사용하고 위 설계표의 role과 "
                "related_suspect_id를 그대로 따릅니다.\n"
                "2. decisive_evidence_id는 E01로 설정합니다. "
                "범인의 evidence_effects에서 E01의 값은 "
                "기존 규칙대로 30~45로 설정합니다.\n"
                "3. 범인의 evidence_effects에는 "
                "E01, E02, E03, E04, E07을 모두 포함합니다. "
                f"용의자 {innocent_ids[0]}의 evidence_effects에는 "
                "E08을 포함하고, "
                f"용의자 {innocent_ids[1]}의 evidence_effects에는 "
                "E09를 포함합니다. "
                "다른 규칙상 필요한 연결도 빠뜨리지 않습니다.\n"
                "4. 증거의 name과 desc에는 모든 용의자의 이름을 "
                "쓰지 않습니다. 직업, 물건의 특징, 시간, "
                "출입 기록 등으로 간접적으로 설명합니다.\n"
                "5. name과 desc에는 '결정적', '정황', '알리바이', "
                "'모순', '미스리드', '범인'이라는 단어를 "
                "쓰지 않습니다. role 필드에는 정해진 역할명을 씁니다.\n"
                "6. 문장은 간결하게 쓰되, 시간·장소·인물 관계와 "
                "추리에 필요한 사실은 생략하지 않습니다. "
                "desc는 1~2문장, truth는 3~5문장으로 작성합니다. "
                "briefing은 기존 규칙대로 3~5문장입니다.\n"
                "7. 용의자 맞춤 질문은 각 4개로 작성하되 "
                "관계·알리바이·감정 분류를 모두 포함합니다. "
                "각 증거의 question도 반드시 작성합니다.\n"
                "8. 조사 구역은 3개로 구성하고 "
                "각 구역에 단서를 3개씩 배치합니다. "
                "E01과 E02는 서로 다른 구역에 배치합니다.\n"
                "9. speech_style, public, private, forbidden 등 "
                "기존 JSON의 필수 항목을 생략하지 않습니다. "
                "설명문·마크다운 없이 완성된 JSON 하나만 반환합니다.\n"
                "10. 기존 규칙과 위 조건을 모두 지켜 "
                "선택한 카테고리의 새 사건을 작성하십시오."
            ),
        },
    ]
    messages.append({
        "role": "user",
        "content": (
            "[사건 묘사 방식]\n"
            "이 사건은 허구의 추리 게임에 사용됩니다. "
            "핵심은 용의자의 진술과 출입 기록, 시간, "
            "물건의 특징을 비교해 진실을 찾는 것입니다.\n"
            "사건의 피해나 사망 사실은 간결하게 서술하고, "
            "상처·혈흔 등의 잔혹한 묘사와 "
            "범행 방법의 단계별 세부 설명은 피합니다.\n"
            "비공개 truth에는 누가 어떤 동기로 사건에 "
            "관여했는지와 증거·시간의 연결 관계를 "
            "이해할 수 있을 만큼만 간결하게 설명합니다.\n"
            "기존 용의자 수, 증거 역할, 조사 구역, "
            "질문, 카테고리 및 난이도 조건은 유지합니다."
        ),
    })

    if compact:
        messages.append({
            "role": "user",
            "content": (
                "[간결한 초기 사건 작성]\n"
                "게임 시작에 필요한 완전한 사건을 간결하게 작성합니다.\n"
                "범인, 동기, 시간 관계, 용의자 지식, 모든 필수 단서와 "
                "판정에 필요한 정보는 이번 응답에서 모두 확정합니다.\n"
                "1. 단서는 기존 조건 안에서 정확히 9개로 작성합니다.\n"
                "2. 조사 구역은 3개, 각 구역의 단서는 3개로 배치하고 "
                "결정적 단서는 서로 다른 구역에 배치합니다.\n"
                "3. 증거 desc는 추리에 필요한 사실을 보존한 "
                "짧은 1~2문장으로 작성합니다.\n"
                "4. briefing은 기존대로 3~5문장을 유지합니다.\n"
                "5. 용의자별 맞춤 질문은 4개이며 "
                "관계·알리바이·감정을 모두 포함합니다.\n"
                "6. speech_style의 기존 항목은 모두 유지하되 "
                "각 설명을 짧게 작성하고 예시 대사는 짧은 2개로 합니다.\n"
                "7. 이미지 프롬프트, 음성 데이터, 장식용 긴 문서는 "
                "이번 사건 JSON에 추가하지 않습니다.\n"
                "8. 필수 필드를 생략하거나 증거의 사실관계를 "
                "나중에 결정하는 방식으로 작성하지 않습니다.\n"
                "9. 기존 난이도·카테고리·정답 보호·검사 규칙을 "
                "그대로 지킵니다."
            ),
        })
    attempts = []
    backup = None   # 규칙은 통과했지만 검토 지적이 남은 가장 최근 사건 (예비용)

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            content = _call_ai(messages, temperature=0.3)
            case = to_server_truth(_extract_json(content))
            case["difficulty"] = difficulty

            if category is not None:
                case["category_id"] = category_id
                case["category_name"] = category["name"]

            errors = validate_generated_case(case)
            if not errors:
                # 형식이 맞으면 내용 품질 검사 (AI 호출 없음). [경고]는 재생성 사유에서 제외
                errors, _warnings = split_warnings(check_case_quality(case, expected_culprit_id=culprit_id))
        except CaseRequestStopped as e:
            attempts.append({
                "attempt": attempt,
                "errors": [str(e)],
                "review": "외부 요청 중단으로 건너뜀",
            })

            # 서비스 오류나 필터 제한을 사건 내용 오류처럼
            # 반복 요청하지 않습니다.
            return {
                "success": False,
                "case": None,
                "file": None,
                "attempts": attempts,
            }

        except Exception as e:
            content = ""
            case = None
            errors = [f"AI 응답 처리 실패: {e}"]

        # 규칙 검사를 통과한 사건만 검토 AI에게 보냄
        review = "규칙 검사 실패로 건너뜀"
        if not errors:
            try:
                problems = review_case(case)

            except CaseRequestStopped as e:
                attempts.append({
                    "attempt": attempt,
                    "errors": [str(e)],
                    "review": "검토 요청 중단",
                })

                return {
                    "success": False,
                    "case": None,
                    "file": None,
                    "attempts": attempts,
                }
            if problems is None:
                review = "검토 AI 호출 실패로 건너뜀"
            elif not problems:
                review = "통과"
            else:
                review = f"문제 {len(problems)}개"
                backup = (case, problems)
                errors = [f"[AI 검토] {p}" for p in problems]

        attempts.append({"attempt": attempt, "errors": errors, "review": review})

        if not errors:
            file_path = save_case(case)
            return {"success": True, "case": case, "file": file_path.name, "attempts": attempts}

        # 검사기·검토 AI가 찾은 문제를 생성 AI에게 알려 주고 고치게 함
        if content:
            messages.append({"role": "assistant", "content": content})
        messages.append({
            "role": "user",
            "content": "검사에서 아래 문제가 나왔습니다. 모두 고쳐서 전체 JSON을 다시 보내 주십시오.\n- "
                       + "\n- ".join(errors),
        })

    # 끝까지 완벽한 사건이 없으면, 규칙은 지킨 예비 사건을 검토 지적과 함께 저장
    if backup:
        case, problems = backup
        case["review_notes"] = problems
        file_path = save_case(case)
        return {"success": True, "case": case, "file": file_path.name, "attempts": attempts}

    return {"success": False, "case": None, "file": None, "attempts": attempts}


def save_case(case: dict) -> Path:
    CASES_DIR.mkdir(exist_ok=True)
    stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
    file_path = CASES_DIR / f"case_{stamp}.json"
    number = 2
    # 같은 초에 저장해도 덮어쓰지 않도록 (old 폴더로 옮긴 파일 이름과도 겹치지 않게)
    while file_path.exists() or (OLD_CASES_DIR / file_path.name).exists():
        file_path = CASES_DIR / f"case_{stamp}_{number:02d}.json"
        number += 1
    file_path.write_text(json.dumps(case, ensure_ascii=False, indent=2), encoding="utf-8")
    archive_old_cases()
    return file_path


def archive_old_cases() -> list[str]:
    """사건이 MAX_ACTIVE_CASES개를 넘으면 오래된 것부터 cases/old 폴더로 옮깁니다. (삭제하지 않음)"""
    files = list_case_files()   # 파일 이름의 날짜·시각 순서 = 오래된 순서
    moved = []
    if len(files) <= MAX_ACTIVE_CASES:
        return moved

    OLD_CASES_DIR.mkdir(exist_ok=True)
    for file_path in files[:len(files) - MAX_ACTIVE_CASES]:
        target = OLD_CASES_DIR / file_path.name
        number = 2
        while target.exists():   # old 폴더의 파일을 덮어쓰지 않음
            target = OLD_CASES_DIR / f"{file_path.stem}_old{number}.json"
            number += 1
        shutil.move(str(file_path), str(target))
        moved.append(file_path.name)
    return moved


def mark_played(file_path: Path) -> None:
    """게임에 한 번 쓰인 사건에 '플레이함' 표시를 남깁니다."""
    try:
        case = json.loads(file_path.read_text(encoding="utf-8"))
        case["played"] = True
        file_path.write_text(json.dumps(case, ensure_ascii=False, indent=2), encoding="utf-8")
    except (OSError, ValueError):
        pass   # 표시를 못 남겨도 게임 진행에는 문제 없음


def load_case(file_path: Path) -> dict:
    """저장된 사건을 읽습니다. JSON은 키가 문자열이라 용의자 ID를 숫자로 되돌립니다."""
    case = json.loads(file_path.read_text(encoding="utf-8"))
    case["suspects"] = {int(s_id): s for s_id, s in case["suspects"].items()}
    return case


def list_case_files() -> list[Path]:
    if not CASES_DIR.exists():
        return []
    return sorted(CASES_DIR.glob("case_*.json"))


# 터미널에서 직접 실행하면 사건을 하나 만들어 결과를 보여 줍니다.
#   python case_generator.py
if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    print(f"사건 생성 중... (모델: {CASE_MODEL_NAME})")
    result = generate_case()

    for a in result["attempts"]:
        status = "통과" if not a["errors"] else f"문제 {len(a['errors'])}개"
        print(f"  시도 {a['attempt']}: {status} (AI 검토: {a['review']})")
        for error in a["errors"]:
            print(f"    - {error}")

    if result["success"]:
        case = result["case"]
        print(f"\n✅ 저장됨: cases/{result['file']}")
        for note in case.get("review_notes", []):
            print(f"  ⚠ 남은 검토 지적: {note}")
        print(f"제목: {case['title']}")
        print(f"개요: {case['case_summary']}")
        for s_id, s in case["suspects"].items():
            print(f"  용의자 {s_id}: {s['name']} ({s['job']})")
        print(f"단서 {len(case['evidence_db'])}개")
    else:
        print("\n❌ 사건 생성 실패 (서버는 기본 호텔 사건을 사용합니다)")
