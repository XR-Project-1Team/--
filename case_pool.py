import argparse
import copy
import hashlib
import json
import random
import re
import threading
import uuid
from pathlib import Path

from case_generator import CASE_CATEGORIES, load_case
from validator import validate_generated_case
from case_quality import check_case_quality, split_warnings


ROOT = Path(__file__).resolve().parent
POOL_DIR = ROOT / "case_pool_data"
INDEX_FILE = POOL_DIR / "index.json"

_lock = threading.RLock()

RUNTIME_FIELDS = (
    "confessed",
    "presented_evidence",
    "asked_questions",
    "answer_cache",
    "chat_log",
    "statements",
    "voice_dialogue",
    "voice_emotion",
)


def _read_index():
    if not INDEX_FILE.exists():
        return {"entries": {}, "offered": {}}

    data = json.loads(INDEX_FILE.read_text(encoding="utf-8"))

    if not isinstance(data, dict):
        raise ValueError("사건 보관함 인덱스 형식이 잘못됐습니다.")

    data.setdefault("entries", {})
    data.setdefault("offered", {})
    return data


def _write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary = path.with_name(
        path.name + "." + uuid.uuid4().hex + ".tmp"
    )

    temporary.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    temporary.replace(path)


def _clean_case(case):
    result = copy.deepcopy(case)
    result.pop("played", None)

    for suspect in result.get("suspects", {}).values():
        suspect["pressure_level"] = 0

        for field in RUNTIME_FIELDS:
            suspect.pop(field, None)

    return result


def _check_case(case):
    errors = validate_generated_case(case)

    if errors:
        raise ValueError(" / ".join(errors))

    hard_errors, warnings = split_warnings(
        check_case_quality(case)
    )

    if hard_errors:
        raise ValueError(" / ".join(hard_errors))

    if not str(case.get("truth", "")).strip():
        raise ValueError("사건의 비공개 진실이 없습니다.")

    if case.get("review_notes"):
        raise ValueError("해결되지 않은 AI 검토 지적이 있습니다.")

    return warnings


def put_case(case, source="", approved=False):
    clean = _clean_case(case)
    _check_case(clean)

    serialized = json.dumps(
        clean,
        ensure_ascii=False,
        sort_keys=True,
    )

    case_id = hashlib.sha256(
        serialized.encode("utf-8")
    ).hexdigest()[:24]

    with _lock:
        index = _read_index()
        path = POOL_DIR / f"pool_{case_id}.json"

        if not path.exists():
            _write_json(path, clean)

        if case_id not in index["entries"]:
            index["entries"][case_id] = {
                "title": clean.get("title", ""),
                "source": source,
                "category_id": clean.get("category_id", ""),
                "difficulty": clean.get("difficulty", ""),
                "approved": bool(approved),
            }

        elif approved:
            entry = index["entries"][case_id]
            entry["approved"] = True
            entry["category_id"] = clean.get("category_id", "")
            entry["difficulty"] = clean.get("difficulty", "")

        _write_json(INDEX_FILE, index)

    return case_id


def _load_entry(case_id, entry):
    path = POOL_DIR / f"pool_{case_id}.json"

    if not path.is_file():
        raise ValueError("보관된 사건 파일이 없습니다.")

    case = load_case(path)

    # 관리자가 확인한 분류를 이번 세션에 반영합니다.
    case["category_id"] = entry["category_id"]
    case["category_name"] = CASE_CATEGORIES[
        entry["category_id"]
    ]["name"]

    case["difficulty"] = entry["difficulty"]

    _check_case(case)
    return case


def load_pool_case(file_name):
    match = re.fullmatch(
        r"pool_([0-9a-f]{24})\.json",
        file_name or "",
    )

    if not match:
        raise ValueError("잘못된 보관 사건 파일 이름입니다.")

    case_id = match.group(1)

    with _lock:
        index = _read_index()
        entry = index["entries"].get(case_id)

        if not entry or not entry.get("approved"):
            raise ValueError("승인된 보관 사건이 아닙니다.")

        return _load_entry(case_id, entry)


def public_catalog():
    with _lock:
        index = _read_index()

        return [
            {
                "case_id": case_id,
                "title": entry.get("title", ""),
                "category_id": entry.get("category_id", ""),
                "difficulty": entry.get("difficulty", ""),
                "approved": bool(entry.get("approved")),
            }
            for case_id, entry in index["entries"].items()
        ]


def ready_count(category_id, difficulty):
    count = 0

    with _lock:
        index = _read_index()

        for case_id, entry in index["entries"].items():
            if (
                not entry.get("approved")
                or entry.get("category_id") != category_id
                or entry.get("difficulty") != difficulty
            ):
                continue

            try:
                _load_entry(case_id, entry)
                count += 1
            except Exception:
                continue

    return count


def take_case(category_id, difficulty, player_id):
    with _lock:
        index = _read_index()
        candidates = []

        for case_id, entry in index["entries"].items():
            if (
                not entry.get("approved")
                or entry.get("category_id") != category_id
                or entry.get("difficulty") != difficulty
            ):
                continue

            try:
                _load_entry(case_id, entry)
            except Exception:
                continue

            candidates.append((case_id, entry))

        if not candidates:
            raise ValueError(
                "선택한 카테고리·난이도의 준비된 사건이 없습니다. "
                "'새 사건 생성'을 선택하거나 보관함에 사건을 등록하세요."
            )

        offered = index["offered"].setdefault(player_id, [])
        fresh = [
            item for item in candidates
            if item[0] not in offered
        ]

        case_id, entry = random.choice(fresh or candidates)
        reused = not bool(fresh)

        # 완료 이력이 아니라 사건 제공 이력입니다.
        offered.append(case_id)
        index["offered"][player_id] = offered[-200:]

        _write_json(INDEX_FILE, index)

        return {
            "file": f"pool_{case_id}.json",
            "title": entry["title"],
            "reused": reused,
        }


def import_existing():
    files = sorted((ROOT / "cases").glob("case_*.json"))
    files += sorted((ROOT / "cases" / "old").glob("case_*.json"))

    for path in files:
        try:
            case_id = put_case(
                load_case(path),
                source=str(path.relative_to(ROOT)),
                approved=False,
            )

            print("가져옴:", case_id, path.name)

        except Exception as error:
            print("제외:", path.name, str(error))

    print("\n검사를 통과한 파일만 복사했습니다.")
    print("이전에 승인한 사건의 승인은 유지됩니다.")
    print("새로 가져온 사건은 내용을 확인한 뒤 approve로 승인하세요.")


def approve(case_id, category_id, difficulty):
    with _lock:
        index = _read_index()
        entry = index["entries"].get(case_id)

        if entry is None:
            raise ValueError("등록된 사건 ID가 아닙니다.")

        for field, wanted in (
            ("category_id", category_id),
            ("difficulty", difficulty),
        ):
            existing = entry.get(field)

            if existing and existing != wanted:
                raise ValueError(
                    f"기존 {field}={existing}와 지정값 {wanted}가 다릅니다. "
                    "실제 사건 내용을 먼저 확인하세요."
                )

        checked_entry = dict(entry)
        checked_entry["category_id"] = category_id
        checked_entry["difficulty"] = difficulty

        _load_entry(case_id, checked_entry)

        checked_entry["approved"] = True
        index["entries"][case_id] = checked_entry

        _write_json(INDEX_FILE, index)

        print("승인 완료:", case_id, category_id, difficulty)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("import")
    sub.add_parser("list")

    approve_parser = sub.add_parser("approve")
    approve_parser.add_argument("case_id")
    approve_parser.add_argument(
        "--category",
        required=True,
        choices=list(CASE_CATEGORIES),
    )
    approve_parser.add_argument(
        "--difficulty",
        required=True,
        choices=["easy", "standard"],
    )

    args = parser.parse_args()

    if args.command == "import":
        import_existing()

    elif args.command == "list":
        print(json.dumps(
            public_catalog(),
            ensure_ascii=False,
            indent=2,
        ))

    elif args.command == "approve":
        approve(
            args.case_id,
            args.category,
            args.difficulty,
        )