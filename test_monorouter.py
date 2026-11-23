from pathlib import Path

from dotenv import dotenv_values
from openai import OpenAI, APIStatusError


def main():
    # 이 테스트 파일과 같은 폴더의 .env를 직접 읽습니다.
    env_path = Path(__file__).resolve().parent / ".env"
    settings = dotenv_values(env_path)

    api_key = (settings.get("LLM_API_KEY") or "").strip()
    base_url = (settings.get("LLM_BASE_URL") or "").strip()
    model = (settings.get("LLM_MODEL") or "").strip()

    if not api_key:
        print("설정 오류: .env의 LLM_API_KEY가 비어 있습니다.")
        return

    if not base_url:
        print("설정 오류: .env의 LLM_BASE_URL이 비어 있습니다.")
        return

    if model != "gpt-6.1-sol":
        print("설정 오류: LLM_MODEL을 gpt-6.1-sol로 설정하세요.")
        print("현재 모델:", model or "(비어 있음)")
        return

    print("읽은 설정 파일:", env_path)
    print("연결 주소:", base_url)
    print("요청 모델:", model)
    print("API 키:", "설정됨 — 보안상 표시하지 않음")

    try:
        with OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=60,
            max_retries=0,
        ) as client:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": "한국어로 '연결 확인 완료'라고만 답하세요.",
                    }
                ],
                max_completion_tokens=1024,
            )

        if not response.choices:
            print("응답은 받았지만 choices가 없습니다.")
            return

        choice = response.choices[0]
        content = choice.message.content or ""

        print("종료 이유:", choice.finish_reason)

        if content.strip():
            print("답변:", content)
        else:
            print("응답은 받았지만 텍스트가 비어 있습니다.")

    except APIStatusError as error:
        print("HTTP 상태 코드:", error.status_code)

        try:
            body = error.response.json()
        except ValueError:
            body = {}

        if not isinstance(body, dict):
            body = {}

        detail = body.get("error", body)

        if not isinstance(detail, dict):
            detail = {}

        found_message = False

        for field in ("type", "code", "user_message", "message"):
            value = detail.get(field)

            if value is None:
                continue

            # 오류 문구에 키가 포함되는 경우를 대비해 가립니다.
            safe_text = str(value).replace(api_key, "[키 숨김]")

            print(f"{field}: {safe_text[:1500]}")
            found_message = True

        if not found_message:
            print(
                "상세 오류 문구가 없습니다. "
                "MonoRouter 사용 통계에서 해당 요청을 확인하세요."
            )

    except Exception as error:
        print("연결 오류 종류:", type(error).__name__)
        print("인터넷 연결과 서버 주소, 설치된 라이브러리를 확인하세요.")


if __name__ == "__main__":
    main()