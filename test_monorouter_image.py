import base64
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen

from dotenv import dotenv_values
from openai import OpenAI, APIStatusError, APITimeoutError


def main():
    folder = Path(__file__).resolve().parent
    settings = dotenv_values(folder / ".env")

    api_key = (settings.get("IMAGE_API_KEY") or "").strip()
    base_url = (settings.get("IMAGE_BASE_URL") or "").strip()
    model = (settings.get("IMAGE_MODEL") or "").strip()

    if not api_key:
        print("설정 오류: IMAGE_API_KEY가 비어 있습니다.")
        return

    if not base_url:
        print("설정 오류: IMAGE_BASE_URL이 비어 있습니다.")
        return

    if not model:
        print("설정 오류: IMAGE_MODEL이 비어 있습니다.")
        return

    print("이미지 연결 주소:", base_url)
    print("이미지 모델:", model)
    print("API 키: 설정됨 — 표시하지 않음")
    print()
    print("이 테스트는 크레딧을 사용하는 이미지 생성 요청입니다.")

    answer = input(
        "한 장을 생성하려면 '생성'이라고 입력하세요: "
    ).strip()

    if answer != "생성":
        print("취소했습니다. 이미지 API를 호출하지 않았습니다.")
        return

    output_folder = folder / "image_test_output"

    try:
        output_folder.mkdir(parents=True, exist_ok=True)
    except OSError:
        print("저장 폴더를 만들 수 없습니다. 폴더 권한을 확인하세요.")
        return

    # 실제 사건의 증거가 아닌 연결 확인용 가상 소품입니다.
    prompt = (
        "추리 게임에서 사용할 가상의 증거물 재현 이미지. "
        "짙은 회색 책상 위에 놓인 작은 직사각형 금속 문진 하나. "
        "옆에서 비추는 부드러운 조명, 사실적인 재질, 정면에 가까운 구도. "
        "문진 이외의 물건은 넣지 마세요. "
        "글자, 숫자, 이름, 로고, 혈흔, 사람은 넣지 마세요."
    )

    generation_returned = False

    try:
        print("이미지를 생성하는 중입니다. 다시 실행하지 말고 기다려 주세요.")

        with OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=180,
            max_retries=0,
        ) as client:
            result = client.images.generate(
                model=model,
                prompt=prompt,
                size="1024x1024",
            )

        generation_returned = True

        if not result.data:
            print("응답은 받았지만 data에 이미지가 없습니다.")
            print("추가 생성 전에 응답 규격과 사용 통계를 확인하세요.")
            return

        image = result.data[0]
        encoded = getattr(image, "b64_json", None)
        image_url = getattr(image, "url", None)

        if encoded:
            # 이미지 데이터 자체가 반환된 경우
            if encoded.startswith("data:"):
                encoded = encoded.split(",", 1)[1]

            image_bytes = base64.b64decode(
                encoded,
                validate=True,
            )

        elif image_url:
            # 다운로드 주소가 반환된 경우
            parsed = urlparse(image_url)

            if parsed.scheme != "https" or not parsed.netloc:
                print("이미지 주소가 HTTPS가 아니므로 다운로드하지 않았습니다.")
                return

            print("생성 완료. 이미지 파일을 다운로드하는 중입니다.")

            # 다운로드 주소에는 MonoRouter API 키를 전달하지 않습니다.
            with urlopen(image_url, timeout=60) as response:
                image_bytes = response.read()

        else:
            print("응답은 받았지만 b64_json 또는 url이 없습니다.")

            if hasattr(image, "model_dump"):
                print(
                    "이미지 응답의 필드 이름:",
                    list(image.model_dump().keys()),
                )

            print("응답 형식을 확인해야 합니다. 다시 생성하지 마세요.")
            return

        if not image_bytes:
            print("이미지 데이터가 비어 있습니다.")
            return

        # 실제 파일 내용을 보고 확장자를 정합니다.
        if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
            extension = ".png"

        elif image_bytes.startswith(b"\xff\xd8\xff"):
            extension = ".jpg"

        elif (
            image_bytes.startswith(b"RIFF")
            and image_bytes[8:12] == b"WEBP"
        ):
            extension = ".webp"

        else:
            # 알 수 없는 형식도 데이터는 보존합니다.
            extension = ".bin"

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output_path = output_folder / (
            "evidence_test_" + stamp + extension
        )

        output_path.write_bytes(image_bytes)

        if extension == ".bin":
            print("응답 데이터를 저장했지만 이미지 형식 확인이 필요합니다.")
        else:
            print("이미지 생성 및 저장 완료!")

        print("저장 위치:", output_path)

    except APIStatusError as error:
        print("이미지 요청 실패. HTTP 상태 코드:", error.status_code)

        try:
            body = error.response.json()
        except ValueError:
            body = {}

        if isinstance(body, dict):
            detail = body.get("error", body)

            if isinstance(detail, dict):
                for field in (
                    "type",
                    "code",
                    "user_message",
                    "message",
                ):
                    value = detail.get(field)

                    if value is not None:
                        safe_text = str(value)

                        # 설정된 키가 오류 설명에 포함되면 가립니다.
                        for name in ("IMAGE_API_KEY", "LLM_API_KEY"):
                            secret = settings.get(name)

                            if secret:
                                safe_text = safe_text.replace(
                                    secret,
                                    "[키 숨김]",
                                )

                        print(f"{field}: {safe_text[:1000]}")

    except APITimeoutError:
        print("이미지 응답 대기시간이 초과되었습니다.")
        print(
            "서비스에서는 생성이 진행됐을 수 있으므로 "
            "바로 재실행하지 말고 MonoRouter 사용 통계를 확인하세요."
        )

    except Exception as error:
        print("오류 종류:", type(error).__name__)

        if generation_returned:
            print(
                "이미지 API 응답 이후 처리·다운로드·저장 단계에서 "
                "문제가 발생했습니다."
            )
        else:
            print(
                "요청 또는 응답 처리 중 문제가 발생했습니다."
            )

        print(
            "자동 재요청하지 않았습니다. "
            "추가 생성 전에 사용 통계와 오류 내용을 확인하세요."
        )


if __name__ == "__main__":
    main()