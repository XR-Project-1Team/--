import hashlib
import io
import os
import threading
import wave
import httpx
from collections import OrderedDict

from fastapi import HTTPException, Request, Response
from openai import OpenAI, APIStatusError
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool


class SpeechRequest(BaseModel):
    game_id: str
    suspect_id: int
    dialogue: str = Field(min_length=1, max_length=4000)


# 로컬 개발용 음성 캐시입니다.
# 녹음한 사용자 음성은 여기에 저장하지 않습니다.
_speech_cache = OrderedDict()
_cache_lock = threading.Lock()
_tts_lock = threading.Lock()

MAX_CACHED_SPEECH = 30
MAX_RECORDING_BYTES = 2 * 1024 * 1024


def register_voice_routes(app, active_games):
    def get_session(game_id):
        session = active_games.get(game_id)

        if session is None:
            raise HTTPException(404, "진행 중인 게임을 찾지 못했습니다.")

        if session.get("accuse", {}).get("finished"):
            raise HTTPException(409, "이미 종료된 게임입니다.")

        return session

    def new_client():
        api_key = os.getenv("VOICE_API_KEY", "").strip()
        base_url = os.getenv("VOICE_BASE_URL", "").strip()

        if not api_key or not base_url:
            raise HTTPException(
                503,
                ".env의 VOICE_API_KEY와 VOICE_BASE_URL을 확인하세요.",
            )

        return OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=90,
            max_retries=0,
        )

    def upstream_error(error):
        code = getattr(error, "status_code", None)

        print(
            "[음성 API 실패]",
            type(error).__name__,
            "HTTP",
            code,
            flush=True,
        )

        return HTTPException(
            502,
            (
                f"외부 음성 서비스 오류: HTTP {code}"
                if code is not None
                else "외부 음성 서비스 연결에 실패했습니다."
            ),
        )

    def transcribe_bytes(audio_bytes):
        try:
            with new_client() as client:
                result = client.audio.transcriptions.create(
                    model=os.getenv(
                        "STT_MODEL",
                        "gpt-4o-mini-transcribe",
                    ),
                    file=("question.wav", audio_bytes, "audio/wav"),
                    language="ko",
                    response_format="json",
                )

            return (result.text or "").strip()

        except HTTPException:
            raise

        except Exception as error:
            raise upstream_error(error)

    @app.post("/game/voice/transcribe")
    async def transcribe_question(
        request: Request,
        game_id: str,
        suspect_id: int,
    ):
        session = get_session(game_id)

        if suspect_id not in session["truth"]["suspects"]:
            raise HTTPException(404, "용의자를 찾지 못했습니다.")

        # Unity가 보낸 WAV 데이터를 제한된 크기로 읽습니다.
        chunks = []
        size = 0

        async for chunk in request.stream():
            size += len(chunk)

            if size > MAX_RECORDING_BYTES:
                raise HTTPException(413, "녹음 파일이 너무 큽니다.")

            chunks.append(chunk)

        audio_bytes = b"".join(chunks)

        try:
            with wave.open(io.BytesIO(audio_bytes), "rb") as audio:
                duration = (
                    audio.getnframes() / audio.getframerate()
                )

                if (
                    audio.getcomptype() != "NONE"
                    or audio.getnchannels() != 1
                    or audio.getsampwidth() != 2
                    or duration < 0.3
                    or duration > 21
                ):
                    raise ValueError("지원하지 않는 녹음 형식")

        except Exception:
            raise HTTPException(
                400,
                "0.3~20초 정도의 모노 PCM WAV 녹음이 필요합니다.",
            )

        text = await run_in_threadpool(
            transcribe_bytes,
            audio_bytes,
        )

        # 인식 중 게임이 종료되었다면 결과를 진행 중 게임에 적용하지 않습니다.
        get_session(game_id)

        return {"text": text}

    @app.post("/game/voice/speech")
    def speak_dialogue(req: SpeechRequest):
        session = get_session(req.game_id)

        suspect = session["truth"]["suspects"].get(req.suspect_id)

        if suspect is None:
            raise HTTPException(
                404,
                "용의자를 찾지 못했습니다.",
            )

        # 기존 보안 조건을 유지합니다.
        # 서버가 승인한 현재 답변만 음성으로 읽습니다.
        if req.dialogue != suspect.get("voice_dialogue"):
            raise HTTPException(
                409,
                "현재 승인된 용의자 답변과 음성 요청이 다릅니다.",
            )

        local_url = os.getenv(
            "LOCAL_TTS_BASE_URL",
            "http://127.0.0.1:8001",
        ).strip().rstrip("/")

        speaker = os.getenv(
            f"QWEN_SPEAKER_{req.suspect_id}",
            "Sohee",
        ).strip()

        emotion = suspect.get("voice_emotion", "calm")

        if emotion not in ("calm", "nervous", "angry", "broken"):
            emotion = "calm"

        read_timeout = float(
            os.getenv("LOCAL_TTS_TIMEOUT_SECONDS", "600")
        )

        payload = {
            "text": req.dialogue,
            "speaker": speaker,
            "emotion": emotion,
        }

        try:
            # 로컬 접속에 시스템 프록시가 개입하지 않게 합니다.
            # transport 재시도는 추가하지 않습니다.
            with httpx.Client(
                timeout=httpx.Timeout(
                    read_timeout,
                    connect=5.0,
                ),
                trust_env=False,
            ) as local_client:
                response = local_client.post(
                    local_url + "/tts",
                    json=payload,
                )

        except httpx.ConnectError as error:
            print(
                "[로컬 음성 실패] Qwen 서버에 연결할 수 없습니다.",
                flush=True,
            )

            raise HTTPException(
                503,
                "로컬 Qwen 음성 서버가 꺼져 있거나 접속할 수 없습니다.",
            ) from error

        except httpx.TimeoutException as error:
            print(
                "[로컬 음성 실패] 응답 대기시간 초과",
                flush=True,
            )

            raise HTTPException(
                504,
                "로컬 음성 생성 대기시간이 초과되었습니다. "
                "Qwen 서버에서는 생성이 계속될 수 있습니다.",
            ) from error

        except httpx.RequestError as error:
            print(
                "[로컬 음성 실패]",
                type(error).__name__,
                flush=True,
            )

            raise HTTPException(
                502,
                "로컬 음성 서버와 통신하지 못했습니다.",
            ) from error

        if response.status_code != 200:
            print(
                "[로컬 음성 실패] Qwen HTTP",
                response.status_code,
                flush=True,
            )

            if response.status_code == 429:
                raise HTTPException(
                    503,
                    "음성 생성 요청이 밀려 있습니다. "
                    "이번 답변은 자막으로 확인하세요.",
                )

            if response.status_code == 503:
                raise HTTPException(
                    503,
                    "로컬 음성 서버의 GPU 상태를 확인하세요.",
                )

            raise HTTPException(
                502,
                "로컬 음성 생성에 실패했습니다. "
                f"Qwen 서버 응답: HTTP {response.status_code}",
            )

        audio_bytes = response.content

        if not (
            audio_bytes.startswith(b"RIFF")
            and audio_bytes[8:12] == b"WAVE"
        ):
            raise HTTPException(
                502,
                "로컬 음성 서버에서 올바른 WAV를 받지 못했습니다.",
            )

        # 생성하는 동안 게임이 종료됐거나
        # 다음 답변으로 바뀌었다면 이전 음성을 사용하지 않습니다.
        current_session = get_session(req.game_id)

        current_suspect = current_session["truth"]["suspects"].get(
            req.suspect_id
        )

        if (
            current_suspect is None
            or current_suspect.get("voice_dialogue") != req.dialogue
        ):
            raise HTTPException(
                409,
                "게임 상태 또는 용의자의 답변이 바뀌었습니다.",
            )

        print(
            "[로컬 음성 전달 완료] 용의자 ID:",
            req.suspect_id,
            flush=True,
        )

        return Response(
            audio_bytes,
            media_type="audio/wav",
            headers={"Cache-Control": "no-store"},
        )