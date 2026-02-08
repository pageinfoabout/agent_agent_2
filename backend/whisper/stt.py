
from __future__ import annotations

import asyncio
import base64
import json

import time
import weakref
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode

import aiohttp
import httpx

import openai
from livekit import rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    stt,
    utils,
)
from livekit.agents.types import (
    NOT_GIVEN,
    NotGivenOr,
)
from livekit.agents.utils import AudioBuffer, is_given
from openai.types.audio import TranscriptionVerbose
from openai.types.beta.realtime.transcription_session_update_param import (
    SessionTurnDetection,
)

from .log import logger
from .models import GroqAudioModels, STTModels


# OpenAI Realtime API has a timeout of 15 mins, we'll attempt to restart the session
# before that timeout is reached
_max_session_duration = 10 * 60
# emit interim transcriptions every 0.5 seconds
_delta_transcript_interval = 0.5
SAMPLE_RATE = 24000
NUM_CHANNELS = 1


@dataclass
class _STTOptions:
    model: STTModels | str
    language: str
    detect_language: bool
    turn_detection: SessionTurnDetection
    prompt: NotGivenOr[str] = NOT_GIVEN
    noise_reduction_type: NotGivenOr[str] = NOT_GIVEN


class STT(stt.STT):
    def __init__(
        self,
        *,
        language: str = "ru",
        detect_language: bool = False,
        
        prompt: NotGivenOr[str] = NOT_GIVEN,
        turn_detection: NotGivenOr[SessionTurnDetection] = NOT_GIVEN,
        noise_reduction_type: NotGivenOr[str] = NOT_GIVEN,
   
     
        use_realtime: bool = False,
        model: GroqAudioModels | str = "whisper-large-v3-turbo",
        api_key: NotGivenOr[str] = NOT_GIVEN,
        base_url: str  = 'https:localhost:8000',
        client: httpx.AsyncClient | None = None,    
   
    ):
    

        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=use_realtime, interim_results=use_realtime, aligned_transcript=False
            )
        )
        if detect_language:
            language = ""

        if not is_given(turn_detection):
            turn_detection = {
                "type": "server_vad",
                "threshold": 0.5,
                "prefix_padding_ms": 600,
                "silence_duration_ms": 350,
            }

        self._opts = _STTOptions(
            language=language,
            detect_language=detect_language,
            model=model,
            prompt=prompt,
            turn_detection=turn_detection,
        )
        if is_given(noise_reduction_type):
            self._opts.noise_reduction_type = noise_reduction_type

        self._client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(60.0, connect=15.0),
            limits=httpx.Limits(max_keepalive_connections=20)
        )

        self._streams = weakref.WeakSet[SpeechStream]()
        self._session: aiohttp.ClientSession | None = None
        self._pool = utils.ConnectionPool[aiohttp.ClientWebSocketResponse](
            max_session_duration=_max_session_duration,
            connect_cb=self._connect_ws,
            close_cb=self._close_ws,
        )

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return self._client._base_url.netloc.decode("utf-8")

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SpeechStream:
        if is_given(language):
            self._opts.language = language
        stream = SpeechStream(
            stt=self,
            pool=self._pool,
            conn_options=conn_options,
        )
        self._streams.add(stream)
        return stream

    def update_options(
        self,
        *,
        model: NotGivenOr[STTModels | GroqAudioModels | str] = NOT_GIVEN,
        language: NotGivenOr[str] = NOT_GIVEN,
        detect_language: NotGivenOr[bool] = NOT_GIVEN,
        prompt: NotGivenOr[str] = NOT_GIVEN,
        turn_detection: NotGivenOr[SessionTurnDetection] = NOT_GIVEN,
        noise_reduction_type: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        """
        Update the options for the speech stream. Most options are updated at the
        connection level. SpeechStreams will be recreated when options are updated.

        Args:
            language: The language to transcribe in.
            detect_language: Whether to automatically detect the language.
            model: The model to use for transcription.
            prompt: Optional text prompt to guide the transcription. Only supported for whisper-1.
            turn_detection: When using realtime, this controls how model detects the user is done speaking.
            noise_reduction_type: Type of noise reduction to apply. "near_field" or "far_field"
        """  # noqa: E501
        if is_given(model):
            self._opts.model = model
        if is_given(language):
            self._opts.language = language
        if is_given(detect_language):
            self._opts.detect_language = detect_language
            self._opts.language = ""
        if is_given(prompt):
            self._opts.prompt = prompt
        if is_given(turn_detection):
            self._opts.turn_detection = turn_detection
        if is_given(noise_reduction_type):
            self._opts.noise_reduction_type = noise_reduction_type

        for stream in self._streams:
            if is_given(language):
                stream.update_options(language=language)

    async def _connect_ws(self, timeout: float) -> aiohttp.ClientWebSocketResponse:
        prompt = self._opts.prompt if is_given(self._opts.prompt) else ""
        transcription_config: dict[str, Any] = {
            "model": self._opts.model,
        }
        if prompt:
            transcription_config["prompt"] = prompt
        if self._opts.language:
            transcription_config["language"] = self._opts.language

        input_config: dict[str, Any] = {
            "format": {
                "type": "audio/pcm",
                "rate": SAMPLE_RATE,
            },
            "transcription": transcription_config,
            "turn_detection": self._opts.turn_detection,
        }

        if self._opts.noise_reduction_type:
            input_config["noise_reduction"] = {"type": self._opts.noise_reduction_type}

        realtime_config: dict[str, Any] = {
            "type": "session.update",
            "session": {
                "type": "transcription",
                "audio": {
                    "input": input_config,
                },
            },
        }

        query_params: dict[str, str] = {
            "intent": "transcription",
        }
        headers = {
            "User-Agent": "LiveKit Agents",
            "Authorization":  "Bearer dummy",
        }
        url = "http://localhost:5000"
        session = self._ensure_session()
        ws = await asyncio.wait_for(session.ws_connect(url, headers=headers), timeout)
        await ws.send_json(realtime_config)
        return ws

    async def _close_ws(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        await ws.close()

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()

        return self._session

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        try:
            if is_given(language):
                self._opts.language = language
            data = rtc.combine_audio_frames(buffer).to_wav_bytes()
            prompt = self._opts.prompt if is_given(self._opts.prompt) else openai.omit

            format = "json"
            if self._opts.model == "whisper-1":
                # verbose_json returns language and other details, only supported for whisper-1
                format = "verbose_json"

            resp = await self._client.audio.transcriptions.create(
                file=(
                    "file.wav",
                    data,
                    "audio/wav",
                ),
                model=self._opts.model,  # type: ignore
                language=self._opts.language,
                prompt=prompt,
                response_format=format,
                timeout=httpx.Timeout(30, connect=conn_options.timeout),
            )

            sd = stt.SpeechData(text=resp.text, language=self._opts.language)
            if isinstance(resp, TranscriptionVerbose) and resp.language:
                sd.language = resp.language

            return stt.SpeechEvent(
                type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                alternatives=[sd],
            )

        except openai.APITimeoutError:
            raise APITimeoutError() from None
        except openai.APIStatusError as e:
            raise APIStatusError(
                e.message, status_code=e.status_code, request_id=e.request_id, body=e.body
            ) from None
        except Exception as e:
            raise APIConnectionError() from e


class SpeechStream(stt.SpeechStream):
    def __init__(
        self,
        *,
        stt: STT,
        conn_options: APIConnectOptions,
        pool: utils.ConnectionPool[aiohttp.ClientWebSocketResponse],
    ) -> None:
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=SAMPLE_RATE)

        self._pool = pool
        self._language = stt._opts.language
        self._request_id = ""
        self._reconnect_event = asyncio.Event()

    def update_options(
        self,
        *,
        language: str,
    ) -> None:
        self._language = language
        self._pool.invalidate()
        self._reconnect_event.set()

    @utils.log_exceptions(logger=logger)
    async def _run(self) -> None:
        closing_ws = False

        @utils.log_exceptions(logger=logger)
        async def send_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            nonlocal closing_ws

            # forward audio to OAI in chunks of 50ms
            audio_bstream = utils.audio.AudioByteStream(
                sample_rate=SAMPLE_RATE,
                num_channels=NUM_CHANNELS,
                samples_per_channel=SAMPLE_RATE // 20,
            )

            async for data in self._input_ch:
                frames: list[rtc.AudioFrame] = []
                if isinstance(data, rtc.AudioFrame):
                    frames.extend(audio_bstream.write(data.data.tobytes()))
                elif isinstance(data, self._FlushSentinel):
                    frames.extend(audio_bstream.flush())

                for frame in frames:
                    encoded_frame = {
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(frame.data.tobytes()).decode("utf-8"),
                    }
                    await ws.send_json(encoded_frame)

            closing_ws = True

        @utils.log_exceptions(logger=logger)
        async def recv_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            nonlocal closing_ws
            current_text = ""
            last_interim_at: float = 0
            connected_at = time.time()
            item_audio_timing: dict[str, dict[str, int]] = {}
            while True:
                msg = await ws.receive()
                if msg.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    if closing_ws:  # close is expected, see SpeechStream.aclose
                        return

                    # this will trigger a reconnection, see the _run loop
                    raise APIStatusError(
                        message="OpenAI Realtime STT connection closed unexpectedly",
                        status_code=ws.close_code or -1,
                        body=f"{msg.data=} {msg.extra=}",
                    )

                if msg.type != aiohttp.WSMsgType.TEXT:
                    logger.warning("unexpected OpenAI message type %s", msg.type)
                    continue

                try:
                    data = json.loads(msg.data)
                    msg_type = data.get("type")
                    if msg_type == "input_audio_buffer.speech_started":
                        item_id = data.get("item_id", "")
                        audio_start_ms = data.get("audio_start_ms", 0)
                        item_audio_timing[item_id] = {"start_ms": audio_start_ms}

                    elif msg_type == "input_audio_buffer.speech_stopped":
                        item_id = data.get("item_id", "")
                        audio_end_ms = data.get("audio_end_ms", 0)
                        if item_id in item_audio_timing:
                            item_audio_timing[item_id]["end_ms"] = audio_end_ms

                    elif msg_type == "conversation.item.input_audio_transcription.delta":
                        delta = data.get("delta", "")
                        if delta:
                            current_text += delta
                            if time.time() - last_interim_at > _delta_transcript_interval:
                                self._event_ch.send_nowait(
                                    stt.SpeechEvent(
                                        type=stt.SpeechEventType.INTERIM_TRANSCRIPT,
                                        alternatives=[
                                            stt.SpeechData(
                                                text=current_text,
                                                language=self._language,
                                            )
                                        ],
                                    )
                                )
                                last_interim_at = time.time()

                    elif msg_type == "conversation.item.input_audio_transcription.completed":
                        current_text = ""
                        transcript = data.get("transcript", "")
                        item_id = data.get("item_id", "")

                        if transcript:
                            self._event_ch.send_nowait(
                                stt.SpeechEvent(
                                    type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                                    alternatives=[
                                        stt.SpeechData(
                                            text=transcript,
                                            language=self._language,
                                        )
                                    ],
                                )
                            )

                        audio_duration = 0.0
                        if item_id in item_audio_timing:
                            timing = item_audio_timing[item_id]
                            start_ms = timing.get("start_ms", 0)
                            end_ms = timing.get("end_ms", 0)
                            if end_ms > start_ms:
                                audio_duration = (end_ms - start_ms) / 1000.0
                            del item_audio_timing[item_id]

                        self._event_ch.send_nowait(
                            stt.SpeechEvent(
                                type=stt.SpeechEventType.RECOGNITION_USAGE,
                                alternatives=[],
                                recognition_usage=stt.RecognitionUsage(
                                    audio_duration=audio_duration
                                ),
                            )
                        )

                        # restart session if needed
                        if time.time() - connected_at > _max_session_duration:
                            logger.info("resetting Realtime STT session due to timeout")
                            self._pool.remove(ws)
                            self._reconnect_event.set()
                            return

                except Exception:
                    logger.exception("failed to process OpenAI message")

        while True:
            closing_ws = False  # reset the flag
            async with self._pool.connection(timeout=self._conn_options.timeout) as ws:
                tasks = [
                    asyncio.create_task(send_task(ws)),
                    asyncio.create_task(recv_task(ws)),
                ]
                tasks_group = asyncio.gather(*tasks)
                wait_reconnect_task = asyncio.create_task(self._reconnect_event.wait())
                try:
                    done, _ = await asyncio.wait(
                        (tasks_group, wait_reconnect_task),
                        return_when=asyncio.FIRST_COMPLETED,
                    )

                    # propagate exceptions from completed tasks
                    for task in done:
                        if task != wait_reconnect_task:
                            task.result()

                    if wait_reconnect_task not in done:
                        break

                    self._reconnect_event.clear()
                finally:
                    await utils.aio.gracefully_cancel(*tasks, wait_reconnect_task)
                    tasks_group.cancel()
                    tasks_group.exception()  # retrieve the exception