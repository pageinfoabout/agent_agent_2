import asyncio
import websockets
import json
import base64
import numpy as np
from faster_whisper import WhisperModel
import uuid
import warnings
import torch
warnings.filterwarnings("ignore")

print("🔄 Loading Faster-Whisper large-v3-turbo (ANTI-HALLUCINATION)...")
device = "cuda" if torch.cuda.is_available() else "cpu"

# 🛡️ ANTI-HALLUCINATION MODEL
model = WhisperModel(
    "large-v3-turbo", 
    device=device, 
    compute_type="float16" if device == "cuda" else "int8",
    download_root="/tmp/whisper_models"
)

print(f"✅ Faster-Whisper ready on {device} - ЖЕСТКИЙ АНТИ-ШУМ")

clients = {}
BAD_PATTERNS = [
    "субтитры", "dima", "torzok", "семкин", "егорова", "семёнкин",
    "продолжение", "следует", "спасибо за", "переводчик", "редактор",
    "озвучка", "автор", "корректор", "студия", "спонсор"
]

async def stt_handler(websocket):
    client_id = str(uuid.uuid4())[:8]
    clients[client_id] = {
        'ws': websocket, 
        'audio_buffer': [], 
        'last_time': 0,
        'speech_energy': 0.0
    }
    print(f"✅ LIVEKIT CONNECTED: {client_id}")
    
    try:
        await websocket.send(json.dumps({"type": "session.created", "session_id": client_id}))
        await websocket.send(json.dumps({"type": "session.updated", "session_id": client_id}))
        
        async for message in websocket:
            data = json.loads(message)
            if data.get('type') == 'input_audio_buffer.append':
                item_id = data.get('item_id', str(uuid.uuid4())[:8])
                await handle_audio(client_id, item_id, data)
                
    except Exception as e:
        print(f"❌ {client_id}: {e}")
    finally:
        clients.pop(client_id, None)

async def handle_audio(client_id, item_id, data):
    client = clients[client_id]
    
    # Decode PCM → float32
    audio_b64 = data['audio']
    audio_bytes = base64.b64decode(audio_b64)
    audio_np = np.frombuffer(audio_bytes, np.int16).astype(np.float32) / 32768.0
    
    client['audio_buffer'].extend(audio_np)
    now = asyncio.get_event_loop().time()
    
    # ⚡ РЕАЛ-ТАЙМ: 600-1200ms чанки
    buffer_sec = len(client['audio_buffer']) / 24000
    if buffer_sec >= 0.6 and (now - client['last_time'] > 1.0):
        await process_whisper(client_id, item_id)

async def process_whisper(client_id, item_id):
    client = clients[client_id]
    audio_array = np.array(client['audio_buffer'])
    client['audio_buffer'] = []
    client['last_time'] = asyncio.get_event_loop().time()
    
    duration = len(audio_array) / 24000
    
    # 🔥 ЖЕСТКИЙ ФИЛЬТР ШУМА (убивает ТВ/субтитры)
    energy = np.mean(audio_array**2)
    loudness = np.max(np.abs(audio_array))
    speech_frames = np.abs(np.convolve(audio_array, np.ones(512)/512, mode='valid'))
    speech_peak = np.max(speech_frames)
    
    # 🚫 ОТСЕВ ТВ/ШУМА
    if (energy < 0.008 or loudness < 0.12 or speech_peak < 0.15 or 
        duration < 0.4 or duration > 10.0):
        print(f"📺 ШУМ (E:{energy:.4f} L:{loudness:.3f} P:{speech_peak:.3f}) → Skip")
        return
    
    print(f"🔊 Processing {duration:.1f}s (E:{energy:.4f})")
    
    try:
        # 🛡️ СУПЕР-STRICT VAD + КОНТЕКСТ
        segments, info = model.transcribe(
            audio_array,
            language="ru",
            beam_size=1,
            vad_filter=True,
            vad_parameters={
                "speech_thresh": 0.8,           # 80% уверенности
                "min_speech_duration_ms": 600,  # Минимум 0.6с речи
                "max_speech_duration_s": 12,
                "min_silence_duration_ms": 500,
                "speech_pad_ms": 100,
                "window_size_samples": 768
            },
            initial_prompt="Разговор по телефону о записи на прием к врачу",  # Контекст!
            condition_on_previous_text=False
        )
        
        # Собираем текст
        transcript_parts = []
        for segment in segments:
            text = segment.text.strip()
            if len(text) > 1:
                transcript_parts.append(text)
        
        transcript = " ".join(transcript_parts).strip()
        
        # 🛡️ ФИНАЛЬНЫЙ АНТИ-ХАЛЛЮЦИНАЦИОННЫЙ ФИЛЬТР
        transcript_lower = transcript.lower()
        if (len(transcript) < 3 or 
            len(transcript.split()) < 1 or
            any(pattern in transcript_lower for pattern in BAD_PATTERNS)):
            print(f"❌ FILTERED: '{transcript}'")
            return
        
        # ✅ ПРОВЕРКА КАЧЕСТВА (должно быть осмысленно)
        word_count = len(transcript.split())
        if word_count > 20 or word_count < 1:  # Слишком длинные/короткие
            print(f"❌ BAD LENGTH: {word_count} words")
            return
        
        print(f"✅ '{transcript}' ({item_id}) [{duration:.1f}s]")
        
        # LiveKit события
        await client['ws'].send(json.dumps({
            "type": "input_audio_buffer.speech_started",
            "item_id": item_id, 
            "audio_start_ms": 0
        }))
        await asyncio.sleep(0.03)
        
        await client['ws'].send(json.dumps({
            "type": "input_audio_buffer.speech_stopped", 
            "item_id": item_id, 
            "audio_end_ms": int(duration * 1000)
        }))
        await asyncio.sleep(0.03)
        
        await client['ws'].send(json.dumps({
            "type": "conversation.item.input_audio_transcription.completed",
            "item_id": item_id,
            "transcript": transcript,
            "language": "ru",
            "confidence": float(info.language_probability) if info else 0.95
        }))
        
    except Exception as e:
        print(f"❌ Whisper ERROR: {e}")

async def main():
    print("🚀 ULTRA-FAST + ANTI-HALLUCINATION WHISPER STT")
    print("📺 ТВ субтитры/шум = АВТОМАТИЧЕСКИЙ ОТСЕВ")
    print("⚡ Latency: 0.6-1.2s чанки")
    server = await websockets.serve(stt_handler, "0.0.0.0", 5000)
    await server.wait_closed()

if __name__ == '__main__':
    asyncio.run(main())
