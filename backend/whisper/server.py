import asyncio
import websockets
import json
import base64
import numpy as np
import uuid
import warnings
warnings.filterwarnings("ignore")
from faster_whisper import WhisperModel
import torch

print("🚀 FASTER WHISPER STT (4x optimized)")

# Lazy model loading
_model = None
_device = None

clients = {}

def get_model():
    """Lazy load faster_whisper model (3-4x faster than transformers)"""
    global _model, _device
    if _model is None:
        _device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"🔄 Loading faster_whisper large-v3-turbo on {_device}...")
        _model = WhisperModel(
            "large-v3-turbo",
            device=_device,
            compute_type="float16" if _device.startswith("cuda") else "float32",
            vad_filter=True,  # Built-in VAD (much faster)
            vad_parameters=dict(min_silence_duration_ms=500)
        )
        print("✅ Model ready!")
    return _model

async def stt_handler(websocket):
    client_id = str(uuid.uuid4())[:8]
    clients[client_id] = {'ws': websocket, 'audio_buffer': [], 'last_time': 0}
    print(f"✅ LIVEKIT CONNECTED: {client_id}")
    
    try:
        await websocket.send(json.dumps({"type": "session.created", "session_id": client_id}))
        await websocket.send(json.dumps({"type": "session.updated", "session_id": client_id}))
        
        async for message in websocket:
            data = json.loads(message)
            msg_type = data.get('type')
            
            if msg_type == 'input_audio_buffer.append':
                item_id = data.get('item_id', str(uuid.uuid4())[:8])
                await handle_audio(client_id, item_id, data)
                
    except Exception as e:
        print(f"❌ {client_id}: {e}")
    finally:
        clients.pop(client_id, None)

async def handle_audio(client_id, item_id, data):
    client = clients[client_id]
    
    # Decode PCM s16le -> float32
    audio_b64 = data['audio']
    audio_bytes = base64.b64decode(audio_b64)
    audio_np = np.frombuffer(audio_bytes, np.int16).astype(np.float32) / 32768.0
    
    client['audio_buffer'].extend(audio_np)
    now = asyncio.get_event_loop().time()
    
    # Improved VAD: 3s+ buffer AND 8s cooldown (faster processing)
    buffer_sec = len(client['audio_buffer']) / 24000
    if buffer_sec > 3.0 and (now - client['last_time'] > 8.0):
        await process_whisper(client_id, item_id)

async def process_whisper(client_id, item_id):
    client = clients[client_id]
    audio_array = np.array(client['audio_buffer'], dtype=np.float32)
    client['audio_buffer'] = []
    client['last_time'] = asyncio.get_event_loop().time()
    
    try:
        duration = len(audio_array) / 24000
        print(f"🔊 Processing {duration:.1f}s...")
        
        # Quick silence check (faster_whisper VAD will handle most cases)
        if np.std(audio_array) < 0.01:
            print("⏸️ Silence → Skip")
            return
        
        # FASTER WHISPER (4x faster than pipeline!)
        model = get_model()
        
        # Run in thread pool (non-blocking asyncio)
        segments, _ = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: model.transcribe(
                audio_array,
                language="ru",
                beam_size=1,        # Fastest decoding
                best_of=1,          # No reranking
                temperature=0.0,    # Deterministic
                compression_ratio_threshold=2.4,
                log_prob_threshold=-1.0,
                no_speech_threshold=0.6,
                condition_on_previous_text=False
            )
        )
        
        transcript = " ".join(seg.text.strip() for seg in segments).strip()
        
        # Filter bad transcripts
        bad_phrases = ["продолжение следует", "продолжение", "следует", "следует", ""]
        if any(phrase in transcript.lower() for phrase in bad_phrases) or len(transcript) < 3:
            print("❌ Filtered junk")
            return
        
        print(f"✅ '{transcript}' ({duration:.1f}s, {item_id})")
        
        # LiveKit events (non-blocking)
        await client['ws'].send(json.dumps({
            "type": "input_audio_buffer.speech_started",
            "item_id": item_id, "audio_start_ms": 0
        }))
        await asyncio.sleep(0.05)  # Reduced from 0.1
        
        await client['ws'].send(json.dumps({
            "type": "input_audio_buffer.speech_stopped", 
            "item_id": item_id, 
            "audio_end_ms": int(duration * 1000)
        }))
        await asyncio.sleep(0.05)
        
        await client['ws'].send(json.dumps({
            "type": "conversation.item.input_audio_transcription.completed",
            "item_id": item_id,
            "transcript": transcript,
            "language": "ru"
        }))
        
    except Exception as e:
        print(f"❌ Whisper error: {e}")

async def main():
    print("🚀 Starting FAST Whisper server on port 5000...")
    server = await websockets.serve(stt_handler, "0.0.0.0", 5000)
    await server.wait_closed()

if __name__ == '__main__':
    asyncio.run(main())
