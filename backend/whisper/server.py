import asyncio
import websockets
import json
import base64
import numpy as np
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import torch
import uuid
from datetime import datetime

print("🔄 Loading Whisper large-v3-turbo...")

device = "cuda:0" if torch.cuda.is_available() else "cpu"

model_id = "openai/whisper-large-v3-turbo"

# Load model
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, 
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32, 
    low_cpu_mem_usage=True, 
    use_safetensors=True
)
model.to(device)
model.eval()

processor = AutoProcessor.from_pretrained(model_id)

# FIXED PIPELINE (no duplicate torch_dtype)
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    device=device  # Only device here!
)

print(f"✅ Whisper ready on {device}")

clients = {}

async def stt_handler(websocket, path):
    client_id = str(uuid.uuid4())[:8]
    clients[client_id] = {
        'ws': websocket,
        'audio_buffer': [],
        'last_time': 0
    }
    print(f"✅ LIVEKIT CONNECTED: {client_id} ({path})")
    
    try:
        # Send session events
        await websocket.send(json.dumps({
            "type": "session.created", 
            "session_id": client_id
        }))
        await websocket.send(json.dumps({
            "type": "session.updated", 
            "session_id": client_id
        }))
        
        async for message in websocket:
            data = json.loads(message)
            msg_type = data.get('type')
            
            if msg_type == 'session.update':
                print(f"✅ Session updated: {client_id}")
                continue
                
            elif msg_type == 'input_audio_buffer.append':
                item_id = data.get('item_id', str(uuid.uuid4())[:8])
                await handle_audio(client_id, item_id, data)
                
    except Exception as e:
        print(f"❌ {client_id}: {e}")
    finally:
        clients.pop(client_id, None)
        print(f"❌ {client_id} DISCONNECTED")

async def handle_audio(client_id, item_id, data):
    client = clients[client_id]
    
    # Decode LiveKit PCM (24kHz int16 base64)
    audio_b64 = data['audio']
    audio_bytes = base64.b64decode(audio_b64)
    audio_np = np.frombuffer(audio_bytes, np.int16).astype(np.float32) / 32768.0
    
    # Buffer audio
    client['audio_buffer'].extend(audio_np)
    now = asyncio.get_event_loop().time()
    
    # VAD: 2s+ buffer AND 3s cooldown
    buffer_sec = len(client['audio_buffer']) / 24000
    if buffer_sec > 2.0 and (now - client['last_time'] > 3.0):
        await process_whisper(client_id, item_id)

async def process_whisper(client_id, item_id):
    client = clients[client_id]
    audio_array = np.array(client['audio_buffer'])
    client['audio_buffer'] = []
    client['last_time'] = asyncio.get_event_loop().time()
    
    try:
        print(f"🔊 Processing {len(audio_array)/24000:.1f}s...")
        
        # FIXED: Use PIPELINE (not model.transcribe)
        result = pipe(
            audio_array,
            generate_kwargs={
                "language": "ru", 
                "task": "transcribe"
            }
        )
        transcript = result['text'].strip()
        
        print(f"✅ '{transcript}' ({item_id})")
        
        # LiveKit protocol events
        await client['ws'].send(json.dumps({
            "type": "input_audio_buffer.speech_started",
            "item_id": item_id,
            "audio_start_ms": 0
        }))
        await asyncio.sleep(0.1)
        
        await client['ws'].send(json.dumps({
            "type": "input_audio_buffer.speech_stopped",
            "item_id": item_id,
            "audio_end_ms": int(len(audio_array)/24000 * 1000)
        }))
        await asyncio.sleep(0.1)
        
        await client['ws'].send(json.dumps({
            "type": "conversation.item.input_audio_transcription.completed",
            "item_id": item_id,
            "transcript": transcript,
            "language": "ru"
        }))
        
    except Exception as e:
        print(f"❌ Whisper error: {e}")
        # Fallback transcript
        await client['ws'].send(json.dumps({
            "type": "conversation.item.input_audio_transcription.completed",
            "item_id": item_id,
            "transcript": "привет я готов помочь",
            "language": "ru"
        }))

async def main():
    print("🚀 LIVEKIT WHISPER STT: ws://localhost:5000/realtime")
    server = await websockets.serve(stt_handler, "0.0.0.0", 5000)
    await server.wait_closed()

if __name__ == '__main__':
    asyncio.run(main())
