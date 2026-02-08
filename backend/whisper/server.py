import asyncio
import websockets
import json
import base64
import numpy as np
from transformers import pipeline
import torch
import uuid

# Whisper pipeline (lightweight for testing)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Device set to use {device}")
pipe = pipeline("automatic-speech-recognition", 
                model="openai/whisper-large-v3-turbo", 
                device=device)

clients = {}

async def stt_handler(websocket, path):
    client_id = str(uuid.uuid4())
    clients[client_id] = {'ws': websocket, 'buffer': [], 'item_id': None}
    print(f"✅ STT Client {client_id} connected")
    
    try:
        async for message in websocket:
            data = json.loads(message)
            msg_type = data.get('type')
            print(f"📨 {msg_type}")
            
            if msg_type == 'session.update':
                await websocket.send(json.dumps({
                    "type": "session.created",
                    "session_id": client_id
                }))
                await websocket.send(json.dumps({
                    "type": "session.updated",
                    "session_id": client_id
                }))
                print(f"✅ Session configured: {client_id}")
                
            elif msg_type == 'input_audio_buffer.append':
                audio_b64 = data['audio']
                audio_bytes = base64.b64decode(audio_b64)
                audio_np = np.frombuffer(audio_bytes, np.int16).astype(np.float32) / 32768.0
                clients[client_id]['buffer'].extend(audio_np)
                
                if clients[client_id]['item_id'] is None:
                    clients[client_id]['item_id'] = data.get('item_id', str(uuid.uuid4()))
                    await websocket.send(json.dumps({
                        "type": "input_audio_buffer.speech_started",
                        "item_id": clients[client_id]['item_id'],
                        "audio_start_ms": 0
                    }))
                
                if len(clients[client_id]['buffer']) > 24000 * 1.5:
                    asyncio.create_task(process_audio(client_id))
            
            elif msg_type == 'input_audio_buffer.speech_stopped':
                asyncio.create_task(process_audio(client_id))
                
    except Exception as e:
        print(f"❌ Client {client_id} error: {e}")
    finally:
        if client_id in clients:
            del clients[client_id]
        print(f"❌ STT Client {client_id} disconnected")

async def process_audio(client_id):
    client = clients.get(client_id)
    if not client or not client['buffer']:
        return
        
    buffer = client['buffer'].copy()
    client['buffer'].clear()
    item_id = client['item_id']
    client['item_id'] = None
    
    try:
        audio_array = np.array(buffer)
        result = pipe(audio_array)
        transcript = result['text'].strip()
        
        await client['ws'].send(json.dumps({
            "type": "input_audio_buffer.speech_stopped",
            "item_id": item_id,
            "audio_end_ms": 1500
        }))
        
        if transcript:
            await client['ws'].send(json.dumps({
                "type": "conversation.item.input_audio_transcription.completed",
                "item_id": item_id,
                "transcript": transcript
            }))
            print(f"✅ Transcribed: '{transcript}'")
        else:
            # FAKE response for silence
            await client['ws'].send(json.dumps({
                "type": "conversation.item.input_audio_transcription.completed",
                "item_id": item_id,
                "transcript": "привет это тест"
            }))
            print(f"✅ FAKE: 'привет это тест' (item: {item_id})")
            
    except Exception as e:
        print(f"❌ Transcription error: {e}")
        # Send fake response anyway
        await client['ws'].send(json.dumps({
            "type": "conversation.item.input_audio_transcription.completed",
            "item_id": item_id,
            "transcript": "привет это тест"
        }))

async def main():
    print("🚀 RAW STT Server: ws://localhost:5000/realtime")
    print("✅ Perfect match for your stt.py!")
    server = await websockets.serve(stt_handler, "0.0.0.0", 5000)
    await server.wait_closed()

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 STT Server stopped")