import asyncio
import websockets
import json
import uuid
import time

clients = {}

async def stt_handler(websocket, path):
    client_id = str(uuid.uuid4())
    clients[client_id] = {'ws': websocket}
    print(f"✅ FAKE STT Client {client_id} CONNECTED")
    
    try:
        async for message in websocket:
            data = json.loads(message)
            msg_type = data.get('type')
            print(f"📨 {msg_type}")
            
            if msg_type == 'session.update':
                # REQUIRED handshake
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
                item_id = data.get('item_id', str(uuid.uuid4()))
                print(f"🎤 Audio received: {item_id}")
                
                # IMMEDIATE response (no Whisper delay)
                await asyncio.sleep(0.1)  # Tiny delay
                
                await websocket.send(json.dumps({
                    "type": "input_audio_buffer.speech_started",
                    "item_id": item_id,
                    "audio_start_ms": 0
                }))
                
                await asyncio.sleep(0.5)  # Simulate processing
                
                await websocket.send(json.dumps({
                    "type": "input_audio_buffer.speech_stopped",
                    "item_id": item_id,
                    "audio_end_ms": 500
                }))
                
                # FAKE RUSSIAN TRANSCRIPTION
                await websocket.send(json.dumps({
                    "type": "conversation.item.input_audio_transcription.completed",
                    "item_id": item_id,
                    "transcript": "привет я понимаю русский язык и готов помочь"
                }))
                print(f"✅ FAKE STT COMPLETE: {item_id}")
                
    except Exception as e:
        print(f"❌ Client {client_id} error: {e}")
    finally:
        if client_id in clients:
            del clients[client_id]
        print(f"❌ FAKE STT Client {client_id} DISCONNECTED")

async def main():
    print("🚀 FAKE STT Server: ws://localhost:5000/realtime")
    print("✅ 100% STABLE - No Whisper crashes!")
    server = await websockets.serve(stt_handler, "0.0.0.0", 5000)
    await server.wait_closed()

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Fake STT Server stopped")