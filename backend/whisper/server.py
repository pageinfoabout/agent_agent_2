import asyncio
import websockets
import json
import uuid
import time

clients = {}

# FIXED: NO 'path' parameter for websockets 12.0+
async def stt_handler(websocket):
    client_id = str(uuid.uuid4())[:8]
    clients[client_id] = {'ws': websocket}
    print(f"✅ STT Client {client_id} CONNECTED")
    
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
                print(f"✅ Session OK: {client_id}")
                
            elif msg_type == 'input_audio_buffer.append':
                item_id = data.get('item_id', str(uuid.uuid4())[:8])
                
                # Speech started
                await websocket.send(json.dumps({
                    "type": "input_audio_buffer.speech_started",
                    "item_id": item_id,
                    "audio_start_ms": 0
                }))
                
                # Realistic processing delay
                await asyncio.sleep(0.3)
                
                # Speech stopped
                await websocket.send(json.dumps({
                    "type": "input_audio_buffer.speech_stopped",
                    "item_id": item_id,
                    "audio_end_ms": 300
                }))
                
                # RUSSIAN TRANSCRIPTION
                await websocket.send(json.dumps({
                    "type": "conversation.item.input_audio_transcription.completed",
                    "item_id": item_id,
                    "transcript": "привет я готов помочь как ваши дела",
                    "words": [{"word": "привет", "start": 0, "end": 100}]
                }))
                print(f"✅ STT DONE: {item_id}")
                
    except Exception as e:
        print(f"❌ Client {client_id}: {e}")
    finally:
        clients.pop(client_id, None)
        print(f"❌ Client {client_id} DISCONNECTED")

async def main():
    print("🚀 STT Server: ws://localhost:5000/realtime")
    print("✅ FIXED for websockets 12.0+ (no path param)")
    server = await websockets.serve(stt_handler, "0.0.0.0", 5000)
    await server.wait_closed()

if __name__ == '__main__':
    asyncio.run(main())