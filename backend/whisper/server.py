from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit, join_room, leave_room
import numpy as np
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import base64
import asyncio
import threading
import io
from collections import deque
import time

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Load Whisper model (same as yours)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
model_id = "openai/whisper-large-v3-turbo"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)
processor = AutoProcessor.from_pretrained(model_id)
pipe = pipeline("automatic-speech-recognition", model=model, tokenizer=processor.tokenizer, 
                feature_extractor=processor.feature_extractor, torch_dtype=torch_dtype, device=device)

# Per-session audio buffer (24kHz PCM)
audio_buffers = {}
item_timings = {}

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')
    # Cleanup
    for sid in audio_buffers.keys():
        if sid == request.sid:
            del audio_buffers[sid]

@socketio.on('message')
def handle_message(data):
    msg_type = data.get('type')
    session_id = data.get('session_id', request.sid)
    
    if msg_type == 'session.update':
        # Respond to initial config from stt.py
        emit('session.created', {'session_id': session_id})
        emit('session.updated', {'session_id': session_id})
        audio_buffers[request.sid] = deque(maxlen=24000*10)  # 10s buffer
        print(f"Session configured: {session_id}")
    
    elif msg_type == 'input_audio_buffer.append':
        # Decode 24kHz PCM audio chunk
        audio_b64 = data['audio']
        audio_bytes = base64.b64decode(audio_b64)
        audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        
        # Buffer audio
        audio_buffers[request.sid].extend(audio_np)
        
        item_id = data.get('item_id', 'item-1')
        item_timings.setdefault(request.sid, {})[item_id] = {'start_ms': int(time.time() * 1000)}
        
        # Send speech started
        emit('input_audio_buffer.speech_started', {
            'item_id': item_id,
            'audio_start_ms': 0
        })
        
        # Process when enough audio accumulated (~1-2s) or on speech_stopped
        if len(audio_buffers[request.sid]) > 24000 * 1.5:  # 1.5s
            asyncio.run_coroutine_threadsafe(process_audio(request.sid, item_id), socketio.loop)
    
    elif msg_type == 'input_audio_buffer.speech_stopped':
        item_id = data.get('item_id')
        asyncio.run_coroutine_threadsafe(process_audio(request.sid, item_id), socketio.loop)

async def process_audio(sid, item_id):
    """Process buffered audio with Whisper"""
    if sid not in audio_buffers or not audio_buffers[sid]:
        return
    
    # Convert buffer to numpy array and resample if needed
    audio_array = np.array(audio_buffers[sid])
    
    # Transcribe with your Whisper pipeline
    result = pipe(audio_array)
    transcription = result['text'].strip()
    
    # Get timing
    timing = item_timings.get(sid, {}).get(item_id, {})
    audio_end_ms = timing.get('end_ms', int(time.time() * 1000))
    audio_start_ms = timing.get('start_ms', 0)
    audio_duration_ms = audio_end_ms - audio_start_ms
    
    # Send final events
    socketio.emit('input_audio_buffer.speech_stopped', {
        'item_id': item_id,
        'audio_end_ms': audio_duration_ms
    }, room=request.sid)
    
    socketio.emit('conversation.item.input_audio_transcription.completed', {
        'item_id': item_id,
        'transcript': transcription
    }, room=request.sid)
    
    # Clear buffer for next utterance
    audio_buffers[sid].clear()
    if item_id in item_timings.get(sid, {}):
        del item_timings[sid][item_id]

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)
