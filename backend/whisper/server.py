from flask import Flask, request, jsonify
import numpy as np
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import io
import soundfile as sf

app = Flask(__name__)

print("🔄 Loading Whisper large-v3-turbo...")
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3-turbo"

# Load model with Russian optimization
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, 
    torch_dtype=torch_dtype, 
    low_cpu_mem_usage=True, 
    use_safetensors=True
)
model.to(device)
model.eval()  # Inference mode

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)

print(f"✅ Whisper API ready on {device}!")

@app.route('/transcribe', methods=['POST'])
def transcribe():
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        audio_bytes = audio_file.read()
        
        # Handle different audio formats (int16 → float32)
        if len(audio_bytes) < 1000:
            return jsonify({'error': 'Audio too short'}), 400
        
        # Try int16 first (LiveKit format)
        try:
            audio_data = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        except:
            # Fallback to float32
            audio_data = np.frombuffer(audio_bytes, dtype=np.float32)
        
        # Resample to 16kHz if needed (Whisper expects 16kHz)
        if len(audio_data) > 16000 * 30:  # Max 30s
            audio_data = audio_data[:16000*30]
        
        print(f"🔊 Processing {len(audio_data)/16000:.1f}s audio...")
        
        # Transcribe (Russian priority)
        result = pipe(
            audio_data, 
            generate_kwargs={"language": "ru", "task": "transcribe"},
            return_timestamps=False
        )
        
        transcription = result['text'].strip()
        
        if transcription:
            print(f"✅ '{transcription[:50]}...'")
            return jsonify({
                'transcription': transcription,
                'language': 'ru',
                'duration': len(audio_data)/16000
            })
        else:
            return jsonify({'transcription': '', 'language': 'ru'})
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model': 'whisper-large-v3-turbo'})

if __name__ == '__main__':
    print("🚀 Whisper API: POST /transcribe (multipart/form-data)")
    print("📱 curl -F audio=@test.wav http://localhost:5000/transcribe")
    app.run(host='0.0.0.0', port=5000, debug=False)