# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
import requests
import torch
import torchaudio
import numpy as np
from pathlib import Path

logging.basicConfig(level=logging.INFO)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, default='37.9.15.195')
    parser.add_argument('--port', type=int, default=50000)
    parser.add_argument('--mode', default='zero_shot', choices=['sft', 'zero_shot', 'cross_lingual', 'instruct'])
    parser.add_argument('--tts_text', type=str, required=True)
    parser.add_argument('--spk_id', type=str, default='中文女')
    parser.add_argument('--prompt_text', type=str, default='reference voice')
    parser.add_argument('--prompt_wav', type=str, default='prompts/prompt.wav')
    parser.add_argument('--instruct_text', type=str, default='default instruct')
    parser.add_argument('--tts_wav', type=str, default='output.wav')
    args = parser.parse_args()
    
    # Проверка файла
    prompt_path = Path(args.prompt_wav)
    if not prompt_path.exists():
        logging.error(f"❌ Файл не найден: {prompt_path}")
        return
    
    url = f"http://{args.host}:{args.port}/inference_{args.mode}"
    logging.info(f"📡 Отправка на: {url}")
    
    try:
        if args.mode == 'sft':
            response = requests.post(url, data={
                'tts_text': args.tts_text,
                'spk_id': args.spk_id
            }, timeout=60)
            
        elif args.mode == 'zero_shot':
            with open(args.prompt_wav, 'rb') as f:
                files = [('prompt_wav', ('prompt_wav', f, 'application/octet-stream'))]
                response = requests.post(url, data={
                    'tts_text': args.tts_text,
                    'prompt_text': args.prompt_text
                }, files=files, timeout=120)
                
        elif args.mode == 'cross_lingual':
            with open(args.prompt_wav, 'rb') as f:
                files = [('prompt_wav', ('prompt_wav', f, 'application/octet-stream'))]
                response = requests.post(url, data={
                    'tts_text': args.tts_text
                }, files=files, timeout=120)
                
        else:  # instruct
            response = requests.post(url, data={
                'tts_text': args.tts_text,
                'spk_id': args.spk_id,
                'instruct_text': args.instruct_text
            }, timeout=60)
        
        response.raise_for_status()
        tts_audio = response.content
        
        logging.info(f"📥 Получено {len(tts_audio)} байт")
        
        # Фикс четности для int16
        if len(tts_audio) % 2 != 0:
            tts_audio = tts_audio[:-1]
            logging.info("🔧 Обрезано до четного количества байт")
        
        tts_speech = torch.from_numpy(np.frombuffer(tts_audio, dtype=np.int16)).unsqueeze(0)
        torchaudio.save(args.tts_wav, tts_speech, 22050)
        logging.info(f'✅ Сохранено: {args.tts_wav}')
        
    except requests.exceptions.RequestException as e:
        logging.error(f"❌ HTTP ошибка: {e}")
    except Exception as e:
        logging.error(f"❌ Ошибка: {e}")

if __name__ == "__main__":
    main()
