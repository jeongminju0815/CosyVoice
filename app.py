import uvicorn
from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse
import time
import struct
import numpy as np
import os

from setproctitle import setproctitle
import sys
sys.path.append('third_party/Matcha-TTS')

from cosyvoice.cli.cosyvoice import CosyVoice
from cosyvoice.utils.file_utils import load_wav

import torchaudio
import torch

from cosyvoice.utils.train_utils import (
init_distributed,
init_dataset_and_dataloader,
init_optimizer_and_scheduler,
init_summarywriter, save_model,
wrap_cuda_model, check_modify_and_save_config)
from hyperpyyaml import load_hyperpyyaml

tag_metadata = [{'name': 'synthesis'}]
app = FastAPI(
    title='CosyVoice TTS',
    version='1.0.0',
    openapi_tags=tag_metadata
)
max_wav_value = 32768.0

def _convert_audio_to_pcm16(wav_int16, sample_rate):
    data = wav_int16
    # dkind = data.dtype.kind
    fs = sample_rate

    len_header = 36 # len(header_data)
    chunk_size = len_header + data.nbytes 
    chunk_size_le = struct.pack('<I', chunk_size)

    header_data = b''
    header_data += b'RIFF'
    header_data += chunk_size_le
    header_data += b'WAVE'
    header_data += b'fmt '

    format_tag = 0x0001  # WAVE_FORMAT_PCM
    channels = 1
    bit_depth = data.dtype.itemsize * 8
    bytes_per_second = fs * (bit_depth // 8) * channels
    block_align = channels * (bit_depth // 8)

    fmt_chunk_data = struct.pack('<HHIIHH', format_tag, channels, fs, bytes_per_second, block_align, bit_depth)
    header_data += struct.pack('<I', len(fmt_chunk_data))
    header_data += fmt_chunk_data

    if (chunk_size) > 0xFFFFFFFF:
        raise ValueError("Data exceeds wave file size limit")

    data_chunk_data = b'data'
    data_chunk_data += struct.pack('<I', data.nbytes)
    header_data += data_chunk_data

    data_bytes = data.tobytes()

    data_pcm16 = header_data
    data_pcm16 += data_bytes
    return data_pcm16


@app.get('/tts/stream', tags=['synthesis'])
def process_tts_stream(text: str, sid: str):
    start_time = time.time()
    
    total_tensor = []
    for i, j in enumerate(cosyvoice.inference_sft(text, sid, stream=False)):
        total_tensor.append(j['tts_speech'])
        # torchaudio.save(f"{save_path}_{i}.wav", j['tts_speech'], cosyvoice.sample_rate)
    audio = torch.cat(total_tensor, dim=-1)
    torchaudio.save(f"app_test.wav", j['tts_speech'], cosyvoice.sample_rate)
    audio = audio.numpy()

    audio = audio * max_wav_value
    audio = audio.astype(np.int16)

    end_time = time.time()

    print(f'  TTS inference: {end_time - start_time:.3f}s')
    
    audio = _convert_audio_to_pcm16(audio, 22050)
    return Response(content=audio,
                media_type='audio/wav')


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"]= "2"
    setproctitle(f"COSY VOICE TEST")

    cosyvoice = CosyVoice('./pretrained_models/CosyVoice-300M-25Hz',load_jit=False, load_trt=False)
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

    config_file = "./pretrained_models/CosyVoice-300M-25Hz/cosyvoice.yaml"
    override_dict = {k: None for k in ['llm', 'flow', 'hift'] if k != "llm"}

    with open(config_file, 'r') as f:
        configs = load_hyperpyyaml(f, overrides=override_dict)

    # ### llm 모델 변경  
    # epoch = 16
    # model = configs["llm"]
    # model.load_state_dict(torch.load(f"/data/minju/TTS/CosyVoice/examples/libritts/cosyvoice/exp/250111-son-ploonet/llm/torch_ddp/epoch_{epoch}_whole.pt", map_location='cuda'), strict=False)
    # cosyvoice.model.llm = model.to("cuda")

    # ### spk 모델 변경 
    # emb_path = "/data/minju/TTS/CosyVoice/embedding_models/son-ploonet/spk2info.pt"
    # emb_type = emb_path.split("/")[-2]
    # spk2info = torch.load(emb_path, map_location = "cuda")
    # cosyvoice.frontend.spk2info = spk2info

    # print("============= INFO ===============")
    # print(epoch)
    # print(emb_type)


    print(cosyvoice.list_available_spks())
    print(len(cosyvoice.list_available_spks()))
    print("============================")

    uvicorn.run(app, host='0.0.0.0', port=60705)
    # uvicorn.run("app:app", host='0.0.0.0', port=60705, reload=True)
