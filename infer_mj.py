import sys
sys.path.append('third_party/Matcha-TTS')

from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio

import torch
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

model = "CosyVoice-300M-son"
cosyvoice = CosyVoice(f'/data/minju/TTS/CosyVoice/pretrained_models/{model}',load_jit=False, load_trt=False, fp16=False)
print(cosyvoice)

config_file = f"/data/minju/TTS/CosyVoice/pretrained_models/{model}/cosyvoice.yaml"
print(model)
# override_dict = {k: None for k in ['llm', 'flow', 'hift'] if k != "llm"}

# with open(config_file, 'r') as f:
#     configs = load_hyperpyyaml(f, overrides=override_dict)

print(cosyvoice.list_available_spks())

### llm 모델 변경

llm_change = False
spk = 'lye'

if llm_change:
    model = configs["llm"]
    model.load_state_dict(torch.load("/data/minju/TTS/CosyVoice/examples/libritts/cosyvoice/exp/cosyvoice/llm/torch_ddp/epoch_105_whole.pt", map_location='cuda'), strict=False)
    cosyvoice.model.llm = model.to("cuda").half()

for i, j in enumerate(cosyvoice.inference_sft('끊임없이 올라가기만 하고 재미없어요. 내려오기도 해야 재밌죠. 오르락 내리락이 있으니까 재미라는게 생기는거죠.', spk, stream=False)):
    torchaudio.save(f'250109-output/sft_{model}_{spk}_{i}.wav', j['tts_speech'], cosyvoice.sample_rate)