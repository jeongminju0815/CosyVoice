import torch
import os

model = torch.load('/data/minju/TTS/CosyVoice_flow_cache/examples/libritts/cosyvoice2/data/250411_son_ploonet_emo_multi_24k_long/spk2embedding.pt')
print(model.keys())
print(len(model.keys()))

new_model = {}
for key, value in model.items():
    new_model[key] = {'embedding':torch.tensor(value).unsqueeze(0)}

#torch.save(new_model, "/data/minju/TTS/CosyVoice/pretrained_models/CosyVoice-300M-son-ploonet/spk2info.pt")
save_path = "/data/minju/TTS/CosyVoice_flow_cache/embedding_models/250411_son_ploonet_emo_multi_24k_long"
os.makedirs(save_path, exist_ok=True)
torch.save(new_model, f"{save_path}/spk2info.pt")