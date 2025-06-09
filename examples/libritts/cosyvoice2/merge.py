import torch

# 두 .pt 파일 로드
mode = 'spk2embedding.pt'
path1 = '/data/minju/TTS/CosyVoice_flow_cache/examples/libritts/cosyvoice2/data/250422_all_24k_filter_newname'
path2 = '/data/minju/TTS/CosyVoice_flow_cache/examples/libritts/cosyvoice2/data/250509_lj_vctk_libri'

dict1 = torch.load(f"{path1}/{mode}")
dict2 = torch.load(f"{path2}/{mode}")

# dict 병합 (key 중복 시 dict2 값이 우선됨)
merged_dict = {**dict1, **dict2}

# 병합된 dict 저장
save_path = '/data/minju/TTS/CosyVoice_flow_cache/examples/libritts/cosyvoice2/data/250512_lj_vctk_libri_slx'
torch.save(merged_dict, f"{save_path}/{mode}")
