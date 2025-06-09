import torch
from collections import OrderedDict
 
# 모델 로드
model = torch.load("/data/minju/TTS/CosyVoice_flow_cache/examples/libritts/cosyvoice2/exp/250512_lj_vctk_libri_slx/llm/torch_ddp/epoch_6_whole.pt")
 
# 체크포인트 정보 확인 (키 확인용)
print("원본 체크포인트 키:", model.keys())
 
# 일반 딕셔너리인 경우 OrderedDict로 변환
if not isinstance(model, OrderedDict):
    ordered_model = OrderedDict(model)
else:
    ordered_model = model
 
# epoch, step 관련 키 삭제
if 'epoch' in ordered_model:
    del ordered_model['epoch']
if 'step' in ordered_model:
    del ordered_model['step']
 
torch.save(ordered_model, "/data/minju/TTS/CosyVoice_flow_cache/pretrained_models/llm.pt")