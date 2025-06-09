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
import os
import time
from tqdm import tqdm
from hyperpyyaml import load_hyperpyyaml
from modelscope import snapshot_download
import torch
from cosyvoice.cli_mj.frontend import CosyVoiceFrontEnd
from cosyvoice.cli_mj.model import CosyVoiceModel, CosyVoice2Model
from cosyvoice.utils.file_utils import logging
from cosyvoice.utils.class_utils import get_model_type


class CosyVoice:

    def __init__(self, model_dir, load_jit=False, load_trt=False, fp16=False):
        self.instruct = True if '-Instruct' in model_dir else False
        self.model_dir = model_dir
        self.fp16 = fp16
        if not os.path.exists(model_dir):
            model_dir = snapshot_download(model_dir)
        with open('{}/cosyvoice.yaml'.format(model_dir), 'r') as f:
            configs = load_hyperpyyaml(f)
        assert get_model_type(configs) != CosyVoice2Model, 'do not use {} for CosyVoice initialization!'.format(model_dir)
        self.frontend = CosyVoiceFrontEnd(configs['get_tokenizer'],
                                          configs['feat_extractor'],
                                          '{}/campplus.onnx'.format(model_dir),
                                          '{}/speech_tokenizer_v1.onnx'.format(model_dir),
                                          '{}/spk2info.pt'.format(model_dir),
                                          configs['allowed_special'])
        self.sample_rate = configs['sample_rate']
        if torch.cuda.is_available() is False and (load_jit is True or load_trt is True or fp16 is True):
            load_jit, load_trt, fp16 = False, False, False
            logging.warning('no cuda device, set load_jit/load_trt/fp16 to False')
        self.model = CosyVoiceModel(configs['llm'], configs['flow'], configs['hift'], fp16)
        self.model.load('{}/llm.pt'.format(model_dir),
                        '{}/flow.pt'.format(model_dir),
                        '{}/hift.pt'.format(model_dir))
        if load_jit:
            self.model.load_jit('{}/llm.text_encoder.{}.zip'.format(model_dir, 'fp16' if self.fp16 is True else 'fp32'),
                                '{}/llm.llm.{}.zip'.format(model_dir, 'fp16' if self.fp16 is True else 'fp32'),
                                '{}/flow.encoder.{}.zip'.format(model_dir, 'fp16' if self.fp16 is True else 'fp32'))
        if load_trt:
            self.model.load_trt('{}/flow.decoder.estimator.{}.v100.plan'.format(model_dir, 'fp16' if self.fp16 is True else 'fp32'))
        del configs

    def list_available_spks(self):
        spks = list(self.frontend.spk2info.keys())
        return spks

    def inference_sft(self, tts_text, spk_id, stream=False, speed=1.0, text_frontend=True):
        print(self.frontend.text_normalize(tts_text, split=True, text_frontend=text_frontend))
        for i in tqdm(self.frontend.text_normalize(tts_text, split=True, text_frontend=text_frontend)):
            model_input = self.frontend.frontend_sft(i, spk_id)
            start_time = time.time()
            logging.info('synthesis text {}'.format(i))
            for model_output in self.model.tts(**model_input, stream=stream, speed=speed):
                speech_len = model_output['tts_speech'].shape[1] / self.sample_rate
                logging.info('yield speech len {}, rtf {}'.format(speech_len, (time.time() - start_time) / speech_len))
                yield model_output
                start_time = time.time()

    def inference_zero_shot(self, tts_text, prompt_text, prompt_speech_16k, stream=False, speed=1.0, text_frontend=True):
        prompt_text = self.frontend.text_normalize(prompt_text, split=False, text_frontend=text_frontend)
        for i in tqdm(self.frontend.text_normalize(tts_text, split=True, text_frontend=text_frontend)):
            if len(i) < 0.5 * len(prompt_text):
                logging.warning('synthesis text {} too short than prompt text {}, this may lead to bad performance'.format(i, prompt_text))
            model_input = self.frontend.frontend_zero_shot(i, prompt_text, prompt_speech_16k, self.sample_rate)
            start_time = time.time()
            logging.info('synthesis text {}'.format(i))
            for model_output in self.model.tts(**model_input, stream=stream, speed=speed):
                speech_len = model_output['tts_speech'].shape[1] / self.sample_rate
                logging.info('yield speech len {}, rtf {}'.format(speech_len, (time.time() - start_time) / speech_len))
                yield model_output
                start_time = time.time()

    def inference_cross_lingual(self, tts_text, prompt_speech_16k, stream=False, speed=1.0, text_frontend=True):
        for i in tqdm(self.frontend.text_normalize(tts_text, split=True, text_frontend=text_frontend)):
            model_input = self.frontend.frontend_cross_lingual(i, prompt_speech_16k, self.sample_rate)
            start_time = time.time()
            logging.info('synthesis text {}'.format(i))
            for model_output in self.model.tts(**model_input, stream=stream, speed=speed):
                speech_len = model_output['tts_speech'].shape[1] / self.sample_rate
                logging.info('yield speech len {}, rtf {}'.format(speech_len, (time.time() - start_time) / speech_len))
                yield model_output
                start_time = time.time()

    def inference_instruct(self, tts_text, spk_id, instruct_text, stream=False, speed=1.0, text_frontend=True):
        assert isinstance(self.model, CosyVoiceModel), 'inference_instruct is only implemented for CosyVoice!'
        if self.instruct is False:
            raise ValueError('{} do not support instruct inference'.format(self.model_dir))
        instruct_text = self.frontend.text_normalize(instruct_text, split=False, text_frontend=text_frontend)
        for i in tqdm(self.frontend.text_normalize(tts_text, split=True, text_frontend=text_frontend)):
            model_input = self.frontend.frontend_instruct(i, spk_id, instruct_text)
            start_time = time.time()
            logging.info('synthesis text {}'.format(i))
            for model_output in self.model.tts(**model_input, stream=stream, speed=speed):
                speech_len = model_output['tts_speech'].shape[1] / self.sample_rate
                logging.info('yield speech len {}, rtf {}'.format(speech_len, (time.time() - start_time) / speech_len))
                yield model_output
                start_time = time.time()

    def inference_vc(self, source_speech_16k, prompt_speech_16k, stream=False, speed=1.0):
        model_input = self.frontend.frontend_vc(source_speech_16k, prompt_speech_16k, self.sample_rate)
        start_time = time.time()
        for model_output in self.model.vc(**model_input, stream=stream, speed=speed):
            speech_len = model_output['tts_speech'].shape[1] / self.sample_rate
            logging.info('yield speech len {}, rtf {}'.format(speech_len, (time.time() - start_time) / speech_len))
            yield model_output
            start_time = time.time()


class CosyVoice2(CosyVoice):

    def __init__(self, model_dir, load_jit=False, load_trt=False, fp16=False):
        self.instruct = True if '-Instruct' in model_dir else False
        self.model_dir = model_dir
        self.fp16 = fp16
        if not os.path.exists(model_dir):
            model_dir = snapshot_download(model_dir)
        with open('{}/cosyvoice.yaml'.format(model_dir), 'r') as f:
            configs = load_hyperpyyaml(f, overrides={'qwen_pretrain_path': os.path.join(model_dir, 'CosyVoice-BlankEN')})
        assert get_model_type(configs) == CosyVoice2Model, 'do not use {} for CosyVoice2 initialization!'.format(model_dir)
        self.frontend = CosyVoiceFrontEnd(configs['get_tokenizer'],
                                          configs['feat_extractor'],
                                          '{}/campplus.onnx'.format(model_dir),
                                          '{}/speech_tokenizer_v2.onnx'.format(model_dir),
                                          '{}/spk2info.pt'.format(model_dir),
                                          configs['allowed_special'])
        self.sample_rate = configs['sample_rate']
        if torch.cuda.is_available() is False and (load_jit is True or load_trt is True or fp16 is True):
            load_jit, load_trt, fp16 = False, False, False
            logging.warning('no cuda device, set load_jit/load_trt/fp16 to False')
        self.model = CosyVoice2Model(configs['llm'], configs['flow'], configs['hift'], fp16)
        self.model.load('{}/llm.pt'.format(model_dir),
                        '{}/flow.pt'.format(model_dir),
                        '{}/hift.pt'.format(model_dir))
        if load_jit:
            self.model.load_jit('{}/flow.encoder.{}.zip'.format(model_dir, 'fp16' if self.fp16 is True else 'fp32'))
        if load_trt:
            self.model.load_trt('{}/flow.decoder.estimator.{}.v100.plan'.format(model_dir, 'fp16' if self.fp16 is True else 'fp32'))
        del configs

    def inference_instruct(self, *args, **kwargs):
        raise NotImplementedError('inference_instruct is not implemented for CosyVoice2!')

    def inference_instruct2(self, tts_text, instruct_text, prompt_speech_16k, stream=False, speed=1.0, text_frontend=True):
        assert isinstance(self.model, CosyVoice2Model), 'inference_instruct2 is only implemented for CosyVoice2!'
        for i in tqdm(self.frontend.text_normalize(tts_text, split=True, text_frontend=text_frontend)):
            model_input = self.frontend.frontend_instruct2(i, instruct_text, prompt_speech_16k, self.sample_rate)
            start_time = time.time()
            logging.info('synthesis text {}'.format(i))
            for model_output in self.model.tts(**model_input, stream=stream, speed=speed):
                speech_len = model_output['tts_speech'].shape[1] / self.sample_rate
                logging.info('yield speech len {}, rtf {}'.format(speech_len, (time.time() - start_time) / speech_len))
                yield model_output
                start_time = time.time()

if __name__ == "__main__":
    import sys
    sys.path.append('/data/minju/TTS/CosyVoice/third_party/Matcha-TTS')

    from cosyvoice.cli_mj.cosyvoice import CosyVoice
    from cosyvoice.utils.file_utils import load_wav
    import torchaudio

    cosyvoice = CosyVoice('/data/minju/TTS/CosyVoice/pretrained_models/CosyVoice-300M-son-ploonet',load_jit=False, load_trt=False)

    import torch
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    from cosyvoice.utils.train_utils import (
    init_distributed,
    init_dataset_and_dataloader,
    init_optimizer_and_scheduler,
    init_summarywriter, save_model,
    wrap_cuda_model, check_modify_and_save_config)
    from hyperpyyaml import load_hyperpyyaml

    config_file = "/data/minju/TTS/CosyVoice/pretrained_models/CosyVoice-300M-son-ploonet/cosyvoice.yaml"

    override_dict = {k: None for k in ['llm', 'flow', 'hift'] if k != "llm"}

    with open(config_file, 'r') as f:
        configs = load_hyperpyyaml(f, overrides=override_dict)

    #configs['train_conf'].update(vars(args))

    print(configs)

    ## 임베딩 파일 경로 변경
    print(cosyvoice.list_available_spks())
    print(len(cosyvoice.list_available_spks()))

    emb_path = "/data/minju/TTS/CosyVoice/embedding_models/son-ploonet/spk2info.pt"
    # emb_path = "/data/minju/TTS/CosyVoice/embedding_models/slx/spk2info.pt"
    # emb_path = "/data/minju/TTS/CosyVoice/embedding_models/semi/spk2info.pt"
    emb_path = "/data/minju/TTS/CosyVoice/embedding_models/senti-slx/spk2info.pt"
    # emb_path = "/data/minju/TTS/CosyVoice/embedding_models/cardergarden/spk2info.pt"
    emb_type = emb_path.split("/")[-2]
    spk2info = torch.load(emb_path, map_location = "cuda")
    cosyvoice.frontend.spk2info = spk2info

    print(cosyvoice.list_available_spks())
    print(len(cosyvoice.list_available_spks()))

    # sft usage
    # change stream=True for chunk stream inference
    epoch = 0

    model = configs["llm"]
    model.load_state_dict(torch.load(f"/data/minju/TTS/CosyVoice/examples/libritts/cosyvoice/exp/250114-senti-slx/llm/torch_ddp/epoch_{epoch}_whole.pt", map_location='cuda'), strict=False)
    cosyvoice.model.llm = model.to("cuda")

    speaker = "lye"
    text = "오늘부터 사나흘에 걸쳐 충남과 호남 지역 제주도 등에 많은 눈이 내리겠습니다. 이 시각 대설주의보가 내려진 전북 고창의 모습을 보면 거리에 눈이 쌓인 가운데 굵은 눈이 펑펑 쏟아지고 있습니다. 서해안에서 강한 눈구름대가 유입되면서 서쪽 지역을 중심으로 눈이 내리고 있습니다."
    # text = "이 시각 대설주의보가 내려진 전북 고창의 모습을 보면 거리에 눈이 쌓인 가운데 굵은 눈이 펑펑 쏟아지고 있습니다."
    # text = "끊임없이 올라가기만 하고 재미없어요. 내려오기도 해야 재밌죠. 오르락 내리락이있으니까 재미라는게 생기는거죠."
    text = "그룹 블랙핑크 로제와 팝스타 브루노 마스가 함께 부른 아파트가 빌보드 차트에서 전주보다 이십 구 계단 상승하며 오위에 올랐습니다."
    # text = "이 직무대행은 13일 국회 행정안전위원회 현안 질의에 출석해 더불어민주당 모경종 의원으로부터 국민의힘 의원들의 체포 저지 행동에 대한 질문을 받고 적극적으로 체포를 저지하면 현행범이 될 수 있다고 말했다."
    # text = "대장 내시경 검사가 대장암 발생률과 사망률을 낮추는 데 큰 효과가 있다는 대규모 분석 결과가 나왔어."
    text = "그리운 연진에게! 혹시 기억하니? 내가 여름을 아주 싫어했던거~ 다행히, 더 더워지기 전에 이사를 했어."
    output_path = "output/250114"
    os.makedirs(output_path, exist_ok=True)
    save_path = f"{output_path}/sft_{speaker}_{epoch}_{text[:3]}"

    total_tensor = []

    for i, j in enumerate(cosyvoice.inference_sft(text, speaker, stream=False)):
        print(j['tts_speech'])
        total_tensor.append(j['tts_speech'])
        torchaudio.save(f"{save_path}_{i}.wav", j['tts_speech'], cosyvoice.sample_rate)
        #Audio(f"{save_path}_{i}.wav")

    # concatenated_tensor = torch.cat(total_tensor, dim=-1)
    # torchaudio.save('test.wav', concatenated_tensor, cosyvoice.sample_rate)