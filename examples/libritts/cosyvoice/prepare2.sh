#!/bin/bash
# Copyright 2024 Alibaba Inc. All Rights Reserved.
. ./path.sh || exit 1;

stage=-1
stop_stage=3

data_url=www.openslr.org/resources/60
data_dir=long-custom
pretrained_model_dir=../../../pretrained_models/CosyVoice-300M


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "Extract discrete speech token, you will get utt2speech_token.pt in data/$x dir"
  for x in $data_dir; do
    tools/extract_speech_token.py --dir data/$x \
      --onnx_path $pretrained_model_dir/speech_tokenizer_v1.onnx --num_thread 32
  done
fi