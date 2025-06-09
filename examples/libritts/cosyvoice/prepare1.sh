#!/bin/bash
# Copyright 2024 Alibaba Inc. All Rights Reserved.
. ./path.sh || exit 1;

stage=-1
stop_stage=3

data_url=www.openslr.org/resources/60
data_dir=long-custom
pretrained_model_dir=../../../pretrained_models/CosyVoice-300M


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then #stage <=1 and stop_stage >= 1
  echo "Extract campplus speaker embedding, you will get spk2embedding.pt and utt2embedding.pt in data/$x dir"
  for x in $data_dir; do
    tools/extract_embedding.py --dir data/$x \
      --onnx_path $pretrained_model_dir/campplus.onnx --num_thread 32
  done
fi