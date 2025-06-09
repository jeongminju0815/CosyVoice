#!/bin/bash
# Copyright 2024 Alibaba Inc. All Rights Reserved.
. ./path.sh || exit 1;

stage=-1
stop_stage=3

data_url=www.openslr.org/resources/60
data_dir=/mnt/lyuxiang.lx/data/tts/openslr/libritts
pretrained_model_dir=../../../pretrained_models/CosyVoice-300M

echo `pwd`
# inference

echo "Run inference. Please make sure utt in tts_text is in prompt_data"
for mode in sft zero_shot; do
python cosyvoice/bin/inference.py --mode $mode \
    --gpu 0 \
    --config conf/cosyvoice.yaml \
    --prompt_data data/test-other/parquet/data.list \
    --prompt_utt2data data/test-other/parquet/utt2data.list \
    --tts_text `pwd`/tts_text.json \
    --llm_model $pretrained_model_dir/llm.pt \
    --flow_model $pretrained_model_dir/flow.pt \
    --hifigan_model $pretrained_model_dir/hift.pt \
    --result_dir `pwd`/output/$mode
done