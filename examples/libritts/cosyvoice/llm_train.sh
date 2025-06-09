#!/bin/bash
# train llm
# export PYTHONPATH=$(pwd):$PYTHONPATH  # cosyvoice 디렉토리가 포함된 경로 추가
. ./path.sh || exit 1;
export CUDA_VISIBLE_DEVICES="0,1,2"
num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
job_id=1986
dist_backend="nccl"
num_workers=2
prefetch=100
train_engine=torch_ddp

stage=-1
stop_stage=3

pretrained_model_dir=../../../pretrained_models/CosyVoice-300M
echo "Run train. We only support llm traning for now. If your want to train from scratch, please use conf/cosyvoice.fromscratch.yaml"
if [ $train_engine == 'deepspeed' ]; then
echo "Notice deepspeed has its own optimizer config. Modify conf/ds_stage2.json if necessary"
fi
cat data/250225-long-custom/parquet/data.list > data/train.data.list
cat data/250225-long-custom-eval/parquet/data.list > data/dev.data.list
#for model in llm flow hifigan; do
for model in llm; do
torchrun --nnodes=1 --nproc_per_node=$num_gpus \
    --rdzv_id=$job_id --rdzv_backend="c10d" --rdzv_endpoint="localhost:1234" \
    cosyvoice/bin/train.py \
    --train_engine $train_engine \
    --config conf/cosyvoice_ko.yaml \
    --train_data data/train.data.list \
    --cv_data data/dev.data.list \
    --model $model \
    --checkpoint $pretrained_model_dir/$model.pt \
    --model_dir `pwd`/exp/250225-long-custom/$model/$train_engine \
    --tensorboard_dir `pwd`/tensorboard/250225-long-custom/$model/$train_engine \
    --ddp.dist_backend $dist_backend \
    --num_workers ${num_workers} \
    --prefetch ${prefetch} \
    --pin_memory \
    --use_amp \
    --deepspeed_config ./conf/ds_stage2.json \
    --deepspeed.save_states model+optimizer
done
