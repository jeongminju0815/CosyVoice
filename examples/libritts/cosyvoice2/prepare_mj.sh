. ./path.sh || exit 1;

stage=-1
stop_stage=3

data_url=www.openslr.org/resources/60
data_dir=/mnt/lyuxiang.lx/data/tts/openslr/libritts
pretrained_model_dir=../../../pretrained_models/CosyVoice2-0.5B
train_dir=250410-son-ploonet-emo-multi-24k
eval_dir=250410-son-ploonet-emo-multi-24k-eval

# if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
#   echo "Extract campplus speaker embedding, you will get spk2embedding.pt and utt2embedding.pt in data/$x dir"
#   for x in $train_dir $eval_dir; do
#     tools/extract_embedding.py --dir data/$x \
#       --onnx_path $pretrained_model_dir/campplus.onnx --num_thread 20
#   done
# fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "Extract discrete speech token, you will get utt2speech_token.pt in data/$x dir"
  for x in $train_dir $eval_dir; do
    tools/extract_speech_token.py --dir data/$x \
      --onnx_path $pretrained_model_dir/speech_tokenizer_v2.onnx --num_thread 20
  done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "Prepare required parquet format data, you should have prepared wav.scp/text/utt2spk/spk2utt/utt2embedding.pt/spk2embedding.pt/utt2speech_token.pt"
  for x in $train_dir $eval_dir; do
    mkdir -p data/$x/parquet
    tools/make_parquet_list.py --num_utts_per_parquet 1000 \
      --num_processes 20 \
      --src_dir data/$x \
      --des_dir data/$x/parquet
  done
fi