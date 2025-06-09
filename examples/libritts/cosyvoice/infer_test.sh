python cosyvoice/bin/inference.py --mode zero_shot \
    --gpu 2 \
    --config /data/minju/TTS/CosyVoice_origin/pretrained_models/CosyVoice2-0.5B/cosyvoice.yaml \
    --prompt_data data/semi-eval/parquet/data.list \
    --prompt_utt2data data/semi-eval/parquet/utt2data.list \
    --tts_text `pwd`/tts_text_lye.json \
    --llm_model /data/minju/TTS/CosyVoice_origin/pretrained_models/CosyVoice2-0.5B/llm.pt \
    --flow_model /data/minju/TTS/CosyVoice_origin/pretrained_models/CosyVoice2-0.5B/flow.pt \
    --hifigan_model /data/minju/TTS/CosyVoice_origin/pretrained_models/CosyVoice2-0.5B/hift.pt \
    --result_dir `pwd`/exp/cosyvoice/test-clean/mj_test