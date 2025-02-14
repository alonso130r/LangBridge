export TRANSFORMERS_CACHE=/mnt/sda/dongkeun/huggingface
export HF_DATASETS_CACHE=/mnt/sda/dongkeun/huggingface_datasets
export CUDA_VISIBLE_DEVICES=3

python eval_langbridge.py \
  --checkpoint_path checkpoints/metamath-lb-9b-barlow\
  --enc_tokenizer kaist-ai/langbridge_encoder_tokenizer \
  --tasks msvamp_en,msvamp_es,msvamp_fr,msvamp_de,msvamp_ru,msvamp_zh,msvamp_ja,msvamp_th,msvamp_sw,msvamp_bn\
  --instruction_template metamath \
  --batch_size 1 \
  --output_path eval_outputs/msvamp/llemma-langbridge_9b \
  --device cuda:0 \
  --no_cache


