export TRANSFORMERS_CACHE=/mnt/sda/dongkeun/huggingface
export HF_DATASETS_CACHE=/mnt/sda/dongkeun/huggingface_datasets
export CUDA_VISIBLE_DEVICES=1

python eval_langbridge.py \
  --checkpoint_path checkpoints/orca2-lb-9b-ams \
  --enc_tokenizer kaist-ai/langbridge_encoder_tokenizer \
  --tasks copa_gen \
  --instruction_template orca \
  --batch_size 32 \
  --output_path eval_outputs/xcopa/orca2-langbrige_9b \
  --device cuda:0 \
  --no_cache \