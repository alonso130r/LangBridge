export TRANSFORMERS_CACHE=/mnt/sda/dongkeun/huggingface
export HF_DATASETS_CACHE=/mnt/sda/dongkeun/huggingface_datasets

# Change the visible GPUs
export CUDA_VISIBLE_DEVICES=0,1

# Replace 'python' with 'torchrun' and specify --nproc_per_node
torchrun --nproc_per_node=2 eval_langbridge.py \
  --checkpoint_path checkpoints/llemma-lb-9b-barlow/'epoch=1-step=3125' \
  --enc_tokenizer kaist-ai/langbridge_encoder_tokenizer \
  --dec_tokenizer EleutherAI/llemma_7b \
  --num_fewshot 8\
  --tasks mgsm_en,mgsm_es,mgsm_fr,mgsm_de,mgsm_ru,mgsm_zh,mgsm_ja,mgsm_th,mgsm_sw,mgsm_bn,mgsm_te\
  --batch_size 1 \
  --output_path eval_outputs/mgsm/llemma-langbrige_9b \
  --device cuda \
  --no_cache