#!/bin/env bash
export OMP_NUM_THREADS=8
export TRANSFORMERS_CACHE=/mnt/sda/dongkeun/huggingface
export HF_DATASETS_CACHE=/mnt/sda/dongkeun/huggingface_datasets

export CUDA_VISIBLE_DEVICES=0,1,2,3
NUM_GPU=4

ARGS="
--n_gpu $NUM_GPU
--strategy deepspeed_stage_2
--output_dir checkpoints/orca2-lb-9b-barlow
--run_name orca2-lb-9b
--seed 42
--train_set_path DKYoon/slimorca-200k-english
--output_exists True
--enc_name_or_path ../../mST5-saved-2
--enc_lora_path ../../trained_models/mST5-barlow-final-true-2/final_model
--lm_name_or_path microsoft/Orca-2-7b
--alignments linear
--enc_hidden_size 4096
--lm_hidden_size 4096
--max_length 128
--max_length_enc 1024
--freeze_language_model True
--freeze_encoder True
--learning_rate_alignment 6e-4
--learning_rate_enc 2e-5
--w_decay_alignment 0.0
--w_decay_enc 0.1
--warmup_steps 0
--per_device_train_batch_size 4
--per_device_eval_batch_size 16
--gradient_accumulation_steps 8
--logging_steps 10
--num_train_epochs 1
--dataloader_num_workers 16
--bf16 True
"

echo $ARGS
if [ $NUM_GPU == 1 ]; then
    echo "running on a single GPU"
    python train_langbridge.py $ARGS
else
    echo "running on multiple GPUs"
    torchrun --nproc_per_node $NUM_GPU train_langbridge.py $ARGS
fi