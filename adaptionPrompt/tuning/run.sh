

CUDA_VISIBLE_DEVICES=0 python finetune.py \
    --dataset_path data/train \
    --lora_rank 8 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --max_steps 50000 \
    --save_steps 500 \
    --save_total_limit 200 \
    --learning_rate 1e-4 \
    --fp16 \
    --remove_unused_columns false \
    --logging_steps 50 \
    --output_dir output_adaptionPrompt
