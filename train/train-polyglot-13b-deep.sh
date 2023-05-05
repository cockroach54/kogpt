torchrun --nproc_per_node=4 --master_port=34321 train-polyglot-13b-deep.py \
    --data_path='../data/alpaca_data.json' \
    --output_dir='../models' \
    --bf16 True \
    --num_train_epochs=2 \
    --per_device_train_batch_size=1 \
    --gradient_accumulation_steps=8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --model_max_length 1024 \
    --deepspeed=ds_zero3-nooffload.json \
    --do_train \
    --logging_steps 1

