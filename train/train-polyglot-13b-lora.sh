
# torchrun --nproc_per_node=4 --master_port=34321 train-polyglot-13b-lora.py \
python train-polyglot-13b-lora.py \
    --data_path='../data/alpaca_data.json' \
    --output_dir='../lora' \
    --bf16 True \
    --num_train_epochs 3 \
    --per_device_train_batch_size 3 \
    --per_device_eval_batch_size 3 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --report_to "none" \
    --model_max_length 1024 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'GPTNeoXLayer' \
    --logging_steps 1

    # --use_mps_device "true" \
    # --model_name='EleutherAI/polyglot-ko-1.3b' \



# options:
#   -h, --help            show this help message and exit
#   --model_name_or_path MODEL_NAME_OR_PATH
#   --data_path DATA_PATH
#                         Path to the training data. (default: None)
#   --output_dir OUTPUT_DIR
#                         The output directory where the model predictions and checkpoints will be written. (default: None)
#   --overwrite_output_dir [OVERWRITE_OUTPUT_DIR]
#                         Overwrite the content of the output directory. Use this to continue training if output_dir points to a checkpoint directory. (default: False)
#   --do_train [DO_TRAIN]
#                         Whether to run training. (default: False)
#   --do_eval [DO_EVAL]   Whether to run eval on the dev set. (default: False)
#   --do_predict [DO_PREDICT]
#                         Whether to run predictions on the test set. (default: False)
#   --evaluation_strategy {no,steps,epoch}
#                         The evaluation strategy to use. (default: no)
#   --prediction_loss_only [PREDICTION_LOSS_ONLY]
#                         When performing evaluation and predictions, only returns the loss. (default: False)
#   --per_device_train_batch_size PER_DEVICE_TRAIN_BATCH_SIZE
#                         Batch size per GPU/TPU core/CPU for training. (default: 8)
#   --per_device_eval_batch_size PER_DEVICE_EVAL_BATCH_SIZE
#                         Batch size per GPU/TPU core/CPU for evaluation. (default: 8)
#   --per_gpu_train_batch_size PER_GPU_TRAIN_BATCH_SIZE
#                         Deprecated, the use of `--per_device_train_batch_size` is preferred. Batch size per GPU/TPU core/CPU for training. (default: None)
#   --per_gpu_eval_batch_size PER_GPU_EVAL_BATCH_SIZE
#                         Deprecated, the use of `--per_device_eval_batch_size` is preferred. Batch size per GPU/TPU core/CPU for evaluation. (default: None)
#   --gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS
#                         Number of updates steps to accumulate before performing a backward/update pass. (default: 1)
#   --eval_accumulation_steps EVAL_ACCUMULATION_STEPS
#                         Number of predictions steps to accumulate before moving the tensors to the CPU. (default: None)
#   --eval_delay EVAL_DELAY
#                         Number of epochs or steps to wait for before the first evaluation can be performed, depending on the evaluation_strategy. (default: 0)
#   --learning_rate LEARNING_RATE
#                         The initial learning rate for AdamW. (default: 5e-05)
#   --weight_decay WEIGHT_DECAY
#                         Weight decay for AdamW if we apply some. (default: 0.0)
#   --adam_beta1 ADAM_BETA1
#                         Beta1 for AdamW optimizer (default: 0.9)
#   --adam_beta2 ADAM_BETA2
#                         Beta2 for AdamW optimizer (default: 0.999)
#   --adam_epsilon ADAM_EPSILON
#                         Epsilon for AdamW optimizer. (default: 1e-08)
#   --max_grad_norm MAX_GRAD_NORM
#                         Max gradient norm. (default: 1.0)
#   --num_train_epochs NUM_TRAIN_EPOCHS
#                         Total number of training epochs to perform. (default: 3.0)
#   --max_steps MAX_STEPS
#                         If > 0: set total number of training steps to perform. Override num_train_epochs. (default: -1)
#   --lr_scheduler_type {linear,cosine,cosine_with_restarts,polynomial,constant,constant_with_warmup,inverse_sqrt}
#                         The scheduler type to use. (default: linear)
#   --warmup_ratio WARMUP_RATIO
#                         Linear warmup over warmup_ratio fraction of total steps. (default: 0.0)
#   --warmup_steps WARMUP_STEPS
#                         Linear warmup over warmup_steps. (default: 0)
#   --log_level {debug,info,warning,error,critical,passive}
#                         Logger log level to use on the main node. Possible choices are the log levels as strings: 'debug', 'info', 'warning', 'error' and 'critical', plus a
#                         'passive' level which doesn't set anything and lets the application set the level. Defaults to 'passive'. (default: passive)
#   --log_level_replica {debug,info,warning,error,critical,passive}
#                         Logger log level to use on replica nodes. Same choices and defaults as ``log_level`` (default: warning)
#   --log_on_each_node [LOG_ON_EACH_NODE]
#                         When doing a multinode distributed training, whether to log once per node or just once on the main node. (default: True)
#   --no_log_on_each_node
#                         When doing a multinode distributed training, whether to log once per node or just once on the main node. (default: False)
#   --logging_dir LOGGING_DIR
#                         Tensorboard log dir. (default: None)
#   --logging_strategy {no,steps,epoch}
#                         The logging strategy to use. (default: steps)
#   --logging_first_step [LOGGING_FIRST_STEP]
#                         Log the first global_step (default: False)
#   --logging_steps LOGGING_STEPS
#                         Log every X updates steps. (default: 500)
#   --logging_nan_inf_filter [LOGGING_NAN_INF_FILTER]
#                         Filter nan and inf losses for logging. (default: True)
#   --no_logging_nan_inf_filter
#                         Filter nan and inf losses for logging. (default: False)
#   --save_strategy {no,steps,epoch}
#                         The checkpoint save strategy to use. (default: steps)
#   --save_steps SAVE_STEPS
#                         Save checkpoint every X updates steps. (default: 500)
#   --save_total_limit SAVE_TOTAL_LIMIT
#                         Limit the total amount of checkpoints. Deletes the older checkpoints in the output_dir. Default is unlimited checkpoints (default: None)
#   --save_on_each_node [SAVE_ON_EACH_NODE]
#                         When doing multi-node distributed training, whether to save models and checkpoints on each node, or only on the main one (default: False)
#   --no_cuda [NO_CUDA]   Do not use CUDA even when it is available (default: False)
#   --use_mps_device [USE_MPS_DEVICE]
#                         Whether to use Apple Silicon chip based `mps` device. (default: False)
#   --seed SEED           Random seed that will be set at the beginning of training. (default: 42)
#   --data_seed DATA_SEED
#                         Random seed to be used with data samplers. (default: None)
#   --jit_mode_eval [JIT_MODE_EVAL]
#                         Whether or not to use PyTorch jit trace for inference (default: False)
#   --use_ipex [USE_IPEX]
#                         Use Intel extension for PyTorch when it is available, installation: 'https://github.com/intel/intel-extension-for-pytorch' (default: False)
#   --bf16 [BF16]         Whether to use bf16 (mixed) precision instead of 32-bit. Requires Ampere or higher NVIDIA architecture or using CPU (no_cuda). This is an experimental API
#                         and it may change. (default: False)
#   --fp16 [FP16]         Whether to use fp16 (mixed) precision instead of 32-bit (default: False)
#   --fp16_opt_level FP16_OPT_LEVEL
#                         For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']. See details at https://nvidia.github.io/apex/amp.html (default: O1)
#   --half_precision_backend {auto,cuda_amp,apex,cpu_amp}
#                         The backend to be used for half precision. (default: auto)
#   --bf16_full_eval [BF16_FULL_EVAL]
#                         Whether to use full bfloat16 evaluation instead of 32-bit. This is an experimental API and it may change. (default: False)
#   --fp16_full_eval [FP16_FULL_EVAL]
#                         Whether to use full float16 evaluation instead of 32-bit (default: False)
#   --tf32 TF32           Whether to enable tf32 mode, available in Ampere and newer GPU architectures. This is an experimental API and it may change. (default: None)
#   --local_rank LOCAL_RANK
#                         For distributed training: local_rank (default: -1)
#   --xpu_backend {mpi,ccl,gloo}
#                         The backend to be used for distributed training on Intel XPU. (default: None)
#   --tpu_num_cores TPU_NUM_CORES
#                         TPU: Number of TPU cores (automatically passed by launcher script) (default: None)
#   --tpu_metrics_debug [TPU_METRICS_DEBUG]
#                         Deprecated, the use of `--debug tpu_metrics_debug` is preferred. TPU: Whether to print debug metrics (default: False)
#   --debug DEBUG         Whether or not to enable debug mode. Current options: `underflow_overflow` (Detect underflow and overflow in activations and weights), `tpu_metrics_debug`
#                         (print debug metrics on TPU). (default: )
#   --dataloader_drop_last [DATALOADER_DROP_LAST]
#                         Drop the last incomplete batch if it is not divisible by the batch size. (default: False)
#   --eval_steps EVAL_STEPS
#                         Run an evaluation every X steps. (default: None)
#   --dataloader_num_workers DATALOADER_NUM_WORKERS
#                         Number of subprocesses to use for data loading (PyTorch only). 0 means that the data will be loaded in the main process. (default: 0)
#   --past_index PAST_INDEX
#                         If >=0, uses the corresponding part of the output as the past state for next step. (default: -1)
#   --run_name RUN_NAME   An optional descriptor for the run. Notably used for wandb logging. (default: None)
#   --disable_tqdm DISABLE_TQDM
#                         Whether or not to disable the tqdm progress bars. (default: None)
#   --remove_unused_columns [REMOVE_UNUSED_COLUMNS]
#                         Remove columns not required by the model when using an nlp.Dataset. (default: True)
#   --no_remove_unused_columns
#                         Remove columns not required by the model when using an nlp.Dataset. (default: False)
#   --label_names LABEL_NAMES [LABEL_NAMES ...]
#                         The list of keys in your dictionary of inputs that correspond to the labels. (default: None)
#   --load_best_model_at_end [LOAD_BEST_MODEL_AT_END]
#                         Whether or not to load the best model found during training at the end of training. (default: False)
#   --metric_for_best_model METRIC_FOR_BEST_MODEL
#                         The metric to use to compare two different models. (default: None)
#   --greater_is_better GREATER_IS_BETTER
#                         Whether the `metric_for_best_model` should be maximized or not. (default: None)
#   --ignore_data_skip [IGNORE_DATA_SKIP]
#                         When resuming training, whether or not to skip the first epochs and batches to get to the same training data. (default: False)
#   --sharded_ddp SHARDED_DDP
#                         Whether or not to use sharded DDP training (in distributed training only). The base option should be `simple`, `zero_dp_2` or `zero_dp_3` and you can add
#                         CPU-offload to `zero_dp_2` or `zero_dp_3` like this: zero_dp_2 offload` or `zero_dp_3 offload`. You can add auto-wrap to `zero_dp_2` or `zero_dp_3` with the
#                         same syntax: zero_dp_2 auto_wrap` or `zero_dp_3 auto_wrap`. (default: )
#   --fsdp FSDP           Whether or not to use PyTorch Fully Sharded Data Parallel (FSDP) training (in distributed training only). The base option should be `full_shard`,
#                         `shard_grad_op` or `no_shard` and you can add CPU-offload to `full_shard` or `shard_grad_op` like this: full_shard offload` or `shard_grad_op offload`. You
#                         can add auto-wrap to `full_shard` or `shard_grad_op` with the same syntax: full_shard auto_wrap` or `shard_grad_op auto_wrap`. (default: )
#   --fsdp_min_num_params FSDP_MIN_NUM_PARAMS
#                         This parameter is deprecated. FSDP's minimum number of parameters for Default Auto Wrapping. (useful only when `fsdp` field is passed). (default: 0)
#   --fsdp_config FSDP_CONFIG
#                         Config to be used with FSDP (Pytorch Fully Sharded Data Parallel). The value is either afsdp json config file (e.g., `fsdp_config.json`) or an already loaded
#                         json file as `dict`. (default: None)
#   --fsdp_transformer_layer_cls_to_wrap FSDP_TRANSFORMER_LAYER_CLS_TO_WRAP
#                         This parameter is deprecated. Transformer layer class name (case-sensitive) to wrap, e.g, `BertLayer`, `GPTJBlock`, `T5Block` .... (useful only when `fsdp`
#                         flag is passed). (default: None)
#   --deepspeed DEEPSPEED
#                         Enable deepspeed and pass the path to deepspeed json config file (e.g. ds_config.json) or an already loaded json file as a dict (default: None)
#   --label_smoothing_factor LABEL_SMOOTHING_FACTOR
#                         The label smoothing epsilon to apply (zero means no label smoothing). (default: 0.0)
#   --optim OPTIM
#   --optim_args OPTIM_ARGS
#                         Optional arguments to supply to optimizer. (default: None)
#   --adafactor [ADAFACTOR]
#                         Whether or not to replace AdamW by Adafactor. (default: False)
#   --group_by_length [GROUP_BY_LENGTH]
#                         Whether or not to group samples of roughly the same length together when batching. (default: False)
#   --length_column_name LENGTH_COLUMN_NAME
#                         Column name with precomputed lengths to use when grouping by length. (default: length)
#   --report_to REPORT_TO [REPORT_TO ...]
#                         The list of integrations to report the results and logs to. (default: None)
#   --ddp_find_unused_parameters DDP_FIND_UNUSED_PARAMETERS
#                         When using distributed training, the value of the flag `find_unused_parameters` passed to `DistributedDataParallel`. (default: None)
#   --ddp_bucket_cap_mb DDP_BUCKET_CAP_MB
#                         When using distributed training, the value of the flag `bucket_cap_mb` passed to `DistributedDataParallel`. (default: None)
#   --dataloader_pin_memory [DATALOADER_PIN_MEMORY]
#                         Whether or not to pin memory for DataLoader. (default: True)
#   --no_dataloader_pin_memory
#                         Whether or not to pin memory for DataLoader. (default: False)
#   --skip_memory_metrics [SKIP_MEMORY_METRICS]
#                         Whether or not to skip adding of memory profiler reports to metrics. (default: True)
#   --no_skip_memory_metrics
#                         Whether or not to skip adding of memory profiler reports to metrics. (default: False)
#   --use_legacy_prediction_loop [USE_LEGACY_PREDICTION_LOOP]
#                         Whether or not to use the legacy prediction_loop in the Trainer. (default: False)
#   --push_to_hub [PUSH_TO_HUB]
#                         Whether or not to upload the trained model to the model hub after training. (default: False)
#   --resume_from_checkpoint RESUME_FROM_CHECKPOINT
#                         The path to a folder with a valid checkpoint for your model. (default: None)
#   --hub_model_id HUB_MODEL_ID
#                         The name of the repository to keep in sync with the local `output_dir`. (default: None)
#   --hub_strategy {end,every_save,checkpoint,all_checkpoints}
#                         The hub strategy to use when `--push_to_hub` is activated. (default: every_save)
#   --hub_token HUB_TOKEN
#                         The token to use to push to the Model Hub. (default: None)
#   --hub_private_repo [HUB_PRIVATE_REPO]
#                         Whether the model repository is private or not. (default: False)
#   --gradient_checkpointing [GRADIENT_CHECKPOINTING]
#                         If True, use gradient checkpointing to save memory at the expense of slower backward pass. (default: False)
#   --include_inputs_for_metrics [INCLUDE_INPUTS_FOR_METRICS]
#                         Whether or not the inputs will be passed to the `compute_metrics` function. (default: False)
#   --fp16_backend {auto,cuda_amp,apex,cpu_amp}
#                         Deprecated. Use half_precision_backend instead (default: auto)
#   --push_to_hub_model_id PUSH_TO_HUB_MODEL_ID
#                         The name of the repository to which push the `Trainer`. (default: None)
#   --push_to_hub_organization PUSH_TO_HUB_ORGANIZATION
#                         The name of the organization in with to which push the `Trainer`. (default: None)
#   --push_to_hub_token PUSH_TO_HUB_TOKEN
#                         The token to use to push to the Model Hub. (default: None)
#   --mp_parameters MP_PARAMETERS
#                         Used by the SageMaker launcher to send mp-specific args. Ignored in Trainer (default: )
#   --auto_find_batch_size [AUTO_FIND_BATCH_SIZE]
#                         Whether to automatically decrease the batch size in half and rerun the training loop again each time a CUDA Out-of-Memory was reached (default: False)
#   --full_determinism [FULL_DETERMINISM]
#                         Whether to call enable_full_determinism instead of set_seed for reproducibility in distributed training. Important: this will negatively impact the
#                         performance, so only use it for debugging. (default: False)
#   --torchdynamo TORCHDYNAMO
#                         This argument is deprecated, use `--torch_compile_backend` instead. (default: None)
#   --ray_scope RAY_SCOPE
#                         The scope to use when doing hyperparameter search with Ray. By default, `"last"` will be used. Ray will then use the last checkpoint of all trials, compare
#                         those, and select the best one. However, other options are also available. See the Ray documentation
#                         (https://docs.ray.io/en/latest/tune/api_docs/analysis.html#ray.tune.ExperimentAnalysis.get_best_trial) for more options. (default: last)
#   --ddp_timeout DDP_TIMEOUT
#                         Overrides the default timeout for distributed training (value should be given in seconds). (default: 1800)
#   --torch_compile [TORCH_COMPILE]
#                         If set to `True`, the model will be wrapped in `torch.compile`. (default: False)
#   --torch_compile_backend TORCH_COMPILE_BACKEND
#                         Which backend to use with `torch.compile`, passing one will trigger a model compilation. (default: None)
#   --torch_compile_mode TORCH_COMPILE_MODE
#                         Which mode to use with `torch.compile`, passing one will trigger a model compilation. (default: None)
#   --cache_dir CACHE_DIR
#   --model_max_length MODEL_MAX_LENGTH
#                         Maximum sequence length. Sequences will be right padded (and possibly truncated). (default: 512)