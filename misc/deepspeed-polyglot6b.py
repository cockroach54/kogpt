#!/usr/bin/env python
# coding: utf-8

import warnings

warnings.filterwarnings("ignore")


# ======== get model and tokenizer =========
import torch
import deepspeed
from transformers import GPTNeoForCausalLM, AutoTokenizer, AutoModelForCausalLM

torch.distributed.init_process_group(backend="nccl")
deepspeed.init_distributed("nccl")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# tokenizer = AutoTokenizer.from_pretrained(
#     "kakaobrain/kogpt",
#     #     revision="KoGPT6B-ryan1.5b",
#     revision="KoGPT6B-ryan1.5b-float16",  # or float32 version: revision=KoGPT6B-ryan1.5b
#     bos_token="[BOS]",
#     eos_token="[EOS]",
#     unk_token="[UNK]",
#     pad_token="[PAD]",
#     mask_token="[MASK]",
#     # cache_dir=training_args.cache_dir,
# )

# tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neo-2.7B')
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/polyglot-ko-5.8b")
print(tokenizer)


# IGNORE_INDEX = -100
# DEFAULT_PAD_TOKEN = "[PAD]"
# DEFAULT_EOS_TOKEN = "</s>"
# DEFAULT_BOS_TOKEN = "</s>"
# DEFAULT_UNK_TOKEN = "</s>"

# tokenizer = AutoTokenizer.from_pretrained(
#     "skt/kogpt2-base-v2",
#     padding_side="right",
#     model_max_length=512,
# )
# tokenizer.add_special_tokens(
#     {
#         "eos_token": DEFAULT_EOS_TOKEN,
#         "bos_token": DEFAULT_BOS_TOKEN,
#         "unk_token": DEFAULT_UNK_TOKEN,
#     }
# )
# tokenizer.pad_token = tokenizer.eos_token
# print(tokenizer)

# ---------------------

# model = AutoModelForCausalLM.from_pretrained(
#     "kakaobrain/kogpt",
#     #         revision="KoGPT6B-ryan1.5b",
#     revision="KoGPT6B-ryan1.5b-float16",  # or float32 version: revision=KoGPT6B-ryan1.5b
#     pad_token_id=tokenizer.pad_token_id,
#     torch_dtype=torch.float16,
#     # torch_dtype="auto",
#     low_cpu_mem_usage=True,
#     # -------
# #     cache_dir=training_args.cache_dir,
#     # model_max_length=training_args.model_max_length,
#     # padding_side="right",
#     # use_fast=False,
# ).to(device)

# model = GPTNeoForCausalLM.from_pretrained('EleutherAI/gpt-neo-2.7B').to(device)
model = AutoModelForCausalLM.from_pretrained(
    "EleutherAI/polyglot-ko-5.8b",
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
    torch_dtype=torch.float16,
).to(device=device, non_blocking=True)


# ======== get dataset =========
from datasets import load_dataset

block_size = 512
datasets = load_dataset(
    "text", data_files={"train": "oo.txt"}
)  # , "validation": 'oo.txt'})


def tokenize_function(examples):
    ret = tokenizer(
        examples["text"],
    )
    return ret


#     return {
#         'input_ids': ret['input_ids'],
#         'attention_mask': ret['attention_mask']
#     }


def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


tokenized_datasets = datasets.map(
    tokenize_function, batched=True, num_proc=4, remove_columns=["text"]
)

lm_datasets = tokenized_datasets.map(
    group_texts,
    batched=True,
    batch_size=1000,
    num_proc=4,
)


#  ====================== deepspeed train ===================

from dataclasses import dataclass


@dataclass
class Arg:
    num_workers = 1
    batch_size = 8
    lr = 1e-3
    num_epochs = 3
    gradient_clip_val = 5


args = Arg()


config = {
    "fp16": {
        "enabled": True,
    },
    "zero_optimization": {
        "stage": 2,
        "contiguous_gradients": True,
        "overlap_comm": True,
        "offload_optimizer": {"device": "cpu", "pin_memory": True, "fast_init": True},
    },
    "optimizer": {
        "type": "Adam",
        "params": {"lr": 3e-05, "betas": [0.9, 0.999], "eps": 1e-8},
    },
    #     "train_batch_size": 1,
    "train_micro_batch_size_per_gpu": 4,
    "gradient_accumulation_steps": 8,
}

# config = {
#     "fp16": {
#         "enabled": "auto",
#         "loss_scale": 0,
#         "loss_scale_window": 1000,
#         "initial_scale_power": 16,
#         "hysteresis": 2,
#         "min_loss_scale": 1
#     },
#     "optimizer": {
# #         "type": "Adam",
#         "params": {
#             "lr": "auto",
#             "betas": "auto",
#             "eps": "auto",
#             "weight_decay": "auto"
#         }
#     },
#     "scheduler": {
#         "type": "WarmupLR",
#         "params": {
#             "warmup_min_lr": "auto",
#             "warmup_max_lr": "auto",
#             "warmup_num_steps": "auto"
#         }
#     },
#     "zero_optimization": {
#         "stage": 3,
#         "overlap_comm": True,
# #         "contiguous_gradients": True,
#         "sub_group_size": 1e9,
#         "reduce_bucket_size": "auto",
#         "stage3_prefetch_bucket_size": "auto",
#         "stage3_param_persistence_threshold": "auto",
#         "stage3_max_live_parameters": 1e9,
#         "stage3_max_reuse_distance": 1e9,
#         "stage3_gather_16bit_weights_on_model_save": True
#     },
#     "gradient_accumulation_steps": 8,
# #     "gradient_clipping": "auto",
# #     "steps_per_print": 2000,
# #     "train_batch_size": 8,
#     "train_micro_batch_size_per_gpu": 1,
#     "wall_clock_breakdown": False
# }

# # kogpt는 파라메터가 contiguous 하지 않아 아래 코드 필요
# for param in model.parameters():
#     param.data = param.data.contiguous()

model, optimizer, _, _ = deepspeed.initialize(
    model=model, config_params=config, model_parameters=model.parameters()
)

losses = []
for step, batch in enumerate(lm_datasets["train"]):
    input_ids, attention_masks, labels = (
        batch["input_ids"],
        batch["attention_mask"],
        batch["labels"],
    )

    #     input_ids = torch.tensor(input_ids).to(device)
    #     attention_masks = torch.tensor(attention_masks).to(device)
    #     labels = torch.tensor(labels).to(device)

    # polyglot5.8B 부터는 데이터를 배치로 받아야 함 (2d)
    input_ids = torch.tensor(input_ids).to(device).reshape(1, -1)
    attention_masks = torch.tensor(attention_masks).to(device).reshape(1, -1)
    labels = torch.tensor(labels).to(device).reshape(1, -1)

    outputs = model(input_ids=input_ids, attention_mask=attention_masks, labels=labels)
    loss = outputs.loss
    print(f"[loss]: {loss.tolist()}")
    losses.append(loss.tolist())

    model.backward(loss)
    model.step()

# print(losses)
model.save_pretrained("./obgpt")
