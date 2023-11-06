#!/usr/bin/env python
# coding: utf-8

import warnings

warnings.filterwarnings("ignore")


# ======== get model and tokenizer =========
import torch
import json
import deepspeed
from transformers import GPTNeoForCausalLM, AutoTokenizer, AutoModelForCausalLM

torch.distributed.init_process_group(backend="nccl")
deepspeed.init_distributed("nccl")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


tokenizer = AutoTokenizer.from_pretrained("EleutherAI/polyglot-ko-5.8b")
tokenizer.model_max_length = 512

print(tokenizer)


# ---------------------

model = AutoModelForCausalLM.from_pretrained(
    "EleutherAI/polyglot-ko-5.8b",
    pad_token_id=tokenizer.pad_token_id,
    torch_dtype=torch.float16,
    # torch_dtype="auto",
    low_cpu_mem_usage=True,
).to(device)

# ======== get dataset =========
IGNORE_INDEX = -100

from dataclasses import dataclass
import torch
from torch.utils.data import Dataset
import os
import json
from tqdm.auto import tqdm
from time import sleep
from typing import List, Any, Union, Dict, Sequence
import transformers
import logging
import random
import copy

PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context.\n\n"
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task.\n\n"
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}


def load_jsonl(fpath: str) -> List[Union[dict, list]]:
    ext = os.path.splitext(fpath)[-1]
    if ext == ".json":
        with open(fpath, "r") as f:
            data = json.load(f)
    elif ext == ".jsonl":
        with open(fpath, "r") as f:
            data = f.readlines()
        data = [json.loads(x) for x in data]
    else:
        raise Exception(f'file extension({ext}) must be one of ["json", "jsonl"]')
    return data


def _tokenize_fn(
    strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer
) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in tqdm(strings)
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [
        _tokenize_fn(strings, tokenizer) for strings in (examples, sources)
    ]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        list_data_dict = load_jsonl(data_path)
        # lsw shuffle data
        random.shuffle(list_data_dict)

        logging.warning("Formatting inputs...")
        prompt_input, prompt_no_input = (
            PROMPT_DICT["prompt_input"],
            PROMPT_DICT["prompt_no_input"],
        )
        sources = [
            prompt_input.format_map(example)
            if example.get("input", "") != ""
            else prompt_no_input.format_map(example)
            for example in list_data_dict
        ]
        targets = [
            f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict
        ]

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


def collate_data(
    input_ids: List[torch.Tensor],
    labels: List[torch.Tensor],
    tokenizer: transformers.PreTrainedTokenizer,
):
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    labels = torch.nn.utils.rnn.pad_sequence(
        labels, batch_first=True, padding_value=IGNORE_INDEX
    )
    attention_masks = input_ids.ne(tokenizer.pad_token_id)
    return dict(input_ids=input_ids, labels=labels, attention_masks=attention_masks)


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels")
        )
        print("[input_ids]:", input_ids)
        #         print("[input_ids]:", tokenizer.decode(input_ids))
        print("[labels]:", labels)
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        ret = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
        # print(ret)
        return ret


# for SFT
train_micro_batch_size_per_gpu = 3
gradient_accumulation_steps = 11
n_gpu = 4
train_batch_size = train_micro_batch_size_per_gpu * gradient_accumulation_steps * n_gpu
BATCH_SIZE = train_micro_batch_size_per_gpu * n_gpu
EPOCHS = 2
#  ====================== deepspeed train ===================

config = {
    "fp16": {
        "enabled": True,
        "auto_cast": False,
        "loss_scale": 0,
        "initial_scale_power": 16,
        "loss_scale_window": 1000,
        "hysteresis": 2,
        "min_loss_scale": 1,
    },
    #     "fp16": {
    #         "enabled": False
    #     },
    "zero_optimization": {
        "stage": 2,
        "contiguous_gradients": True,
        "overlap_comm": True,
        "offload_optimizer": {"device": "cpu", "pin_memory": True, "fast_init": True},
    },
    #      "scheduler": {
    #           "type": "WarmupLR",
    #           "params": {
    #               "warmup_min_lr": 0,
    #               "warmup_max_lr": 0.1,
    #               "warmup_num_steps": 500
    #           }
    #       },
    #     "optimizer": {
    #         "type": "Adam",
    #         "params": {"lr": 5e-07, "betas": [0.9, 0.999], "eps": 1e-8},
    #     },
    "scheduler": {
        "type": "WarmupDecayLR",
        "params": {
            #               "total_num_steps": 14302,
            "total_num_steps": 20000,
            "warmup_min_lr": 0,
            "warmup_max_lr": 5e-07,
            "warmup_num_steps": 500,
        },
    },
    "optimizer": {
        "type": "AdamW",
        "params": {"lr": 5e-07},
    },
    "train_batch_size": train_batch_size,
    "train_micro_batch_size_per_gpu": train_micro_batch_size_per_gpu,
    "gradient_accumulation_steps": gradient_accumulation_steps,
}

model, optimizer, _, _ = deepspeed.initialize(
    model=model, config_params=config, model_parameters=model.parameters()
)

train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path="../data/new/sft.json")
fin_iter = len(train_dataset) // BATCH_SIZE

losses = []
global_step = 0

for epoch in range(1, EPOCHS + 1):
    print(f"*********** epoch: {epoch} ***********")
    for step in tqdm(range(fin_iter + 1)):
        global_step += 1
        model.zero_grad()

        batch = train_dataset[step * BATCH_SIZE : (step + 1) * BATCH_SIZE]
        if len(batch) == 0:
            break

        # lsw for dev
        #     if step == 200:
        #         break

        batch_data = collate_data(batch["input_ids"], batch["labels"], tokenizer)

        input_ids = batch_data["input_ids"].to(device)
        attention_masks = batch_data["attention_masks"].to(device)
        labels = batch_data["labels"].to(device)

        outputs = model(
            input_ids=input_ids, attention_mask=attention_masks, labels=labels
        )
        loss = outputs.loss

        losses.append({"loss": loss.tolist(), "lr": optimizer.param_groups[0]["lr"]})
        print(
            f"[epoch/gstep]: {epoch}/{global_step}, [loss]: {loss.tolist()}, [lr]: {optimizer.param_groups[0]['lr']}, [input]: {input_ids.shape}"
        )

        model.backward(loss)
        model.step()

        if global_step % 20 == 0 and global_step > 0:
            with open("ds-res.json", "w") as f:
                json.dump(losses, f, indent=4)
        if global_step % 300 == 0 and global_step > 0:
            print("save model at global_step:", global_step)
            model.save_pretrained("./sft-poly6b")

with open("ds-res.json", "w") as f:
    json.dump(losses, f, indent=4)

# print(losses)
model.save_pretrained("./sft-poly6b")

print("********* train finished well :) *********")
