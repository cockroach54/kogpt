#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import copy
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence

import torch
import transformers
from torch.utils.data import Dataset
from transformers import Trainer
import random
import utils

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "[EOS]"
DEFAULT_BOS_TOKEN = "[BOS]"
DEFAULT_UNK_TOKEN = "[UNK]"
# PROMPT_DICT = {
#     "prompt_input": (
#         "Below is an instruction that describes a task, paired with an input that provides further context.\n"
#         "아래는 작업을 설명하는 명령어와 추가적 맥락을 제공하는 입력이 짝을 이루는 예제입니다.\n\n"
#         "Write a response that appropriately completes the request.\n요청을 적절히 완료하는 응답을 작성하세요.\n\n"
#         "### Instruction(명령어):\n{instruction}\n\n### Input(입력):\n{input}\n\n### Response(응답):"
#     ),
#     "prompt_no_input": (
#         "Below is an instruction that describes a task.\n"
#         "아래는 작업을 설명하는 명령어입니다.\n\n"
#         "Write a response that appropriately completes the request.\n명령어에 따른 요청을 적절히 완료하는 응답을 작성하세요.\n\n"
#         "### Instruction(명령어):\n{instruction}\n\n### Response(응답):"
#     ),
# }

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

# lsw
# tokenizer = transformers.AutoTokenizer.from_pretrained(
#     "kakaobrain/kogpt",
# #     revision="KoGPT6B-ryan1.5b",
#     revision="KoGPT6B-ryan1.5b-float16",  # or float32 version: revision=KoGPT6B-ryan1.5b
#     bos_token="[BOS]",
#     eos_token="[EOS]",
#     unk_token="[UNK]",
#     pad_token="[PAD]",
#     mask_token="[MASK]",
#     # cache_dir=training_args.cache_dir,
# )
tokenizer = transformers.AutoTokenizer.from_pretrained("EleutherAI/polyglot-ko-12.8b")


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    #     optim: str = field(default="adamw_torch")
    optim: str = field(default="adafactor")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


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
        for text in strings
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
        list_data_dict = utils.jload(data_path)
        #         list_data_dict = utils.jload(data_path)[:50] + utils.jload(data_path)[-50:]
        # lsw shuffle data
        random.shuffle(list_data_dict)
        #         print(list_data_dict)
        # print('-----')

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


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels")
        )
        # print("[input_ids]:", tokenizer.decode(input_ids))
        # print("[labels]:", labels)
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


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(
        tokenizer=tokenizer, data_path=data_args.data_path
    )
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(
        train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
    )


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # lsw for param debug
    # print(model_args.__dict__)
    # print("---")
    # print(data_args.__dict__)
    # print("---")
    # print(training_args.__dict__)
    # exit()

    # model = transformers.AutoModelForCausalLM.from_pretrained(
    #     model_args.model_name_or_path,
    #     cache_dir=training_args.cache_dir,
    # )

    # tokenizer = transformers.AutoTokenizer.from_pretrained(
    #     model_args.model_name_or_path,
    #     cache_dir=training_args.cache_dir,
    #     model_max_length=training_args.model_max_length,
    #     padding_side="right",
    #     use_fast=False,
    # )

    # # lsw
    # tokenizer = transformers.AutoTokenizer.from_pretrained(
    #     "kakaobrain/kogpt",
    #     revision="KoGPT6B-ryan1.5b-float16",  # or float32 version: revision=KoGPT6B-ryan1.5b
    #     bos_token="[BOS]",
    #     eos_token="[EOS]",
    #     unk_token="[UNK]",
    #     pad_token="[PAD]",
    #     mask_token="[MASK]",
    #     cache_dir=training_args.cache_dir,
    # )
    tokenizer.model_max_length = training_args.model_max_length
    #     model = transformers.AutoModelForCausalLM.from_pretrained(
    #         "kakaobrain/kogpt",
    # #         revision="KoGPT6B-ryan1.5b",
    #         revision="KoGPT6B-ryan1.5b-float16",  # or float32 version: revision=KoGPT6B-ryan1.5b
    #         pad_token_id=tokenizer.pad_token_id,
    #         torch_dtype=torch.float16,
    #         # torch_dtype="auto",
    #         low_cpu_mem_usage=True,
    #         # -------
    #         cache_dir=training_args.cache_dir,
    #         # model_max_length=training_args.model_max_length,
    #         # padding_side="right",
    #         # use_fast=False,
    #     )
    model = transformers.AutoModelForCausalLM.from_pretrained(
        "EleutherAI/polyglot-ko-12.8b",
        pad_token_id=tokenizer.pad_token_id,
        torch_dtype=torch.float16,
        # torch_dtype="auto",
        low_cpu_mem_usage=False,
        # -------
        cache_dir=training_args.cache_dir,
    )

    # if tokenizer.pad_token is None:
    #     smart_tokenizer_and_embedding_resize(
    #         special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
    #         tokenizer=tokenizer,
    #         model=model,
    #     )
    # if "llama" in model_args.model_name_or_path:
    #     tokenizer.add_special_tokens(
    #         {
    #             "eos_token": DEFAULT_EOS_TOKEN,
    #             "bos_token": DEFAULT_BOS_TOKEN,
    #             "unk_token": DEFAULT_UNK_TOKEN,
    #         }
    #     )

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    trainer = Trainer(
        model=model, tokenizer=tokenizer, args=training_args, **data_module
    )
    trainer.train()
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    import os

    # os.environ["WANDB_DISABLED"] = "true"
    train()
