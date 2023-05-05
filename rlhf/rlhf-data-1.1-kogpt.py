# https://huggingface.co/docs/transformers/model_doc/gptj

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from time import time

DEVICE = "cuda"

# tokenizer = AutoTokenizer.from_pretrained("beomi/KoAlpaca-Polyglot-12.8B")
# model = AutoModelForCausalLM.from_pretrained("beomi/KoAlpaca-Polyglot-12.8B",
#                                             pad_token_id=tokenizer.pad_token_id,
#                                             eos_token_id=tokenizer.eos_token_id,
#                                             low_cpu_mem_usage=True,
# #                                             torch_dtype='auto'
#                                             torch_dtype=torch.float16
#                                             ).to(DEVICE)

tokenizer = AutoTokenizer.from_pretrained(
    "kakaobrain/kogpt",
    revision="KoGPT6B-ryan1.5b-float16",  # or float32 version: revision=KoGPT6B-ryan1.5b
    bos_token="[BOS]",
    eos_token="[EOS]",
    unk_token="[UNK]",
    pad_token="[PAD]",
    mask_token="[MASK]",
)
model = AutoModelForCausalLM.from_pretrained(
    './kogpt-ft-3',
#     "kakaobrain/kogpt",
    revision="KoGPT6B-ryan1.5b-float16",  # or float32 version: revision=KoGPT6B-ryan1.5b
    pad_token_id=tokenizer.pad_token_id,
    torch_dtype=torch.float16,
    # torch_dtype='auto',
    low_cpu_mem_usage=True,
).to(DEVICE)
_ = model.eval()

# PROMPT_DICT = {
#     "prompt_input": (
#         "Below is an instruction that describes a task, paired with an input that provides further context.\n"
#         "아래는 작업을 설명하는 명령어와 추가적 맥락을 제공하는 입력이 짝을 이루는 예제입니다.\n\n"
#         "Write a detailed response that appropriately completes the request.\n요청을 적절히 완료하는 응답을 자세하게 작성하세요.\n\n"
#         "### Instruction(명령어):\n{instruction}\n\n### Input(입력):\n{user_input}\n\n### Response(응답):"
#     ),
#     "prompt_no_input": (
#         "Below is an instruction that describes a task.\n"
#         "아래는 작업을 설명하는 명령어입니다.\n\n"
#         "Write a detailed response that appropriately completes the request.\n요청을 적절히 완료하는 응답을 자세하게 작성하세요.\n\n"
#         "### Instruction(명령어):\n{instruction}\n\n### Response(응답):"
#     ),
# }


PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context.\n\n"
        "Write a detailed response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{user_input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task.\n\n"
        "Write a detailed response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

def gen(prompt, user_input=None, min_new_tokens=10, max_new_tokens=1024, temperature=0.5):
    st = time()
    if user_input:
        x = PROMPT_DICT['prompt_input'].format(instruction=prompt, user_input=user_input)
    else:
        x = PROMPT_DICT['prompt_no_input'].format(instruction=prompt)
    
    input_ids = tokenizer.encode(x, return_tensors="pt").to(DEVICE)
    gen_tokens = model.generate(
        input_ids, 
        max_new_tokens=max_new_tokens,
        num_return_sequences=1, 
        temperature=temperature,
        no_repeat_ngram_size=6,
        do_sample=True,
        
    )
    gen_text = tokenizer.decode(gen_tokens[0], skip_special_tokens=True)
    end = time()
#     print(f"[Elpsed]: {end-st} sec")
    
    return x, gen_text.replace(x, '')


import json
from tqdm.auto import tqdm
from typing import List, Any, Union

def load_jsonl(fpath:str) -> List[Union[dict, list]]:
    with open(fpath, 'r') as f:
        data = f.readlines()
    data = [json.loads(x) for x in data]
    return data

# for 1.0 data
# with open('./ko_alpaca_data.json', 'r') as f:
#     data = json.load(f)

fpath = '../KoAlpaca_v1.1.jsonl'
data = load_jsonl(fpath)

new_data = []
for d in tqdm(data):
    res = {}
    prompt, user_input, chatgpt_outpout = d.get('instruction', ''), d.get('input', ''), d.get('output', '')
    res.update({'instruction': prompt, 'input': user_input, 'output': [chatgpt_outpout]})
    if user_input == '':
        user_input = None
    for i in range(2):
        final_prompt, generated_ouput = gen(prompt, user_input=user_input, max_new_tokens=1024, temperature=1.5)
        res['output'].append(generated_ouput.strip())
    new_data.append(res)
#     sleep(0.5)

    if len(new_data) == 50:
        with open('./rlhf-data-1.1.jsonl', 'a') as f:
            print('write data:', len(new_data))
            dl = [json.dumps(x, ensure_ascii=False) for x in new_data]
            f.write('\n'.join(dl) + '\n')
            new_data = []
            