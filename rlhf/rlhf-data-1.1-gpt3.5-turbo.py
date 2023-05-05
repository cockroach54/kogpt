import os
import openai
import json
from typing import List
from tqdm.auto import tqdm
from time import sleep
from typing import List, Any, Union

def load_jsonl(fpath:str) -> List[Union[dict, list]]:
    with open(fpath, 'r') as f:
        data = f.readlines()
    data = [json.loads(x) for x in data]
    return data


realpath = os.path.dirname(os.path.realpath(__file__))

OPENAI_API_KEY = None

# set openai api key
if OPENAI_API_KEY is None:
    _key = os.environ.get("OPENAI_API_KEY")
    if _key is None:
        raise Exception("OPENAI_API_KEY 를 입력하지 않았고 시스템 환경변수도 존재하지 않습니다.")
    else:
        OPENAI_API_KEY = _key
openai.api_key = OPENAI_API_KEY

# with open('./ko_alpaca_data.json', 'r') as f:
#     data = json.load(f)

fpath = '../KoAlpaca_v1.1.jsonl'
data = load_jsonl(fpath)


PROMPT_DICT = {
    "prompt_input": (
        "### Instruction:\n{instruction}\n\n### Input:\n{user_input}\n\n"
    ),
    "prompt_no_input": (
        "### Instruction:\n{instruction}\n\n"
    ),
}

SYS_COMMAND = (
    "Below is an instruction that describes a task, Depending on each instruction, additional context may or may not be provided.\n"
    "Write a long detailed response that appropriately completes the each request.\n"
    "Please answer questions written in Korean in Korean and questions written in English in English.\n"
    "In answer text, do not rewrite the question or additional context, and write only the pure content of the answer without any formatting.\n"
    "For each answer, enter two newlines and '>>>' at the beginning of answer.\n\n"
)


def make_prompt(dl:List[dict], start_idx=0):
    ret = ''
    for i, d in enumerate(dl):
        prompt, user_input, chatgpt_outpout = d.get('instruction', ''), d.get('input', ''), d.get('output', '')
        if user_input:
            x = PROMPT_DICT['prompt_input'].format(instruction=prompt, user_input=user_input)
        else:
            x = PROMPT_DICT['prompt_no_input'].format(instruction=prompt)
        ret = ret + f">>>{start_idx+i+1}\n{x}\n\n"
    return SYS_COMMAND+ret

# make exist file as backup file
dfile_name = "rlhf-data-1.1-gpt-3.5-turbo.txt"
backup_name = "rlhf-data-1.1-gpt-3.5-turbo.old.txt"
if os.path.exists(dfile_name):
    if os.path.exists(backup_name):
        raise Exception(f'백업파일({backup_name}) 이 이미 존재합니다. 기존 백업 파일명을 변경해주세요')
    os.rename(dfile_name, backup_name)


batch_size = 20
for i in tqdm(range(0, len(data), batch_size)):
    st = i
    end = i+batch_size
    if i > len(data):
        end = len(data)
        
    prompt = make_prompt(data[st:end], start_idx=st)
    history = [
        {
            "role": "system",
            "content": "You are a helpful assistant",
        },
        {"role": "user", "content": prompt}
    ]
    
    try:
        response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=history)
    except Exception as e:
        done = False
        print("openai api error... retry...")
        while not done:
            sleep(5)
            try:
                response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=history)
                done = True
            except Exception as e:
                pass

    rr = response["choices"][0]["message"]["content"]        
    with open(dfile_name, 'a') as f:
        f.write(rr+'\n\n')

