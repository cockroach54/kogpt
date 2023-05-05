# https://huggingface.co/docs/transformers/model_doc/gptj

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from time import time

DEVICE = "cuda"

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
    "kakaobrain/kogpt",
    revision="KoGPT6B-ryan1.5b-float16",  # or float32 version: revision=KoGPT6B-ryan1.5b
    pad_token_id=tokenizer.pad_token_id,
    torch_dtype=torch.float16,
    # torch_dtype="auto",
    low_cpu_mem_usage=True,
).to(device=DEVICE, non_blocking=True)
_ = model.eval()

prompt = "원숭이 엉덩이는 빨개. 빨가면 사과."
# prompt = """
# 제목 : 그녀는 예뻤다
# 내용 : 그녀는 내 옆으로 살포시 다가와 내 볼을 어루만지기 시작했다.
# 그리고, 그녀는 말했다. \'조금만 더 가까이 와줘..\' 나는 그대로 다가갈 수 밖에 없었다.
# """

# prompt = """
# 아래 질문처럼 입력된 구문에 대한 감정을 긍정, 부정, 중립 중 하나로 평가해줘.

# ### 예시 질문
# 1.입력: 아 더빙.. 진짜 짜증나네요 목소리
# 1.정답: 부정

# 2.입력: 흠...포스터보고 초딩영화줄....오버연기조차 가볍지 않구나
# 2.정답: 긍정

# 3.입력: 사이몬페그의 익살스런 연기가 돋보였던 영화!스파이더맨에서 늙어보이기만 했던 커스틴 던스트가 너무나도 이뻐보였다
# 3.정답: 
# """

prompt = "코스닥이 얼마까지 오를까?"

st = time()
with torch.no_grad():
    tokens = tokenizer.encode(prompt, return_tensors="pt").to(
        device=DEVICE, non_blocking=True
    )
    gen_tokens = model.generate(tokens, do_sample=True, temperature=0.8, max_length=512)
    generated = tokenizer.batch_decode(gen_tokens)[0]

end = time()
print(f"[Elpsed]: {end-st} sec")

print(generated)
