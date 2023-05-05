./t3.sh | tee kogpt-ft-train-230406.log

./t4.sh | tee kogpt-lora-train-230406.log

alias gitlog="git log --decorate --oneline --graph --all"

git config --global user.email "lsw0504@naver.com" &&\
git config --global user.name "cockroach54"

---
fine tuning의 경우는 내부에  model-pallalism 이 적용되어있는듯 
rola의 경우는 내부에  model-pallalism 이 적용 안된듯...  여러 GPU 사용시 에러남

return torch.embedding(weight, input, padding_idx, scale_grad_by_freq, sparse)
RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cuda:1! (when checking argument for argument index in method wrapper_CUDA__index_select)

---
16bit 모델(half)은 gpu는 문제없는데 cpu는 에러남

RuntimeError: "LayerNormKernelImpl" not implemented for 'Half'

lora 모델 체크포인트 로드는 백본이 프리징되어 학습된 파라메터가 없어서인지 train시에  trainer.train(resume_from_checkpoint = './lora/checkpoint-4000') 옵션 쓰면 워닝 메시지 띄우고 그냥 처음부터 진행됨