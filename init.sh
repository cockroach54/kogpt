# 클라우드 인스턴스 올린 후 초기 셋팅

# 시스템 유틸 설치
apt -y install vim htop tmux net-tools ninja-build build-essential libxml2 libaio-dev

# 프로젝트 코드 받아오기
# git clone https://github.com/cockroach54/kogpt.git
cd kogpt
pip install -r requirements.txt

# 쉘스크립트 모드 변경
chmod 775 *.sh

# kogpt 모델 받기
python kogpt.py

# # polyglot 모델 받기
# python kopoly.py

# 
chmod 775 *.sh
alias gitlog='git log --decorate --oneline --graph --all'
echo "alias gitlog='git log --decorate --oneline --graph --all'" >> ~/.bashrc

# for polyglot 12.8b trianing
chmod 775 ./train_v1.1b/*.sh
cp ./KoAlpaca_v1.1.jsonl ./train_v1.1b/data/KoAlpaca_v1.1.json

# # 공개키 등록
# # 안해도 되네??
# echo "ssh-rsa skss~~~~~~ lsw0504@naver.com" >> /root/.ssh/id_rsa.pub
# chmod 600 /root/.ssh/id_rsa.pub

# lora weight 업로드
# ai02 에서
# scp -r -P 40615 /ailab/share/kogpt/lora-weight.pkl root@50.217.254.165:/workspace/kogpt/models/ 