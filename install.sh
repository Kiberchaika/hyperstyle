sudo apt-get update
sudo apt-get install python3-venv
python3 -m venv ~/gpu_ml_env
source ~/gpu_ml_env/bin/activate
cd hyperstyle/
pip install fastapi
pip3 install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install uvicorn scipy
pip install cmake wheel
sudo apt install python3-dev
pip install dlib tqdm matplotlib python-multipart
pip install ninja
python web.py
