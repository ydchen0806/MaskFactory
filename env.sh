pip3 install -r /data/ydchen/VLP/MasaCtrl/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
pip3 install gradio -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install --upgrade typing_extensions
# pip3 install ldm -i https://pypi.tuna.tsinghua.edu.cn/simple
cd /data/ydchen/VLP/MasaCtrl/stable-diffusion/stable-diffusion-main
pip install -e .
cd /data/ydchen/VLP/MasaCtrl/BasicSR
pip install -e .