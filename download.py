# docker run --gpus all -it --rm --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v /home/gowitheflow/Desktop:/workspace ngc.nju.edu.cn/nvidia/pytorch:26.04-py3 /bin/bash
# docker run --gpus all -itd --name mieb --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v /home/gowitheflow/Desktop:/workspace ngc.nju.edu.cn/nvidia/pytorch:26.04-py3 /bin/bash
# python -m venv --system-site-packages /workspace/envs/mieb
# apt-get update && apt-get install -y ffmpeg

# import os
# from huggingface_hub import snapshot_download
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import os
from huggingface_hub import snapshot_download
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

datasets = [
    # "Yiqi-Liu/physics-bench-dynamics-eval",
    # "Yiqi-Liu/physics-bench-fluid-eval",
    "gowitheflowlab/physics-bench-fluid-eval"
    # "gowitheflowlab/physics-bench-dynamics-eval",
    # "gowitheflowlab/physics-bench-fluid-eval",
    # "gowitheflowlab/physics-bench-solid-eval",
    # "gowitheflowlab/physics-bench-optics-eval",
]
for dataset in datasets:
    snapshot_download(
        repo_id=dataset,
        local_dir=f"./datasets/{dataset.split("/")[1]}-wrong-order-correct-caption",
        resume_download=True,
        repo_type="dataset",
        token=""
    )



# docker run --gpus all --network host -itd --name gemini --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v /home/gowitheflow/Desktop:/workspace ngc.nju.edu.cn/nvidia/pytorch:26.04-py3 /bin/bash
