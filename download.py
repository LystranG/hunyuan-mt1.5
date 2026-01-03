import os
from modelscope.hub.snapshot_download import snapshot_download

# 定义下载路径：当前目录下的 hunyuan-model 文件夹
local_dir = os.path.join(os.getcwd(), "hunyuan-model")

print(f"开始下载模型到: {local_dir} ...")

model_dir = snapshot_download(
    'Tencent-Hunyuan/HY-MT1.5-1.8B',
    cache_dir=local_dir,
    revision='master'
)

print(f"下载完成！模型路径为: {model_dir}")
