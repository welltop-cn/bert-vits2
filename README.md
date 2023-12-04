# Bert_Vits2

## 文件结构


## 工作流

### Step 1. 在`Data`路径创造新`speaker`路径
```
speaker = "xxx"

%mkdir Data/{speaker}
%mkdir Data/{speaker}/audios
%mkdir Data/{speaker}/audios/raw
%mkdir Data/{speaker}/audios/wavs
%mkdir Data/{speaker}/filelists
%mkdir Data/{speaker}/models
%cp configs/config.json config.yml  Data/{speaker}
%cp Data/base/*.pth Data/{speaker}/models
!echo "环境初始化完成"
```

### Step 2. 通过##Alice_split_toolset##项目规范化克隆的音频数据，并导入项目

- Alice_split_toolset处理流程
```
python split.py --mono
python clean_list.py --filter_english
python merge.py
python pack.py baki
```

- 数据导入项目
```
%cp Alice_split_toolset/dataset/{speaker}/*wav Data/{speaker}/audios/raw
%cp Alice_split_toolset/dataset/{speaker}/short_character_anno.list  Data/{speaker}/filelists
```

- 修改`Data/{speaker}/config.yml`，主要修改`speaker`相关路径内容，完成后拷贝到项目根目录

### Step 3. 重采样、预处理、生成bert文件
```
!/root/anaconda3/envs/py3.10/bin/python3 resample.py

!/root/anaconda3/envs/py3.10/bin/python3 preprocess_text.py

!/root/anaconda3/envs/py3.10/bin/python3 bert_gen.py
```

### Step 4. 训练模型
```
python3 train_ms.py
```

### Step 5. 使用模型进行推理
```
import os
from server_fastapi import Models
from config import config 


# 加载模型
loaded_models = Models()
models_info = config.server_config.models
for model_info in models_info:
    loaded_models.init_model(
        config_path=model_info["config"],
        model_path=model_info["model"],
        device=model_info["device"],
        language=model_info["language"],
    )

# 打印加载的index=0模型
print(os.path.realpath(loaded_models.models[0].model_path))

# 初始化推理参数
model_id=0
text = "Is my charm too hot for you to resist?"
sdp_ratio = 0.2
noise_scale = 0.2
noise_scale_w = 0.9
length_scale = 1
sid=speaker
language="EN"
hps=loaded_models.models[model_id].hps
net_g=loaded_models.models[model_id].net_g
device=loaded_models.models[model_id].device

# 开始推理
audio_array = infer(
    text=text,
    sdp_ratio=sdp_ratio,
    noise_scale=noise_scale,
    noise_scale_w=noise_scale_w,
    length_scale=length_scale,
    sid=sid,
    language=language,
    hps=hps,
    net_g=net_g,
    device=device,
)
write_wav(f"sample.wav", 44100, audio_array)
```


## Reference
- [fishaudio/Bert-VITS2](https://github.com/fishaudio/Bert-VITS2) 原始项目
- [AliceNavigator/Alice_split_toolset](https://github.com/AliceNavigator/Alice_split_toolset) 数据规范化
