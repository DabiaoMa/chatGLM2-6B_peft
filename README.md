# 项目
chatGLM2-6B的lora, adalora, adaptionPrompt的训练代码和服务部署代码


## 介绍 
    代码主要是在以下三个项目基础上进行的编写，大部分代码来自于下面三个项目
    1. [chatGLM2-6B的官方代码](https://github.com/THUDM/ChatGLM2-6B)
    2. [huggingface的微调方法](https://github.com/huggingface/peft)
    3. [chatGLM-6B的lora训练代码](https://github.com/mymusise/ChatGLM-Tuning)

## 环境
   与官方chatGLM2-6B安装环境一致

## 使用方式

    C.1 lora微调
   
    0. 将chatGLM2-6B的模型文件 (即 pytorch\_model-0000\*.bin) 拷贝到model文件夹。 
    1. 'cd lora/tuning'
    2. 将训练数据转化成类似 `data/train.json` 的格式
    3. 运行tokenize\_dataset\_rows.py: `python tokenize_dataset_rows.py`
    4. 运行run.sh: `sh run.sh`
    5. 训练完毕
    6. `cd lora`
    7. 修改api.py里面lora checkpoint路径
    8. 部署服务: `python api.py`
    -------------------------------------

    C.2 adalora微调
    
    0. 将chatGLM2-6B的模型文件 (即 pytorch\_model-0000\*) 拷贝到model文件夹。
    1. `cd adalora/tuning`
    2. 将训练数据转化成类似 `data/train.json` 的格式
    3. 运行tokenize\_dataset\_rows.py: `python tokenize_dataset_rows.py`
    4. 运行run.sh: `sh run.sh`
    5. 训练完毕
    6. `cd adalora`
    7. 修改api.py里面adalora checkpoint路径
    8. 部署服务: `python api.py`
    ------------------------------------

    C.3 adaptionPrompt微调
       
    0. 将chatGLM2-6B的模型文件 (即 pytorch\_model-0000\*) 拷贝到 adaptionPrompt/tuning/model 文件夹。不要拷贝除pytorch\_model-0000\*外的文件。 
    1. `cd adaptionPrompt/tuning`
    2. 将训练数据转化成类似data/train.json的格式
    3. 运行tokenize\_dataset\_rows.py: `python tokenize_dataset_rows.py`
    4. 运行run.sh: `sh run.sh`
    5. 训练完毕
    6. `cd adaptionPrompt`
    7. 修改api.py里面adaptionPrompt checkpoint路径
    8. 部署服务: `python api.py`
