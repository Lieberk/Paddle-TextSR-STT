# Paddle-TextSR-STT

## 目录

- [1. 简介]()
- [2. 数据集准备]()
- [3. 复现精度]()
- [4. 模型目录与环境]()
    - [4.1 目录介绍]()
    - [4.2 准备环境]()
- [5. 开始使用]()
    - [5.1 模型训练]()
    - [5.2 模型评估]()
    - [5.3 模型预测]()
- [6. 模型推理开发]() 
- [7. 自动化测试脚本]()
- [8. LICENSE]()
- [9. 模型信息]()

## 1. 简介
**论文:** [Scene Text Telescope: Text-Focused Scene Image Super-Resolution](https://ieeexplore.ieee.org/document/9578891/)

论文提出了一个聚焦文本的超分辨率框架，称为场景文本Telescope(STT)。在文本级布局方面，本文提出了一个基于Transformer的超分辨网络(TBSRN)，包含一个自注意模块来提取序列信息，对任意方向的文本具有鲁棒性。在字符级的细节方面，本文提出了一个位置感知模块和一个内容感知模块来突出每个字符的位置和内容。通过观察一些字符在低分辨率条件下看起来难以区分，本文使用加权交叉熵损失解决。


[aistudio在线运行](https://aistudio.baidu.com/aistudio/projectdetail/4654134)

**参考repo:** [scene-text-telescope](https://github.com/FudanVI/FudanOCR/tree/main/scene-text-telescope)


## 2. 数据集准备

TextZoom中的数据集来自两个超分数据集RealSR和SR-RAW，两个数据集都包含LR-HR对，TextZoom有17367对训数据和4373对测试数据，

全部资源[下载地址](https://aistudio.baidu.com/aistudio/datasetdetail/171370)

* TextZoom dataset
* Pretrained weights of CRNN 
* Pretrained weights of Transformer-based recognizer

数据集目录结构：
```
mydata
├── train1
├── train2
├── confuse.pkl
├── crnn.pdparams
├── pretrain_transformer.pdparams
└── test
    ├── easy
    ├── medium
    └── hard
```

## 3. 复现精度

|        Methods       	 |         easy      	|   medium      |    hard       |    avg      	|
|:------------------:    |:------------------:	|:---------:	|:------:   	|:---------:	|
|        官方repo         | 	      0.5979        |   0.4507  	|    0.3418   	|    0.4634  	|
|        复现repo         | 	      0.5911        |   0.4571  	|    0.3418   	|    0.4634  	|

## 4. 模型目录与环境

### 4.1 目录介绍

```
    |--demo                           # 测试使用的样例图片
    |--deploy                         # 预测部署相关
        |--export_model.py            # 导出模型
        |--infer.py                   # 部署预测
    |--dataset                        # 训练和测试数据集
    |--interfaces                     # 模型基础模块
    |--loss                           # 训练损失 
    |--utils                          # 模型工具文件
    |--model                          # 论文模块
    |--test_tipc                      # tipc代码
    |--predict.py                     # 预测代码
    |--eval.py                        # 评估代码
    |--train.py                       # 训练代码
    |----README.md                    # 用户手册
```

### 4.2 准备环境

- 框架：
  - PaddlePaddle >= 2.3.1
- 环境配置：使用`pip install -r requirement.txt`安装依赖。
  
## 5. 开始使用
### 5.1 模型训练

`python train.py --batch_size 16 --epochs 300 --output_dir './checkpoint/'`

部分训练日志如下所示：
```
[16:13:28.652] Epoch: [0][51/1085]	total_loss 11.712 	mse_loss 0.064 	attention_loss 0.005 	recognition_loss 1.672 	
[16:13:49.211] Epoch: [0][101/1085]	total_loss 8.916 	mse_loss 0.035 	attention_loss 0.005 	recognition_loss 1.639 	
[16:14:10.203] Epoch: [0][151/1085]	total_loss 7.713 	mse_loss 0.032 	attention_loss 0.004 	recognition_loss 1.118 	
[16:14:31.162] Epoch: [0][201/1085]	total_loss 7.111 	mse_loss 0.023 	attention_loss 0.005 	recognition_loss 1.417 	
[16:14:52.526] Epoch: [0][251/1085]	total_loss 7.327 	mse_loss 0.026 	attention_loss 0.005 	recognition_loss 1.264 	
[16:15:13.208] Epoch: [0][301/1085]	total_loss 6.467 	mse_loss 0.021 	attention_loss 0.004 	recognition_loss 1.054 	
[16:15:33.862] Epoch: [0][351/1085]	total_loss 6.016 	mse_loss 0.016 	attention_loss 0.004 	recognition_loss 1.462 	
[16:15:54.761] Epoch: [0][401/1085]	total_loss 6.173 	mse_loss 0.016 	attention_loss 0.004 	recognition_loss 1.184 	
[16:16:15.108] Epoch: [0][451/1085]	total_loss 6.294 	mse_loss 0.019 	attention_loss 0.004 	recognition_loss 1.393 	
[16:16:35.534] Epoch: [0][501/1085]	total_loss 6.109 	mse_loss 0.018 	attention_loss 0.004 	recognition_loss 0.907 	
```

模型训练权重和日志保存到./checkpoint/tbsrn_crnn_train/文件下

可以将训练好的[模型权重下载](https://aistudio.baidu.com/aistudio/datasetdetail/171746) 解压后放在本repo/checkpoint/下，直接运行下面5.2评估和5.3预测部分。

### 5.2 模型评估

- 模型评估：`python eval.py --test_data_dir './dataset/mydata/test'`

输出评估结果：
```
evaling medium
loading pretrained crnn model from ./dataset/mydata/crnn.pdparams
100%|███████████████████████████████████████████| 89/89 [00:17<00:00,  5.08it/s]
{'accuracy': {'medium': 0.4571}, 'psnr_avg': 19.168791, 'ssim_avg': 0.653192, 'fps': 80.50087609964052}
evaling hard
loading pretrained crnn model from ./dataset/mydata/crnn.pdparams
100%|███████████████████████████████████████████| 84/84 [00:15<00:00,  5.57it/s]
{'accuracy': {'hard': 0.3418}, 'psnr_avg': 20.072311, 'ssim_avg': 0.728127, 'fps': 87.1674469405844}
evaling easy
loading pretrained crnn model from ./dataset/mydata/crnn.pdparams
100%|█████████████████████████████████████████| 102/102 [00:18<00:00,  5.60it/s]
{'accuracy': {'easy': 0.5911}, 'psnr_avg': 23.914513, 'ssim_avg': 0.86236, 'fps': 88.93309872011676}
```

### 5.3 模型预测

- 模型预测：`python predict.py --image_name demo1.png'`

预测图片demo/demo1.png结果如下：
```
vuscaalda ===> musicaalta 
ps=2.007209300994873
```

## 6. 模型推理开发

- 模型动转静导出：
```
python deploy/export_model.py
```
输出结果：
```
inference model has been saved into deploy
```

- 基于推理引擎的模型预测：
```
python deploy/infer.py --image_name demo1.png
```
输出结果：
```
image_name: demo/demo1.png, vuscaalda ===> musicaalta
```

## 7. 自动化测试脚本
- tipc 所有代码一键测试命令
```
!bash test_tipc/test_train_inference_python.sh test_tipc/configs/TextSR_STT/train_infer_python.txt lite_train_lite_infer
```

结果日志如下
```
[33m Run successfully with command - python3.7 train.py --output_dir=./test_tipc/output/   --epochs=1 --batch_size=16  !  [0m
[33m Run successfully with command - python3.7 eval.py --output_dir=./test_tipc/output/ --resume './test_tipc/output/tbsrn_crnn_train/model_best.pdparams'       !  [0m
[33m Run successfully with command - python3.7 deploy/export_model.py --pretrained './test_tipc/output/tbsrn_crnn_train/model_best.pdparams'   --save_inference_dir=./test_tipc/output/TextSR_STT/lite_train_lite_infer/norm_train_gpus_0!  [0m
[33m Run successfully with command - python3.7 deploy/infer.py --use_gpu=True --save_inference_dir=./test_tipc/output/TextSR_STT/lite_train_lite_infer/norm_train_gpus_0 --batch_size=1   --benchmark=False --image_name=demo1.png > ./test_tipc/output/TextSR_STT/lite_train_lite_infer/python_infer_gpu_batchsize_1.log 2>&1 !  [0m
[33m Run successfully with command - python3.7 deploy/infer.py --use_gpu=False --save_inference_dir=./test_tipc/output/TextSR_STT/lite_train_lite_infer/norm_train_gpus_0 --batch_size=1   --benchmark=False --image_name=demo1.png > ./test_tipc/output/TextSR_STT/lite_train_lite_infer/python_infer_cpu_batchsize_1.log 2>&1 !  [0m
```

## 8. LICENSE

本项目的发布受[Apache 2.0 license](./LICENSE)许可认证。

## 9. 模型信息

| 信息 | 描述 |
| --- | --- |
| 作者 | Lieber|
| 日期 | 2022年10月 |
| 框架版本 | PaddlePaddle==2.3.1 |
| 应用场景 | 图像超分辨率 |
| 硬件支持 | GPU、CPU |
| 在线体验 | [notebook](https://aistudio.baidu.com/aistudio/projectdetail/4654134)
