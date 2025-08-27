## 环境配置
```bash
conda create -n fifth_jittor_comp -y python=3.9
conda activate fifth_jittor_comp
python -m pip install jittor
```

## 训练数据集配置
下载TrainSet.zip, 解压后得到TrainSet文件夹，文件结果为
```
TrainSet
├── images
│   └── train
│       ├── 0.jpg
│       ├── 1.jpg
│       └── ...
└── labels
    ├── train.txt
    ├── val.txt
    └── trainval.txt
```

## 模型训练
执行`python main.py --dataroot TrainSet`即可训练。使用不同的`--modelroot`可保存不同的模型。
训练结果保存在`./model_save`文件夹中。
默认会将测试结果保存在`./result.txt`文件中。

## 测试数据集配置
下载TestSetA.zip, 解压后得到TestSetA文件夹，文件结果为
```
TestSetA
├── images
│   └── test
│       ├── 0.jpg
│       ├── 1.jpg
│       └── ...
```

## 模型推理
执行`python main.py --dataroot TestSetA --testonly`即可进行模型推理。测试结果保存在`./result.txt`文件中。

