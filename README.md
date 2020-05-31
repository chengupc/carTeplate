# carTeplate
# carTeplate
模拟图像车牌识别系统

### 训练卷积only的模型
##### 网络模型:
 ![model](https://github.com/sunnythree/car_plate/blob/master/doc/car_plate_rec.png)
##### 训练

#### Steps ：
##### step1 ：生成数据(训练集、测试集)  
```
    cd data
    python dataProcess.py
```


```
    cd pytorch_model
    python3 train.py 30 0.0001
```
30是在训练集训练的次数，0.0001是学习速率  
##### 测试
```
    python3 test.py
```
将会输出准确率，我训练的car_plate_javer.pt模型能达到98.2的准确率（由于训练和测试数据集都是随机生成的，因此可能不同人测试有差异）。
这个准确率不算高，由于我的笔记本算力有限，没能进一步训练更大、更好的模型，不过我想，这个项目已足以证明不分割直接识别车牌的可行性

### 训练crnn(双向gru)+ctc
##### 网络模型:
 ![model](https://github.com/sunnythree/car_plate/blob/master/doc/crnn-ctc.png)
##### 训练
```
    cd pytorch_model
    python3 train.py 30 0.0001 10
```
30是在训练集训练的次数，0.0001是学习速率,10是batch的大小
##### 测试
```
    python3 test.py
```

#### License
chengch