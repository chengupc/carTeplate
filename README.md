## 车牌识别系统
 
##### 流程+网络模型:
 ![model](https://github.com/chengupc/carTeplate/blob/master/Template/imgs/CNN4%2B3.png)

#### steps：
##### 1.数据生成：训练集（80%） + 测试集（20%）  


```
    cd data
    python3 dataProcess.py  100 # 每个省份生成的数据量
```

##### 2.训练

```
    cd train
    python3 train.py 100 0.000001   # 100 是迭代次数，0.000001 是学习率
```
##### 3. 测试
```
    python3 test.py
```


### 后记
 本算法中的CNN 模型是 四层卷积层和三个全连接层，计算量较大；8G 4核 CPU 机器需要训练超过48小时
