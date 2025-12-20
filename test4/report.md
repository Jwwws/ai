# 人工智能基础第四次实验实验报告
## 景奕瑞-软2304-20232241467
### 1.实验目的
本实验旨在通过构建和训练一个基于全连接神经网络（Fully Connected Neural Network, FCNN）的深度学习模型，利用纽约市出租车历史行程数据，实现对特定条件下出租车行程时间（Trip Duration）和总费用（Total Amount）的预测，并评估模型在回归任务中的性能。

### 2.实验内容与处理
#### 数据获取与预处理
本实验基于2015年12月的纽约出租车数据集（Green Tripdata），使用Python的Pandas库进行数据清洗与特征工程。

##### 数据清洗：

**1.异常值处理** ：移除了行程时间极短（$\le 0$）或过长（$> 24$小时）的记录；移除了费用异常的记录。

**2.地理围栏**：根据纽约市的经纬度范围（经度 -74.25 至 -73.70，纬度 40.49 至 40.91），剔除了不在市区范围内的GPS坐标点。

**3.逻辑清洗：**移除了负费用、负行驶距离及乘客数为0的无效数据。

##### 特征工程：

**1.时间特征提取：**从 pickup_datetime 中提取了小时（hour）、星期几（day_of_week）、是否周末（is_weekend）、月（month）和日（day）。

**2.空间特征计算：**利用 Haversine公式 计算起点与终点之间的球面距离。公式如下：
$$
a = \sin^2(\frac{\Delta\phi}{2}) + \cos \phi_1 \cdot \cos \phi_2 \cdot \sin^2(\frac{\Delta\lambda}{2})
$$

$$
c = 2 \cdot \arctan2(\sqrt{a}, \sqrt{1-a})
$$


$$
d = R \cdot c
$$


**3.类别特征编码**：对支付类型（Payment_type）进行 One-Hot 编码，转换为数值特征。

**4.衍生特征：**计算了平均行驶速度（speed_kmh）和小费比例（tip_ratio）用于辅助分析（注：在预测时使用历史平均值填充）。

##### 数据标准化：

使用 StandardScaler 对连续数值特征（如距离、时间、费用等）进行标准化处理，使其均值为0，方差为1，以加速神经网络的收敛。



核心代码如下

```python
 df['trip_duration'] = (df['dropoff_datetime'] - df['pickup_datetime']).dt.total_seconds()

    # 3. 数据清洗
    print("清洗数据...")

    # a. 移除行程时间为负或过长的记录（> 24小时）
    initial_count = len(df)
    df = df[(df['trip_duration'] > 0) & (df['trip_duration'] <= 86400)]
    print(f"移除 {initial_count - len(df)} 条行程时间异常的记录")

    # b. 移除费用异常的记录（< 2.5美元起步价或> 1000美元）
    amount_col = None
    for col in df.columns:
        if 'total' in col.lower() and 'amount' in col.lower():
            amount_col = col
            break

    if not amount_col:
        raise ValueError("找不到包含'total'和'amount'的列名")

    print(f"检测到费用列: {amount_col}")

    initial_count = len(df)
    df = df[(df[amount_col] >= 2.5) & (df[amount_col] <= 1000)]
    print(f"移除 {initial_count - len(df)} 条费用异常的记录")

    # c. 移除GPS坐标异常的记录（限制在纽约市区范围内）
    pickup_lon_col, pickup_lat_col, dropoff_lon_col, dropoff_lat_col = None, None, None, None

    for col in df.columns:
        if 'pickup' in col.lower() and 'lon' in col.lower():
            pickup_lon_col = col
        if 'pickup' in col.lower() and 'lat' in col.lower():
            pickup_lat_col = col
        if 'dropoff' in col.lower() and 'lon' in col.lower():
            dropoff_lon_col = col
        if 'dropoff' in col.lower() and 'lat' in col.lower():
            dropoff_lat_col = col

    if not all([pickup_lon_col, pickup_lat_col, dropoff_lon_col, dropoff_lat_col]):
        raise ValueError("找不到GPS坐标列")

    print(f"检测到GPS坐标列: {pickup_lon_col}, {pickup_lat_col}, {dropoff_lon_col}, {dropoff_lat_col}")

    initial_count = len(df)
    df = df[(df[pickup_lon_col] > -74.2556) & (df[pickup_lon_col] < -73.7004) &
            (df[pickup_lat_col] > 40.4961) & (df[pickup_lat_col] < 40.9155) &
            (df[dropoff_lon_col] > -74.2556) & (df[dropoff_lon_col] < -73.7004) &
            (df[dropoff_lat_col] > 40.4961) & (df[dropoff_lat_col] < 40.9155)]
    print(f"移除 {initial_count - len(df)} 条GPS坐标异常的记录")

    # d. 移除数值为负的记录
    fare_col, tip_col, tolls_col, trip_dist_col = None, None, None, None

    for col in df.columns:
        if 'fare' in col.lower() and 'amount' in col.lower():
            fare_col = col
        if 'tip' in col.lower() and 'amount' in col.lower():
            tip_col = col
        if 'tolls' in col.lower() and 'amount' in col.lower():
            tolls_col = col
        if 'trip' in col.lower() and 'distance' in col.lower():
            trip_dist_col = col

    if fare_col:
        initial_count = len(df)
        df = df[df[fare_col] >= 0]
        print(f"移除 {initial_count - len(df)} 条Fare_amount负值记录")

    if tip_col:
        initial_count = len(df)
        df = df[df[tip_col] >= 0]
        print(f"移除 {initial_count - len(df)} 条Tip_amount负值记录")

    if tolls_col:
        initial_count = len(df)
        df = df[df[tolls_col] >= 0]
        print(f"移除 {initial_count - len(df)} 条Tolls_amount负值记录")

    if trip_dist_col:
        initial_count = len(df)
        df = df[df[trip_dist_col] >= 0]
        print(f"移除 {initial_count - len(df)} 条Trip_distance负值记录")

    # e. 移除乘客数为0或过高的记录
    passenger_col = None
    for col in df.columns:
        if 'passenger' in col.lower() and 'count' in col.lower():
            passenger_col = col
            break

    if passenger_col:
        initial_count = len(df)
        df = df[(df[passenger_col] >= 1) & (df[passenger_col] <= 6)]
        print(f"移除 {initial_count - len(df)} 条乘客数异常的记录")
```

转换为Haversine距离的代码：

```
# 地球半径（公里）
    R = 6371.0

    # 转换为弧度
    lon1_rad = np.radians(lon1)
    lat1_rad = np.radians(lat1)
    lon2_rad = np.radians(lon2)
    lat2_rad = np.radians(lat2)

    # Haversine公式
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c
```

#### 模型设计

本实验使用 **PyTorch** 深度学习框架构建全连接神经网络（DNNRegression）。

**架构选择**：

**输入层**：维度对应特征工程后的特征数量（约20-25维）。

**隐藏层**：包含两个全连接层，每层 **512** 个神经元。

**激活函数**：使用 **ReLU** (Rectified Linear Unit) 增加非线性表达能力。

**正则化**：在每个隐藏层后加入 **Dropout (rate=0.3)**，防止过拟合。

**输出层**：单个神经元，输出预测值（时间或费用）。

**损失函数**：

由于是回归任务，采用 均方误差 (MSE Loss)：
$$
MSE = \frac{1}{n} \sum_{i=1}^{n} (Y_i - \hat{Y}_i)^2
$$

- **优化器**：
  - 使用 **Adam** 优化器，初始学习率为 $0.001$。

模型定义代码如下：

```python
class DNNRegression(nn.Module):
    def __init__(self, input_dim, hidden_units=[512, 512], dropout_rate=0.3):
        super(DNNRegression, self).__init__()
        layers = []
        # 输入层到第一个隐藏层
        layers.append(nn.Linear(input_dim, hidden_units[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        
        # 隐藏层
        for i in range(len(hidden_units) - 1):
            layers.append(nn.Linear(hidden_units[i], hidden_units[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            
        # 输出层
        layers.append(nn.Linear(hidden_units[-1], 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
```

#### 模型训练

**数据集划分**：

训练集：约 70%

验证集：约 15%

测试集：约 15%

代码实现逻辑：先划分30%为测试集，再将测试集对半划分为验证集和最终测试集。

**训练策略**：

Batch Size：64

Epochs：50

早停机制 ：监控验证集损失（Val Loss），如果连续 3 个 Epoch 损失未下降，则停止训练并保存最佳模型。

### 3.实验结果

#### 评估指标

本实验分别训练了两个模型：行程时间预测模型 和 总费用预测模型。使用MSE (均方误差)在测试集上进行测试

结果如下：

![total_amount_learning_curve](D:\university\3rd\ai\homework1\test4\total_amount_learning_curve.png)

![image-20251218135339611](C:\Users\Jw\AppData\Roaming\Typora\typora-user-images\image-20251218135339611.png)

![image-20251218135347757](C:\Users\Jw\AppData\Roaming\Typora\typora-user-images\image-20251218135347757.png)

![image-20251218135400613](C:\Users\Jw\AppData\Roaming\Typora\typora-user-images\image-20251218135400613.png)

### 4.结果分析

**费用预测**的准确度显著高于时间预测。这是因为出租车计费规则相对固定（主要由距离和时间决定），具有较强的线性规律。

**时间预测**的难度较大，因为路况实时变化、红绿灯、天气等不可控因素未完全包含在输入特征中。

费用预测的散点图主要集中在对角线附近，拟合效果较好。

时间预测在长距离行程上发散度增加，说明长途预测的不确定性更高。

#### 算法性能分析

全连接网络的优势：相较于线性回归，多层全连接网络通过 ReLU 激活函数引入了非线性映射，能够更好地拟合复杂的时空关系（例如：同样距离在不同时段的耗时差异）。

特征工程的重要性：实验发现，Haversine 距离和时间段特征（Hour）是影响预测结果权值最大的两个特征。

### 5.局限性与改进建议

空间特征利用不足：目前仅使用了经纬度坐标和直线距离。

改进：引入 OSRM 或 Google Maps API 获取真实的道路导航距离和预计行驶时间作为输入特征。

改进：使用 CNN 对城市网格化地图进行处理，捕捉区域拥堵特征。

外部因素缺失：

改进融合天气数据（雨雪天车速慢）、交通管制信息等外部数据源。

## 6. 实验结论

本实验成功基于 PyTorch 构建了全连接神经网络模型，完成了从数据清洗、特征提取到模型训练与评估的完整流程。实验结果表明，深度学习模型能够较好地拟合出租车费用规则，并对行程时间提供具有参考价值的预测。虽然在极端路况下的时间预测仍有误差，但通过引入 Dropout 和早停策略，模型具有较好的泛化能力。未来可通过引入图神经网络（GNN）或时空序列模型进一步提升预测精度。
