"""
纽约出租车行程时间与费用预测
基于2015年12月纽约黄色出租车数据集的全连接神经网络实现
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import mindspore as ms
from mindspore import nn, ops, context, Tensor, dataset
from mindspore.train import Model, LossMonitor, CheckpointConfig, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import time
import warnings

warnings.filterwarnings('ignore')

# 设置随机种子确保结果可复现
np.random.seed(42)
ms.set_seed(42)

# 设置MindSpore上下文
context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
print(f"当前使用的设备: {context.get_context('device_target')}")


# =============================
# 1. 数据加载与预处理
# =============================

def load_and_clean_data(file_path):
    """
    加载并清洗纽约出租车数据

    参数:
        file_path: CSV文件路径

    返回:
        清洗后的DataFrame
    """
    print("开始加载数据...")
    start_time = time.time()

    # 读取CSV文件
    df = pd.read_csv(file_path, low_memory=False)

    print(f"原始数据集包含 {len(df)} 条记录")

    # 1. 时间格式转换
    print("处理时间格式...")
    df['pickup_datetime'] = pd.to_datetime(df['lpep_pickup_datetime'], format='%Y/%m/%d %H:%M', errors='coerce')
    df['dropoff_datetime'] = pd.to_datetime(df['Lpep_dropoff_datetime'], format='%Y/%m/%d %H:%M', errors='coerce')

    # 移除时间格式转换失败的记录
    initial_count = len(df)
    df = df.dropna(subset=['pickup_datetime', 'dropoff_datetime'])
    print(f"移除 {initial_count - len(df)} 条时间格式错误的记录")

    # 2. 计算行程时间（秒）
    df['trip_duration'] = (df['dropoff_datetime'] - df['pickup_datetime']).dt.total_seconds()

    # 3. 数据清洗
    print("清洗数据...")

    # a. 移除行程时间为负或过长的记录（> 24小时）
    initial_count = len(df)
    df = df[(df['trip_duration'] > 0) & (df['trip_duration'] <= 86400)]
    print(f"移除 {initial_count - len(df)} 条行程时间异常的记录")

    # b. 移除费用异常的记录（< 2.5美元起步价或> 1000美元）
    initial_count = len(df)
    df = df[(df['Total_amount'] >= 2.5) & (df['Total_amount'] <= 1000)]
    print(f"移除 {initial_count - len(df)} 条费用异常的记录")

    # c. 移除GPS坐标异常的记录（限制在纽约市区范围内）
    initial_count = len(df)
    df = df[(df['Pickup_longitude'] > -74.2556) & (df['Pickup_longitude'] < -73.7004) &
            (df['Pickup_latitude'] > 40.4961) & (df['Pickup_latitude'] < 40.9155) &
            (df['Dropoff_longitude'] > -74.2556) & (df['Dropoff_longitude'] < -73.7004) &
            (df['Dropoff_latitude'] > 40.4961) & (df['Dropoff_latitude'] < 40.9155)]
    print(f"移除 {initial_count - len(df)} 条GPS坐标异常的记录")

    # d. 移除数值为负的记录（如示例中的Fare_amount=-4）
    initial_count = len(df)
    df = df[(df['Fare_amount'] >= 0) & (df['Tip_amount'] >= 0) &
            (df['Tolls_amount'] >= 0) & (df['Trip_distance'] >= 0)]
    print(f"移除 {initial_count - len(df)} 条负值异常的记录")

    # e. 移除乘客数为0或过高的记录
    initial_count = len(df)
    df = df[(df['Passenger_count'] >= 1) & (df['Passenger_count'] <= 6)]
    print(f"移除 {initial_count - len(df)} 条乘客数异常的记录")

    print(f"清洗后数据集包含 {len(df)} 条有效记录")
    print(f"数据加载与清洗完成，耗时: {time.time() - start_time:.2f}秒")

    return df


def haversine_distance(lon1, lat1, lon2, lat2):
    """
    计算两点间的球面距离（Haversine公式）

    参数:
        lon1, lat1: 起点的经度和纬度
        lon2, lat2: 终点的经度和纬度

    返回:
        两点间的距离（公里）
    """
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


def feature_engineering(df):
    """
    特征工程：提取有用特征

    参数:
        df: 清洗后的DataFrame

    返回:
        包含新特征的DataFrame
    """
    print("开始特征工程...")
    start_time = time.time()

    # 1. 时间特征
    df['hour'] = df['pickup_datetime'].dt.hour
    df['day_of_week'] = df['pickup_datetime'].dt.dayofweek  # 0=周一, 6=周日
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)  # 周末=1, 工作日=0
    df['month'] = df['pickup_datetime'].dt.month
    df['day'] = df['pickup_datetime'].dt.day

    # 2. 空间特征 - Haversine距离
    print("计算Haversine距离...")
    df['distance_haversine'] = haversine_distance(
        df['Pickup_longitude'], df['Pickup_latitude'],
        df['Dropoff_longitude'], df['Dropoff_latitude']
    )

    # 3. 速度特征（公里/小时）
    df['speed_kmh'] = (df['distance_haversine'] / (df['trip_duration'] / 3600)).replace([np.inf, -np.inf], np.nan)
    df['speed_kmh'] = df['speed_kmh'].fillna(df['speed_kmh'].median())

    # 4. 类别特征编码
    print("处理类别特征...")

    # Payment_type (支付类型) - One-Hot编码
    payment_types = [1, 2, 3, 4, 5, 6]
    for pt in payment_types:
        df[f'payment_type_{pt}'] = (df['Payment_type'] == pt).astype(int)

    # Trip_type (行程类型) - 二值特征
    df['is_dispatched'] = (df['Trip_type'] == 2).astype(int)

    # 5. 其他衍生特征
    df['total_charges'] = df['Fare_amount'] + df['Extra'] + df['MTA_tax'] + df['Tolls_amount'] + df[
        'improvement_surcharge']
    df['tip_ratio'] = df['Tip_amount'] / df['Fare_amount'].replace(0, np.nan)
    df['tip_ratio'] = df['tip_ratio'].fillna(df['tip_ratio'].median())

    # 6. 移除中间变量和不需要的列
    columns_to_drop = ['lpep_pickup_datetime', 'Lpep_dropoff_datetime',
                       'pickup_datetime', 'dropoff_datetime',
                       'Store_and_fwd_flag', 'RateCodeID', 'Ehail_fee',
                       'VendorID', 'Payment_type', 'Trip_type']
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

    print(f"特征工程完成，耗时: {time.time() - start_time:.2f}秒")
    return df


def prepare_dataset(df, target='trip_duration', test_size=0.3, val_size=0.5):
    """
    准备数据集：标准化、划分训练/验证/测试集

    参数:
        df: 特征工程后的DataFrame
        target: 预测目标 ('trip_duration' 或 'total_amount')
        test_size: 测试集比例
        val_size: 验证集在剩余数据中的比例

    返回:
        X_train, X_val, X_test, y_train, y_val, y_test, scaler
    """
    print(f"准备{target}预测的数据集...")

    # 1. 选择特征和目标变量
    # 定义数值特征
    numeric_features = ['Trip_distance', 'Passenger_count', 'hour', 'day_of_week',
                        'is_weekend', 'month', 'day', 'distance_haversine', 'speed_kmh',
                        'total_charges', 'tip_ratio']

    # 定义类别特征（已经One-Hot编码）
    categorical_features = [f'payment_type_{i}' for i in range(1, 7)] + ['is_dispatched']

    # 合并所有特征
    all_features = numeric_features + categorical_features

    # 检查特征是否存在
    all_features = [feat for feat in all_features if feat in df.columns]

    # 准备X和y
    X = df[all_features].values.astype(np.float32)
    y = df[target].values.astype(np.float32)

    # 2. 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=val_size, random_state=42)

    print(f"训练集: {X_train.shape[0]} 条")
    print(f"验证集: {X_val.shape[0]} 条")
    print(f"测试集: {X_test.shape[0]} 条")

    # 3. 标准化数值特征
    scaler = StandardScaler()

    # 仅对数值特征进行标准化
    numeric_indices = [i for i, feat in enumerate(all_features) if feat in numeric_features]

    # 保存原始数值特征用于后续恢复
    X_train_orig = X_train.copy()
    X_val_orig = X_val.copy()
    X_test_orig = X_test.copy()

    # 标准化数值特征
    X_train[:, numeric_indices] = scaler.fit_transform(X_train[:, numeric_indices])
    X_val[:, numeric_indices] = scaler.transform(X_val[:, numeric_indices])
    X_test[:, numeric_indices] = scaler.transform(X_test[:, numeric_indices])

    print("数据标准化完成")

    return X_train, X_val, X_test, y_train, y_val, y_test, scaler, all_features


# =============================
# 2. 模型定义
# =============================

class DNNRegression(nn.Cell):
    """
    全连接神经网络回归模型
    """

    def __init__(self, input_dim, hidden_units=[512, 512], dropout_rate=0.3):
        """
        初始化模型

        参数:
            input_dim: 输入特征维度
            hidden_units: 每个隐藏层的神经元数量
            dropout_rate: Dropout比率
        """
        super(DNNRegression, self).__init__()
        layers = []

        # 输入层到第一个隐藏层
        layers.append(nn.Dense(input_dim, hidden_units[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(p=dropout_rate))

        # 隐藏层
        for i in range(len(hidden_units) - 1):
            layers.append(nn.Dense(hidden_units[i], hidden_units[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout_rate))

        # 输出层
        layers.append(nn.Dense(hidden_units[-1], 1))

        self.network = nn.SequentialCell(layers)

    def construct(self, x):
        return self.network(x)


def create_model(input_dim, learning_rate=0.001):
    """
    创建并编译模型

    参数:
        input_dim: 输入特征维度
        learning_rate: 学习率

    返回:
        编译好的Model对象
    """
    # 创建模型
    model = DNNRegression(input_dim)

    # 定义损失函数和优化器
    loss_fn = nn.MSELoss()
    optimizer = nn.Adam(model.trainable_params(), learning_rate=learning_rate)

    # 编译模型
    model = Model(model, loss_fn=loss_fn, optimizer=optimizer, metrics={'mae': nn.MAE()})

    return model


# =============================
# 3. 训练与评估
# =============================

def train_model(model, X_train, y_train, X_val, y_val, batch_size=64, epochs=50, save_path="best_model.ckpt"):
    """
    训练模型

    参数:
        model: 编译好的Model对象
        X_train, y_train: 训练数据
        X_val, y_val: 验证数据
        batch_size: 批量大小
        epochs: 最大训练轮数
        save_path: 模型保存路径

    返回:
        训练历史
    """
    print("开始模型训练...")

    # 转换为MindSpore Dataset
    train_dataset = dataset.GeneratorDataset(
        list(zip(X_train, y_train)),
        ["data", "label"],
        shuffle=True
    )
    train_dataset = train_dataset.batch(batch_size)

    val_dataset = dataset.GeneratorDataset(
        list(zip(X_val, y_val)),
        ["data", "label"]
    )
    val_dataset = val_dataset.batch(batch_size)

    # 配置模型保存
    config_ck = CheckpointConfig(save_checkpoint_steps=len(X_train) // batch_size,
                                 keep_checkpoint_max=5)
    ckpoint = ModelCheckpoint(prefix="taxi_dnn", directory="./checkpoints", config=config_ck)

    # 训练并记录历史
    history = model.train(epochs, train_dataset, callbacks=[LossMonitor(), ckpoint],
                          dataset_sink_mode=False, valid_dataset=val_dataset, valid_frequency=1)

    # 保存最佳模型
    ms.save_checkpoint(model.network, save_path)
    print(f"模型已保存至 {save_path}")

    return history


def evaluate_model(model, X_test, y_test, target_name="trip_duration"):
    """
    评估模型性能

    参数:
        model: 训练好的模型
        X_test, y_test: 测试数据
        target_name: 预测目标名称

    返回:
        评估结果字典
    """
    print(f"开始评估{target_name}预测模型...")

    # 转换为MindSpore Dataset
    test_dataset = dataset.GeneratorDataset(
        list(zip(X_test, y_test)),
        ["data", "label"]
    )
    test_dataset = test_dataset.batch(64)

    # 评估模型
    eval_metrics = model.eval(test_dataset)
    print(f"测试集评估结果: {eval_metrics}")

    # 预测
    y_pred = model.predict(Tensor(X_test)).asnumpy().flatten()

    # 计算额外指标
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"{target_name}预测结果:")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²: {r2:.4f}")

    # 绘制预测vs真实值图
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
    plt.xlabel('真实值')
    plt.ylabel('预测值')
    plt.title(f'{target_name}预测: 真实值 vs 预测值')
    plt.savefig(f'{target_name}_prediction.png')
    plt.close()

    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'y_test': y_test,
        'y_pred': y_pred
    }


def plot_learning_curve(history, target_name="trip_duration"):
    """
    绘制学习曲线

    参数:
        history: 训练历史
        target_name: 预测目标名称
    """
    train_loss = [x['loss'] for x in history]
    val_loss = [x['val_loss'] for x in history]

    plt.figure(figsize=(10, 6))
    plt.plot(train_loss, label='训练损失')
    plt.plot(val_loss, label='验证损失')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title(f'{target_name}预测模型学习曲线')
    plt.legend()
    plt.savefig(f'{target_name}_learning_curve.png')
    plt.close()


# =============================
# 4. 预测示例
# =============================

def predict_trip(model, scaler, features, feature_names,
                 pickup_coords, dropoff_coords,
                 pickup_datetime, passenger_count,
                 payment_type=1, trip_type=1):
    """
    预测特定行程的行程时间和费用

    参数:
        model: 训练好的模型
        scaler: 标准化器
        features: 所有特征名称
        pickup_coords: (lon, lat) 起点坐标
        dropoff_coords: (lon, lat) 终点坐标
        pickup_datetime: 出发时间 (datetime对象)
        passenger_count: 乘客数量
        payment_type: 支付类型
        trip_type: 行程类型

    返回:
        预测的行程时间和费用
    """
    # 创建单条记录
    record = {}

    # 1. 时间特征
    record['hour'] = pickup_datetime.hour
    record['day_of_week'] = pickup_datetime.weekday()  # 0=周一, 6=周日
    record['is_weekend'] = 1 if pickup_datetime.weekday() >= 5 else 0
    record['month'] = pickup_datetime.month
    record['day'] = pickup_datetime.day

    # 2. 空间特征
    pickup_lon, pickup_lat = pickup_coords
    dropoff_lon, dropoff_lat = dropoff_coords

    record['distance_haversine'] = haversine_distance(
        pickup_lon, pickup_lat, dropoff_lon, dropoff_lat
    )

    # 3. 乘客特征
    record['Passenger_count'] = passenger_count
    # 使用平均Trip_distance作为估计（实际应用中可以更精确）
    record['Trip_distance'] = record['distance_haversine'] * 1.2  # 估计道路距离

    # 4. 支付特征
    for i in range(1, 7):
        record[f'payment_type_{i}'] = 1 if i == payment_type else 0
    record['is_dispatched'] = 1 if trip_type == 2 else 0

    # 5. 其他特征（使用平均值）
    record['speed_kmh'] = 20.0  # 纽约平均车速
    record['total_charges'] = 15.0  # 平均费用
    record['tip_ratio'] = 0.2  # 平均小费比例

    # 创建DataFrame
    df = pd.DataFrame([record])

    # 按特征顺序排列
    X = df[[f for f in feature_names if f in df.columns]].values.astype(np.float32)

    # 标准化
    numeric_features = ['Trip_distance', 'Passenger_count', 'hour', 'day_of_week',
                        'is_weekend', 'month', 'day', 'distance_haversine', 'speed_kmh',
                        'total_charges', 'tip_ratio']
    numeric_indices = [i for i, feat in enumerate(feature_names) if feat in numeric_features]

    X[:, numeric_indices] = scaler.transform(X[:, numeric_indices])

    # 预测
    prediction = model.predict(Tensor(X)).asnumpy()[0][0]

    return prediction


# =============================
# 5. 主函数
# =============================

def main():
    """
    主函数：执行完整的实验流程
    """
    # 1. 数据加载与预处理
    print("=" * 50)
    print("阶段1: 数据加载与预处理")
    print("=" * 50)

    # 数据文件路径（请根据实际情况修改）
    data_path = "yellow_tripdata_2015-12.csv"

    if not os.path.exists(data_path):
        print(f"错误: 数据文件 {data_path} 不存在！")
        print("请先下载2015年12月纽约出租车数据集并放置在当前目录")
        return

    # 加载并清洗数据
    df = load_and_clean_data(data_path)

    # 特征工程
    df = feature_engineering(df)

    # 2. 模型训练：行程时间预测
    print("\n" + "=" * 50)
    print("阶段2: 行程时间预测模型训练")
    print("=" * 50)

    # 准备行程时间预测数据集
    (X_train_dur, X_val_dur, X_test_dur,
     y_train_dur, y_val_dur, y_test_dur,
     scaler_dur, features_dur) = prepare_dataset(df, target='trip_duration')

    # 创建并训练模型
    model_dur = create_model(X_train_dur.shape[1], learning_rate=0.001)
    history_dur = train_model(model_dur, X_train_dur, y_train_dur, X_val_dur, y_val_dur,
                              save_path="trip_duration_model.ckpt")

    # 评估模型
    results_dur = evaluate_model(model_dur, X_test_dur, y_test_dur, "trip_duration")
    plot_learning_curve(history_dur, "trip_duration")

    # 3. 模型训练：费用预测
    print("\n" + "=" * 50)
    print("阶段3: 费用预测模型训练")
    print("=" * 50)

    # 准备费用预测数据集
    (X_train_amt, X_val_amt, X_test_amt,
     y_train_amt, y_val_amt, y_test_amt,
     scaler_amt, features_amt) = prepare_dataset(df, target='Total_amount')

    # 创建并训练模型
    model_amt = create_model(X_train_amt.shape[1], learning_rate=0.001)
    history_amt = train_model(model_amt, X_train_amt, y_train_amt, X_val_amt, y_val_amt,
                              save_path="total_amount_model.ckpt")

    # 评估模型
    results_amt = evaluate_model(model_amt, X_test_amt, y_test_amt, "total_amount")
    plot_learning_curve(history_amt, "total_amount")

    # 4. 示例预测
    print("\n" + "=" * 50)
    print("阶段4: 示例预测")
    print("=" * 50)

    # 示例1: JFK机场到时代广场
    jfk = (-73.7781, 40.6413)  # JFK机场
    times_square = (-73.9855, 40.7580)  # 时代广场
    pickup_time = datetime(2015, 12, 15, 18, 30)  # 2015-12-15 18:30（周二晚高峰）

    print("\n预测示例1: JFK机场到时代广场")
    print(f"出发时间: {pickup_time}")
    print(f"起点: JFK机场 {jfk}")
    print(f"终点: 时代广场 {times_square}")

    # 预测行程时间
    duration_pred = predict_trip(
        model_dur, scaler_dur, features_dur,
        jfk, times_square,
        pickup_time, passenger_count=2,
        payment_type=1, trip_type=1
    )

    # 预测费用
    amount_pred = predict_trip(
        model_amt, scaler_amt, features_amt,
        jfk, times_square,
        pickup_time, passenger_count=2,
        payment_type=1, trip_type=1
    )

    print(f"预测行程时间: {duration_pred / 60:.1f} 分钟")
    print(f"预测总费用: ${amount_pred:.2f}")

    # 示例2: 纽约市中心短途行程
    downtown_start = (-73.985, 40.748)  # 纽约市中心
    downtown_end = (-73.975, 40.755)  # 纽约市中心
    pickup_time2 = datetime(2015, 12, 10, 10, 15)  # 2015-12-10 10:15（周四上午）

    print("\n预测示例2: 纽约市中心短途行程")
    print(f"出发时间: {pickup_time2}")
    print(f"起点: {downtown_start}")
    print(f"终点: {downtown_end}")

    # 预测行程时间
    duration_pred2 = predict_trip(
        model_dur, scaler_dur, features_dur,
        downtown_start, downtown_end,
        pickup_time2, passenger_count=1,
        payment_type=2, trip_type=1
    )

    # 预测费用
    amount_pred2 = predict_trip(
        model_amt, scaler_amt, features_amt,
        downtown_start, downtown_end,
        pickup_time2, passenger_count=1,
        payment_type=2, trip_type=1
    )

    print(f"预测行程时间: {duration_pred2 / 60:.1f} 分钟")
    print(f"预测总费用: ${amount_pred2:.2f}")

    print("\n实验完成！")
    print("结果已保存为图像文件，模型已保存至本地")


if __name__ == "__main__":
    main()
