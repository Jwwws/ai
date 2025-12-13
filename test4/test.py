"""
纽约出租车行程时间与费用预测
基于2015年12月纽约黄色出租车数据集的全连接神经网络实现（PyTorch版）
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import time
import warnings
from datetime import datetime
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

# 设置随机种子确保结果可复现
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


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

    # 打印列名以调试
    print("\n原始数据列名:")
    print(df.columns.tolist())

    print(f"原始数据集包含 {len(df)} 条记录")

    # 1. 时间格式转换
    print("处理时间格式...")

    # 尝试多种可能的列名
    pickup_time_col = None
    dropoff_time_col = None

    for col in df.columns:
        if 'pickup' in col.lower() and 'time' in col.lower():
            pickup_time_col = col
        if 'dropoff' in col.lower() and 'time' in col.lower():
            dropoff_time_col = col

    if not pickup_time_col:
        raise ValueError("找不到包含'pickup'和'time'的列名")
    if not dropoff_time_col:
        raise ValueError("找不到包含'dropoff'和'time' in col.lower()的列名")

    print(f"检测到pickup时间列: {pickup_time_col}")
    print(f"检测到dropoff时间列: {dropoff_time_col}")

    df['pickup_datetime'] = pd.to_datetime(df[pickup_time_col], errors='coerce')
    df['dropoff_datetime'] = pd.to_datetime(df[dropoff_time_col], errors='coerce')

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
    # 尝试多种可能的费用列名
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
    # 尝试多种可能的坐标列名
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

    # d. 移除数值为负的记录（如示例中的Fare_amount=-4）
    # 尝试多种可能的费用列名
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
    # 尝试多种可能的乘客数列名
    passenger_col = None
    for col in df.columns:
        if 'passenger' in col.lower() and 'count' in col.lower():
            passenger_col = col
            break

    if passenger_col:
        initial_count = len(df)
        df = df[(df[passenger_col] >= 1) & (df[passenger_col] <= 6)]
        print(f"移除 {initial_count - len(df)} 条乘客数异常的记录")
    else:
        print("警告: 未找到乘客数列，跳过乘客数过滤")

    print(f"清洗后数据集包含 {len(df)} 条有效记录")
    print(f"数据加载与清洗完成，耗时: {time.time() - start_time:.2f}秒")

    return df, amount_col, pickup_lon_col, pickup_lat_col, dropoff_lon_col, dropoff_lat_col, passenger_col, fare_col, tip_col


def haversine_distance(lon1, lat1, lon2, lat2):
    """
    计算两点间的球面距离（Haversine公式）

    应用Haversine公式：a = sin²(Δφ/2) + cosφ₁·cosφ₂·sin²(Δλ/2)
    计算中心角：c = 2·atan2(√a, √(1−a))
    计算距离：d = R·c
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


def feature_engineering(df, amount_col, pickup_lon_col, pickup_lat_col,
                        dropoff_lon_col, dropoff_lat_col, passenger_col,
                        fare_col, tip_col):
    """
    特征工程：提取有用特征

    参数:
        df: 清洗后的DataFrame
        amount_col: 费用列名
        pickup_lon_col, pickup_lat_col: 起点坐标列名
        dropoff_lon_col, dropoff_lat_col: 终点坐标列名
        passenger_col: 乘客数列名
        fare_col: 车费列名
        tip_col: 小费列名

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
        df[pickup_lon_col], df[pickup_lat_col],
        df[dropoff_lon_col], df[dropoff_lat_col]
    )

    # 3. 速度特征（公里/小时）
    df['speed_kmh'] = (df['distance_haversine'] / (df['trip_duration'] / 3600)).replace([np.inf, -np.inf], np.nan)
    df['speed_kmh'] = df['speed_kmh'].fillna(df['speed_kmh'].median())

    # 4. 类别特征编码
    print("处理类别特征...")

    # Payment_type (支付类型) - 尝试查找Payment_type列
    payment_col = None
    for col in df.columns:
        if 'payment' in col.lower() and 'type' in col.lower():
            payment_col = col
            break

    if payment_col:
        # Payment_type (支付类型) - One-Hot编码
        payment_types = [1, 2, 3, 4, 5, 6]
        for pt in payment_types:
            df[f'payment_type_{pt}'] = (df[payment_col] == pt).astype(int)
    else:
        print("警告: 未找到Payment_type列，将创建默认列")
        for pt in payment_types:
            df[f'payment_type_{pt}'] = 0
        # 默认设置信用卡支付
        df['payment_type_1'] = 1

    # Trip_type (行程类型) - 尝试查找Trip_type列
    trip_type_col = None
    for col in df.columns:
        if 'trip' in col.lower() and 'type' in col.lower():
            trip_type_col = col
            break

    if trip_type_col:
        df['is_dispatched'] = (df[trip_type_col] == 2).astype(int)
        print(f"使用列 '{trip_type_col}' 创建 is_dispatched 特征")
    else:
        print("警告: 未找到Trip_type列，将使用默认值0")
        df['is_dispatched'] = 0

    # 5. 其他衍生特征
    # 尝试计算total_charges
    if fare_col and 'Extra' in df.columns and 'MTA_tax' in df.columns and 'Tolls_amount' in df.columns and 'improvement_surcharge' in df.columns:
        df['total_charges'] = df[fare_col] + df['Extra'] + df['MTA_tax'] + df['Tolls_amount'] + df[
            'improvement_surcharge']
    else:
        print("警告: 无法计算total_charges，使用Total_amount代替")
        df['total_charges'] = df[amount_col]

    # 计算tip_ratio
    if fare_col and tip_col:
        df['tip_ratio'] = df[tip_col] / df[fare_col].replace(0, np.nan)
        df['tip_ratio'] = df['tip_ratio'].fillna(df['tip_ratio'].median())
    else:
        print("警告: 无法计算tip_ratio，使用默认值0.2")
        df['tip_ratio'] = 0.2

    # 确保Passenger_count存在
    if not passenger_col:
        print("警告: 未找到Passenger_count列，添加默认值1")
        df['Passenger_count'] = 1
    else:
        df = df.rename(columns={passenger_col: 'Passenger_count'})

    # 确保Trip_distance存在
    trip_dist_col = None
    for col in df.columns:
        if 'trip' in col.lower() and 'distance' in col.lower():
            trip_dist_col = col
            break

    if trip_dist_col:
        df = df.rename(columns={trip_dist_col: 'Trip_distance'})
    else:
        print("警告: 未找到Trip_distance列，使用distance_haversine代替")
        df['Trip_distance'] = df['distance_haversine']

    # 6. 移除中间变量和不需要的列 - 修复点：不要删除目标变量列
    columns_to_drop = ['lpep_pickup_datetime', 'Lpep_dropoff_datetime',
                       'pickup_datetime', 'dropoff_datetime',
                       'Store_and_fwd_flag', 'RateCodeID', 'Ehail_fee',
                       'VendorID', 'Payment_type', 'Trip_type', 'trip_type',
                       pickup_lon_col, pickup_lat_col,
                       dropoff_lon_col, dropoff_lat_col]

    # 只删除确实存在的列，但确保不删除目标变量列
    columns_to_drop = [col for col in columns_to_drop if col in df.columns]

    # 特别重要：确保目标变量列不被删除
    if amount_col in columns_to_drop:
        columns_to_drop.remove(amount_col)
    if fare_col and fare_col in columns_to_drop:
        columns_to_drop.remove(fare_col)
    if tip_col and tip_col in columns_to_drop:
        columns_to_drop.remove(tip_col)

    if columns_to_drop:
        print(f"将删除以下中间列: {columns_to_drop}")
        df = df.drop(columns=columns_to_drop)

    print(f"特征工程完成，最终数据集包含 {len(df.columns)} 个特征")
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
        X_train, X_val, X_
        test, y_train, y_val, y_test, scaler
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

    if not all_features:
        raise ValueError("没有找到有效的特征列")

    print(f"使用的特征: {all_features}")

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
# 2. 模型定义（PyTorch版）
# =============================

class DNNRegression(nn.Module):


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


def create_model(input_dim, learning_rate=0.001, device=None):
    """
    创建模型

    参数:
        input_dim: 输入特征维度
        learning_rate: 学习率
        device: 设备 ('cpu' 或 'cuda')

    返回:
        model, optimizer, loss_fn
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建模型
    model = DNNRegression(input_dim).to(device)

    # 定义损失函数和优化器
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    return model, optimizer, loss_fn, device


# =============================
# 3. 训练与评估（PyTorch版）
# =============================

def train_model(model, optimizer, loss_fn, device,
                X_train, y_train, X_val, y_val,
                batch_size=64, epochs=50, save_path="best_model.pth"):
    """
    训练模型

    参数:
        model: PyTorch模型
        optimizer: 优化器
        loss_fn: 损失函数
        device: 设备
        X_train, y_train: 训练数据
        X_val, y_val: 验证数据
        batch_size: 批量大小
        epochs: 最大训练轮数
        save_path: 模型保存路径

    返回:
        训练历史 (train_losses, val_losses)
    """
    print(f"开始模型训练 (设备: {device})...")

    # 转换为PyTorch Tensor
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1).to(device)

    # 创建DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 训练历史
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    early_stop_counter = 0
    early_stop_patience = 3  # 早停耐心值

    # 训练循环
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)

        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
                val_loss += loss.item() * inputs.size(0)

        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        # 早停检查
        if early_stop_counter >= early_stop_patience:
            print(f"早停触发，最佳验证损失: {best_val_loss:.6f}")
            break

        # 打印进度
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

    print(f"模型训练完成，最佳验证损失: {best_val_loss:.6f}")
    print(f"模型已保存至 {save_path}")

    return train_losses, val_losses


def evaluate_model(model, device, X_test, y_test, target_name="trip_duration"):
    """
    评估模型性能

    参数:
        model: 训练好的模型
        device: 设备
        X_test, y_test: 测试数据
        target_name: 预测目标名称

    返回:
        评估结果字典
    """
    print(f"开始评估{target_name}预测模型...")

    # 转换为PyTorch Tensor
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1).to(device)

    # 评估模型
    model.eval()
    with torch.no_grad():
        y_pred_tensor = model(X_test_tensor)
        y_pred = y_pred_tensor.cpu().numpy().flatten()
        test_loss = nn.MSELoss()(y_pred_tensor, y_test_tensor).item()

    # 计算指标
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"{target_name}预测结果:")
    print(f"MSE: {test_loss:.4f}")
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
        'mse': test_loss,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'y_test': y_test,
        'y_pred': y_pred
    }


def plot_learning_curve(train_losses, val_losses, target_name="trip_duration"):
    """
    绘制学习曲线

    参数:
        train_losses: 训练损失列表
        val_losses: 验证损失列表
        target_name: 预测目标名称
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='训练损失')
    plt.plot(val_losses, label='验证损失')
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
                 payment_type=1, trip_type=1, device=None):
    """
    预测特定行程的行程时间和费用

    参数:
        model: 训练好的模型
        scaler: 标准化器
        features: 所有特征
        feature_names: 特征名称列表
        pickup_coords: (lon, lat) 起点坐标
        dropoff_coords: (lon, lat) 终点坐标
        pickup_datetime: 出发时间 (datetime对象)
        passenger_count: 乘客数量
        payment_type: 支付类型
        trip_type: 行程类型
        device: 设备

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

    # 转换为Tensor
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

    # 预测
    model.eval()
    with torch.no_grad():
        prediction = model(X_tensor).cpu().numpy()[0][0]

    return prediction


# =============================
# 5. 主函数
# =============================

def main():
    """
    主函数：执行完整的实验流程
    """
    # 检查设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 1. 数据加载与预处理
    print("=" * 50)
    print("阶段1: 数据加载与预处理")
    print("=" * 50)

    # 数据文件路径（请根据实际情况修改）
    data_path = r"D:\java学习\springcloud\green_tripdata_2015-12.csv"

    if not os.path.exists(data_path):
        print(f"错误: 数据文件 {data_path} 不存在！")
        print("请先下载2015年12月纽约出租车数据集并放置在当前目录")
        return

    # 加载并清洗数据
    df, amount_col, pickup_lon_col, pickup_lat_col, dropoff_lon_col, dropoff_lat_col, passenger_col, fare_col, tip_col = load_and_clean_data(
        data_path)

    # 特征工程
    df = feature_engineering(df, amount_col, pickup_lon_col, pickup_lat_col,
                             dropoff_lon_col, dropoff_lat_col, passenger_col,
                             fare_col, tip_col)

    # 2. 模型训练：行程时间预测
    print("\n" + "=" * 50)
    print("阶段2: 行程时间预测模型训练")
    print("=" * 50)

    # 准备行程时间预测数据集
    (X_train_dur, X_val_dur, X_test_dur,
     y_train_dur, y_val_dur, y_test_dur,
     scaler_dur, features_dur) = prepare_dataset(df, target='trip_duration')

    # 创建并训练模型
    model_dur, optimizer_dur, loss_fn_dur, _ = create_model(
        X_train_dur.shape[1], learning_rate=0.001, device=device
    )

    train_losses_dur, val_losses_dur = train_model(
        model_dur, optimizer_dur, loss_fn_dur, device,
        X_train_dur, y_train_dur, X_val_dur, y_val_dur,
        save_path="trip_duration_model.pth"
    )

    # 评估模型
    results_dur = evaluate_model(model_dur, device, X_test_dur, y_test_dur, "trip_duration")
    plot_learning_curve(train_losses_dur, val_losses_dur, "trip_duration")

    # 3. 模型训练：费用预测
    print("\n" + "=" * 50)
    print("阶段3: 费用预测模型训练")
    print("=" * 50)

    # 准备费用预测数据集
    (X_train_amt, X_val_amt, X_test_amt,
     y_train_amt, y_val_amt, y_test_amt,
     scaler_amt, features_amt) = prepare_dataset(df, target=amount_col)

    # 创建并训练模型
    model_amt, optimizer_amt, loss_fn_amt, _ = create_model(
        X_train_amt.shape[1], learning_rate=0.001, device=device
    )

    train_losses_amt, val_losses_amt = train_model(
        model_amt, optimizer_amt, loss_fn_amt, device,
        X_train_amt, y_train_amt, X_val_amt, y_val_amt,
        save_path="total_amount_model.pth"
    )

    # 评估模型
    results_amt = evaluate_model(model_amt, device, X_test_amt, y_test_amt, "total_amount")
    plot_learning_curve(train_losses_amt, val_losses_amt, "total_amount")

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
        model_dur, scaler_dur, None, features_dur,
        jfk, times_square,
        pickup_time, passenger_count=2,
        payment_type=1, trip_type=1, device=device
    )

    # 预测费用
    amount_pred = predict_trip(
        model_amt, scaler_amt, None, features_amt,
        jfk, times_square,
        pickup_time, passenger_count=2,
        payment_type=1, trip_type=1, device=device
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
        model_dur, scaler_dur, None, features_dur,
        downtown_start, downtown_end,
        pickup_time2, passenger_count=1,
        payment_type=2, trip_type=1, device=device
    )

    # 预测费用
    amount_pred2 = predict_trip(
        model_amt, scaler_amt, None, features_amt,
        downtown_start, downtown_end,
        pickup_time2, passenger_count=1,
        payment_type=2, trip_type=1, device=device
    )

    print(f"预测行程时间: {duration_pred2 / 60:.1f} 分钟")
    print(f"预测总费用: ${amount_pred2:.2f}")

    print("\n实验完成！")
    print("结果已保存为图像文件，模型已保存至本地")


if __name__ == "__main__":
    main()
