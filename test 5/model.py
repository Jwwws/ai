import mindspore.nn as nn
from mindspore.common.initializer import Normal


class CNN(nn.Cell):
    """
    卷积神经网络结构：
    Conv -> ReLU -> MaxPool
    Conv -> ReLU -> MaxPool
    FC -> ReLU
    FC
    """
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=3,
            pad_mode='same',
            weight_init=Normal(0.02)
        )
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            pad_mode='same',
            weight_init=Normal(0.02)
        )
        self.dropout = nn.Dropout(keep_prob=0.8)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Dense(7 * 7 * 64, 128)
        self.fc2 = nn.Dense(128, num_classes)

    def construct(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

