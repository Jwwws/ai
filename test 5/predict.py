import numpy as np
from PIL import Image
import mindspore as ms
from mindspore import context, Tensor, load_checkpoint, load_param_into_net
import mindspore.dataset.vision as vision

from model import CNN


def preprocess_image(image_path):
    img = Image.open(image_path).convert("L")  # 转为灰度图

    # 调整大小为 28x28 像素
    img = img.resize((28, 28))

    # 转为numpy数组
    img_array = np.array(img).astype(np.float32)

    # 如果图片是白底黑字（MNIST 是黑底白字），需要反转
    # 检查图片的平均亮度
    if img_array.mean() > 127:  # 如果偏白
        img_array = 255 - img_array  # 反转颜色

    # 归一化到 [0, 1]
    img_array = img_array / 255.0

    # 应用 MNIST 数据集的标准化参数
    img_array = (img_array - 0.1307) / 0.3081

    # 调整形状为 (batch, channel, height, width)
    img_array = img_array.reshape(1, 1, 28, 28)

    return Tensor(img_array, ms.float32)



def predict(image_path, ckpt_path):
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

    net = CNN(num_classes=10)
    param_dict = load_checkpoint(ckpt_path)
    load_param_into_net(net, param_dict)
    net.set_train(False)

    img = preprocess_image(image_path)
    output = net(img)
    pred = output.argmax(axis=1)

    print("预测结果：", int(pred.asnumpy()[0]))


if __name__ == "__main__":
    predict(
        image_path="./微信图片_20251217141037_217_138.jpg",
        ckpt_path="./checkpoints/mnist_cnn_1-10_1094.ckpt"
    )

