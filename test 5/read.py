# read_mnist_and_save.py
import struct
import numpy as np
import os
import random
from PIL import Image

def load_mnist_images(path):
    with open(path, 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(num, rows, cols)
    return images

def load_mnist_labels(path):
    with open(path, 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels


def save_one_image(img, label, save_path, idx):
    """
    保存单张 MNIST 图片
    """
    os.makedirs(save_path, exist_ok=True)
    image = Image.fromarray(img, mode="L")
    filename = f"mnist_{idx}_label_{label}.png"
    image.save(os.path.join(save_path, filename))
    print(f"已保存: {filename}")


def save_random_images(images, labels, save_path, num=10):
    """
    随机保存 num 张 MNIST 图片
    """
    os.makedirs(save_path, exist_ok=True)
    indices = random.sample(range(len(images)), num)
    for idx in indices:
        img = images[idx]
        label = labels[idx]
        image = Image.fromarray(img, mode="L")
        filename = f"mnist_{idx}_label_{label}.png"
        image.save(os.path.join(save_path, filename))
    print(f"随机保存 {num} 张图片到 {save_path}")


if __name__ == "__main__":
    images = load_mnist_images("data/MNIST/t10k-images-idx3-ubyte")
    labels = load_mnist_labels("data/MNIST/t10k-labels-idx1-ubyte")

    # ① 保存一张随机图片
    idx = random.randint(0, len(images) - 1)
    save_one_image(images[idx], labels[idx], "saved_mnist", idx)

    