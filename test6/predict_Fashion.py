
# ======================================
# Fashion-MNIST 自动反色预测脚本
# ======================================

import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os


# 1. 类别名称

class_names = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]


# 2. 定位当前目录

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "best_model.h5")
img_path = os.path.join(BASE_DIR, "test1.jpg")


# 3. 加载模型

if not os.path.exists(model_path):
    raise FileNotFoundError(" best_model.h5 不存在")

model = tf.keras.models.load_model(model_path)


# 4. 自动反色预处理函数

def preprocess_image_auto_invert(img_path, threshold=0.5):

    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(" 图片读取失败")

    # 灰度
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # resize
    resized = cv2.resize(gray, (28, 28))

    # 归一化
    norm = resized / 255.0

    # 自动判断是否反色
    mean_gray = np.mean(norm)
    invert_flag = mean_gray > threshold

    if invert_flag:
        norm = 1.0 - norm

    input_img = norm.reshape(1, 28, 28, 1)

    return img, gray, resized, norm, input_img, mean_gray, invert_flag


# 5. 执行预处理

img, gray, resized, norm, input_img, mean_gray, inverted = \
    preprocess_image_auto_invert(img_path)

print(f" 平均灰度值: {mean_gray:.3f}")
print(" 是否进行了反色:", "是" if inverted else "否")

plt.figure(figsize=(12,3))

plt.subplot(1,4,1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Original")
plt.axis('off')

plt.subplot(1,4,2)
plt.imshow(gray, cmap='gray')
plt.title("Gray")
plt.axis('off')

plt.subplot(1,4,3)
plt.imshow(resized, cmap='gray')
plt.title("28x28")
plt.axis('off')

plt.subplot(1,4,4)
plt.imshow(norm, cmap='gray')
plt.title("Final Input")
plt.axis('off')

plt.show()



# 7. 预测

pred = model.predict(input_img)
pred_class = np.argmax(pred)
confidence = np.max(pred)

print(" 预测类别:", class_names[pred_class])
print(" 置信度:", confidence)
