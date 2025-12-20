import mindspore as ms
import mindspore.nn as nn
import mindspore.dataset as ds
import mindspore.dataset.vision as vision
from mindspore import context, Model
from mindspore.train.callback import LossMonitor, TimeMonitor, ModelCheckpoint, CheckpointConfig
from mindspore.nn.metrics import Accuracy
import mindspore.dataset.transforms as transforms
import mindspore.common.dtype as mstype


from model import CNN


def create_dataset(data_path, batch_size=64, repeat_size=1, shuffle=True):
    dataset = ds.MnistDataset(data_path, shuffle=shuffle)

    image_transform = [
        vision.Rescale(1.0 / 255.0, 0.0),
        vision.Normalize(mean=(0.1307,), std=(0.3081,)),
        vision.HWC2CHW()
    ]

    label_transform = transforms.TypeCast(mstype.int32)

    dataset = dataset.map(operations=image_transform, input_columns="image")
    dataset = dataset.map(operations=label_transform, input_columns="label")

    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(repeat_size)

    return dataset



def train():
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

    train_dataset = create_dataset(
        data_path="data/MNIST",
        batch_size=64,
        repeat_size=1,
        shuffle=True
    )

    test_dataset = create_dataset(
        data_path="data/MNIST",
        batch_size=64,
        repeat_size=1,
        shuffle=False
    )

    network = CNN(num_classes=10)

    loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    optimizer = nn.Adam(network.trainable_params(), learning_rate=0.0005)

    model = Model(
        network,
        loss_fn=loss_fn,
        optimizer=optimizer,
        metrics={"Accuracy": Accuracy()}
    )

    config_ck = CheckpointConfig(
        save_checkpoint_steps=train_dataset.get_dataset_size(),
        keep_checkpoint_max=5
    )
    ckpt_cb = ModelCheckpoint(
        prefix="mnist_cnn",
        directory="./checkpoints",
        config=config_ck
    )

    print("========== 开始训练 ==========")
    model.train(
        epoch=10,
        train_dataset=train_dataset,
        callbacks=[LossMonitor(), TimeMonitor(), ckpt_cb],
        dataset_sink_mode=False
    )

    print("========== 开始测试 ==========")
    acc = model.eval(test_dataset, dataset_sink_mode=False)
    print("测试集准确率：", acc)


if __name__ == "__main__":
    train()

