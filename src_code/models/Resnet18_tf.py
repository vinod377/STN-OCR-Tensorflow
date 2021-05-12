import tensorflow as tf
import numpy as np
from tensorflow.keras import layers


class ResnetBlock1(tf.keras.Model):
    def __init__(self, filters, kernels):
        super(ResnetBlock1, self).__init__()
        filter1, filter2, filter3, filter4 = filters
        kernel1, kernel2, kernel3 = kernels

        # firs layer
        self.conv2d_a = layers.Conv2D(filters=filter1, kernel_size=kernel1, strides=2, padding='same')
        self.maxPool2d_a = layers.MaxPool2D((2, 2), strides=2)
        self.bn_a = layers.BatchNormalization()

        # first block
        self.conv2d_b = layers.Conv2D(filters=filter1, kernel_size=kernel2, padding='same')
        self.bn_b = layers.BatchNormalization()

        # 2nd block
        self.conv2d_c = layers.Conv2D(filters=filter2, kernel_size=kernel3, strides=2, padding='same')
        self.bn_c = layers.BatchNormalization()

        self.conv2d_d = layers.Conv2D(filters=filter2, kernel_size=kernel2, padding='same')
        self.bn_d = layers.BatchNormalization()

        # 3rd block
        self.conv2d_e = layers.Conv2D(filters=filter3, kernel_size=kernel3, strides=2, padding='same')
        self.bn_e = layers.BatchNormalization()

        self.conv2d_f = layers.Conv2D(filters=filter3, kernel_size=kernel2, padding='same')
        self.bn_f = layers.BatchNormalization()

        # 4th block
        self.conv2d_g = layers.Conv2D(filters=filter4, kernel_size=kernel3, strides=2, padding='same')
        self.bn_g = layers.BatchNormalization()

        self.conv2d_h = layers.Conv2D(filters=filter4, kernel_size=kernel2, padding='same')
        self.bn_h = layers.BatchNormalization()

        self.gbPool = layers.GlobalAveragePooling2D()

        self.dense1 = layers.Dense(200)
        self.softmax = layers.Softmax()

    def call(self, inputs, ):

        layer1 = self.conv2d_a(inputs)
        layer1 = self.maxPool2d_a(layer1)
        layer1 = self.bn_a(layer1)
        layer1 = tf.nn.relu(layer1)
        shortcut_1 = layer1

        for i in range(2):
            inp_1 = shortcut_1
            inp_1 = self.conv2d_b(inp_1)
            inp_1 = self.bn_a(inp_1)
            inp_1 = tf.nn.relu(inp_1)

            inp_1 = self.conv2d_b(inp_1)
            inp_1 = self.bn_a(inp_1)

            shortcut_1 = inp_1 + shortcut_1
            shortcut_1 = tf.nn.relu(shortcut_1)

        shortcut_2 = self.conv2d_c(shortcut_1)
        for i in range(2):
            inp_2 = shortcut_2
            inp_2 = self.conv2d_d(inp_2)
            inp_2 = self.bn_d(inp_2)
            inp_2 = tf.nn.relu(inp_2)

            inp_2 = self.conv2d_d(inp_2)
            inp_2 = self.bn_d(inp_2)

            shortcut_2 = inp_2 + shortcut_2
            shortcut_2 = tf.nn.relu(shortcut_2)

        shortcut_3 = self.conv2d_e(shortcut_2)
        for i in range(2):
            inp_3 = shortcut_3
            inp_3 = self.conv2d_f(inp_3)
            inp_3 = self.bn_f(inp_3)

            inp_3 = tf.nn.relu(inp_3)

            inp_3 = self.conv2d_f(inp_3)
            inp_3 = self.bn_f(inp_3)
            shortcut_3 = shortcut_3 + inp_3
            shortcut_3 = tf.nn.relu(shortcut_3)

        shortcut_4 = self.conv2d_g(shortcut_3)

        for i in range(2):
            inp_4 = shortcut_4
            inp_4 = self.conv2d_h(inp_4)
            inp_4 = self.bn_h(inp_4)
            inp_4 = tf.nn.relu(inp_4)

            inp_4 = self.conv2d_h(inp_4)
            inp_4 = self.bn_h(inp_4)

            shortcut_4 = shortcut_4 + inp_4
            shortcut_4 = tf.nn.relu(shortcut_4)
            print(shortcut_4.shape)

        gb_layer = self.gbPool(shortcut_4)
        logit_layer = self.dense1(gb_layer)
        softmax_layer = self.softmax(logit_layer)

        return softmax_layer


if __name__ == "__main__":
    resnetModel = ResnetBlock1([64, 128, 256, 512], [7, 3, 1])
    data = np.ones((32, 224, 224, 3), dtype=np.float32)
    pred = resnetModel(data)
    correct_pred = np.random.randint(0, 2, size=(32, 1))
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    loss_value = loss(correct_pred, pred).numpy()
    print(loss_value)



