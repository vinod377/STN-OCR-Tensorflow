"""
Implementation of Resnet-variant for cifar-10 dataset as proposed by kaiming He in
"Deep Residual Learning for Image Recognition (https://arxiv.org/abs/1512.03385)".

for model creation,create object of ResnetModelCifar by initializing it  with parameters
input_shape(minimum 32X32),number classes(nb_class) and block_length(3,5,7,9).
in Implementaion,projection layer is used instead of zeroPadding .
"""

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow import keras

class ResnetModelCifar:
    def __init__(self, input_shape, nb_classes=10, block_length=3):
        self.input = layers.Input(shape=input_shape)
        self.nb_classes = nb_classes
        self.block_length = block_length

    def convBnRelu(self, input, filter, size, stride, projection, name):
        if projection:
            stride = 2
        else:
            stride = 1
        inp = layers.Conv2D(filters=filter, kernel_size=size, padding='same', strides=stride,
                            kernel_regularizer=regularizers.l1_l2(l1=1e-4, l2=1e-4), name=name + "conv")(input)
        inp = layers.BatchNormalization(name=name + "Bn")(inp)
        inp = tf.nn.relu(inp, name=name + "relu")
        return inp

    def covnBn(self, input, filter, size, stride, name, padding='valid'):
        inp = layers.Conv2D(filters=filter, kernel_size=size, strides=stride, padding=padding,
                            kernel_regularizer=regularizers.l1_l2(l1=1e-4, l2=1e-4), name=name + "conv")(input)
        inp = layers.BatchNormalization(name=name + "Bn")(inp)
        return inp

    def residualNet(self, inp, filter, size, stride, projection, name="block"):
        if projection:
            skip_conn = self.covnBn(inp, filter, size=1, stride=2, name=name + "skip_conn/")
        else:
            skip_conn = inp
        inp = self.convBnRelu(input=inp, filter=filter, size=size, stride=stride, projection=projection,
                              name=name + "convBnRelu/")
        inp = self.covnBn(input=inp, filter=filter, size=size, stride=stride, padding='same', name=name + "convBn/")
        inp = layers.Add()([skip_conn, inp])
        inp = tf.nn.relu(inp)
        return inp

def ResnetCifar(obj):
    inp = obj.input
    inp = layers.Conv2D(16, 3, padding='same', kernel_regularizer=regularizers.l1_l2(l1=1e-4, l2=1e-4),name="Conv2_x0")(inp)
    inp = layers.BatchNormalization(name="Bn0")(inp)
    inp = tf.nn.relu(inp,name="relu_x0")

    for ind in range(2*obj.block_length):
        inp = obj.residualNet(inp, filter=16, size=3, stride=1, projection=False, name=f"conv2_b1_{ind}/")

    for ind in range(2*obj.block_length ):
        if ind == 0:
            inp = obj.residualNet(inp, filter=32, size=3, stride=1, projection=True, name=f"conv2_b2_{ind}/")
        else:
            inp = obj.residualNet(inp, filter=32, size=3, stride=1, projection=False, name=f"conv2_b2_{ind}/")

    for ind in range(2*obj.block_length):
        if ind == 0:
            inp = obj.residualNet(inp, filter=64, size=3, stride=1, projection=True, name=f"conv2_b3_{ind}/")
        else:
            inp = obj.residualNet(inp, filter=64, size=3, stride=1, projection=False, name=f"conv2_b3_{ind}/")

    inp = layers.GlobalAveragePooling2D(name="gbPoolingLayer_x0")(inp)
    inp = layers.Dense(obj.nb_classes,name="dense_x0")(inp)
    out = layers.Softmax(name="softmax_x0")(inp)

    model = keras.Model(obj.input, out)
    return model

if __name__ == "__main__":
    #nb_classes = 10,block_length=3
    obj = ResnetModelCifar((32, 32, 3),10,3)
    model=ResnetCifar(obj)
    model.summary()



