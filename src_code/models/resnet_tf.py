"""
Implementation of Resnet-18 and Resnet-34 as proposed by kaiming He in
"Deep Residual Learning for Image Recognition (https://arxiv.org/abs/1512.03385)".

For Model creation , Create object of class ResnetModel_18_43 by passing
input shape(minimum 32X32 dim) ,number of classes and flag value which can be
"resnet-18" or "resnet-34" depending on the resnet version one wish to create,
as parametrs.Then call "Residualnetwork" function by passing ResnetModel_18_43 object
as parameter.The deafault model is resnet-18.Projection layer is used to increase the low
dimensional feature map to add high dimensional feature map.
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers

class ResnetModel_18_34:
    def __init__(self,input_shape,nb_classes=10,flag="resnet-18"):
        self.input = layers.Input(shape=input_shape)
        self.nb_classes = nb_classes
        self.flag = flag

    def convBnRelu(self,input,filter,size,projection,name):
        if projection:
            stride=2
        else:
            stride=1
        inp = layers.Conv2D(filters=filter,kernel_size=size,padding = 'same',
                            strides=stride,kernel_regularizer=regularizers.l1_l2(l1=1e-4, l2=1e-4),
                            name=name+"conv")(input)
        inp = layers.BatchNormalization(name=name+"Bn")(inp)
        inp = tf.nn.relu(inp,name=name+"relu")
        return inp

    def covnBn(self,input,filter,size,stride,name,padding='valid'):
        inp = layers.Conv2D(filters=filter, kernel_size=size, strides=stride,padding=padding,
                            kernel_regularizer=regularizers.l1_l2(l1=1e-4, l2=1e-4),name=name+"conv")(input)
        inp = layers.BatchNormalization(name=name+"Bn")(inp)
        return inp

    def residualNet(self,inp,filter,size,stride,projection,name):
        if projection:
            skip_conn = self.covnBn(input=inp,filter=filter,size=1,stride=2,name=name+"skip_conn/")
        else:
            skip_conn = inp

        inp = self.convBnRelu(input = inp,filter=filter,size=size,projection=projection,name=name+"convBnRelu/")
        inp = self.covnBn(input=inp, filter=filter, size=size, stride=stride,padding='same',name=name+"convBn/")

        inp = layers.Add(name=name+"merge")([skip_conn,inp])
        inp = tf.nn.relu(inp,name=name+"relu")
        return inp

def Residualnetwork(obj):

    if obj.flag == "resnet-18":
        block_size = [2,2,2,2]
    elif obj.flag == "resnet-34":
        block_size=[3,4,6,3]
    else:
        print("model flag is incorrect, flag is either resnet-18 or resnet-34")

    input = obj.input
    inp = layers.Conv2D(filters=64,kernel_size=7,
                        strides=2,padding='same',kernel_regularizer=regularizers.l1_l2(l1=1e-4, l2=1e-4),
                        name = "Conv2_x0")(input)

    inp = layers.MaxPooling2D(pool_size=(3,3),strides=2,padding='same',name="maxpooling_x0")(inp)
    inp = layers.BatchNormalization(name="Bn0")(inp)
    inp = tf.nn.relu(inp,name="relu_x0")

    for ind in range(block_size[0]):
        inp = obj.residualNet(inp, filter=64, size=3, stride=1, projection=False, name=f'Conv2_b1_{ind}/')

    for ind in range(block_size[1]):
        if ind==0:
            inp = obj.residualNet(inp, filter=128, size=3, stride=1, projection=True, name=f'Conv2_b2_{ind}/')
        else:
            inp = obj.residualNet(inp, filter=128, size=3, stride=1, projection=False, name=f'Conv2_b2_{ind}/')

    for ind in range(block_size[2]):
        if ind==0:
            inp = obj.residualNet(inp, filter=256, size=3, stride=1, projection=True, name=f'Conv2_b3_{ind}/')
        else:
            inp = obj.residualNet(inp, filter=256, size=3, stride=1, projection=False, name=f'Conv2_b3_{ind}/')

    for ind in range(block_size[3]):
        if ind==0:
            inp = obj.residualNet(inp, filter=512, size=3, stride=1, projection=True, name=f'Conv2_b4_{ind}/')
        else:
            inp = obj.residualNet(inp, filter=512, size=3, stride=1, projection=False, name=f'Conv2_b4_{ind}/')

    inp = layers.GlobalAveragePooling2D(name="gbPoolingLayer_x0")(inp)
    inp = layers.Dense(obj.nb_classes,name="dense_x0")(inp)
    out = layers.Softmax(name="softmax_x0")(inp)
    model = keras.Model(input,out)
    return model

if __name__ == "__main__":
    flag = "resnet-18"
    obj = ResnetModel_18_34([224, 224, 3],1000,flag)
    model = Residualnetwork(obj)
    model.summary()




