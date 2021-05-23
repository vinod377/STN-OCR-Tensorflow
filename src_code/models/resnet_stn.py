import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from src_code.models.resnet_tf import ResnetModel_18_34


kernel_regularizer = regularizers.l1_l2(l1=1e-4, l2=1e-4)


class StnOcr(ResnetModel_18_34):
    def __init__(self, input, nb_classes, detection_filter, recognition_filter):
        """

        :param input: input image
        :param nb_classes: number of charchters
        :param detection_filter: detection network filter sizes
        :param recognition_filter: recognition network filter size
        """
        super(StnOcr, self).__init__(input_shape=input, nb_classes=nb_classes)
        self.num_labels = 3
        self.num_steps = 1
        self.detection_filter = detection_filter
        self.recognition_filter = recognition_filter

    def resnetDetRec(self, flag='detection'):
        if flag == 'detection':
            filter = self.detection_filter
        else:
            filter = self.recognition_filter
        print(filter[0], filter[1], filter[2])
        inp = self.input
        inp = layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', kernel_regularizer=kernel_regularizer)(inp)
        inp = layers.BatchNormalization()(inp)
        inp = layers.AvgPool2D(strides=2)(inp)

        inp = self.residualNet(inp=inp, filter=filter[0], size=3, stride=1, projection=False, name=f'Conv2d_block1/')

        inp = self.residualNet(inp=inp, filter=filter[1], size=3, stride=1, projection=True, name=f'Conv2d_block2/')
        inp = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(inp)

        inp = self.residualNet(inp=inp, filter=filter[2], size=3, stride=1, projection=True, name=f'Conv2d_block3/')
        inp = layers.AvgPool2D(pool_size=5)(inp)
        inp = layers.Flatten()(inp)

        if flag == 'detection':
            inp = layers.Reshape((self.num_steps, -1))(inp)
            inp = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(inp)
            theta = layers.TimeDistributed(layers.Dense(6, activation='sigmoid'))(inp)
            return theta
        else:
            inp = layers.Dense(256, activation='relu')(inp)
            classifiers = []
            for i in range(self.num_labels):
                inp = layers.Dense(11)(inp)
                inp = layers.Reshape((self.num_steps, -1, 11))(inp)
                inp = tf.expand_dims(inp, axis=1)
                classifiers.append(inp)

            inp = layers.concatenate(classifiers, axis=1)
            inp = layers.Reshape((-1, 11))(inp)
            inp = tf.keras.activations.softmax(inp)
            print(inp.shape)
            return inp


if __name__ == "__main__":
    detection_filter = [32, 48, 48]
    recognition_filter = [32,64,128]
    stn_obj = StnOcr((128, 128, 1), 10, detection_filter,recognition_filter)

    stn_obj.resnetDetRec('detection')



