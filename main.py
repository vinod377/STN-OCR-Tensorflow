"""
The script creates localisation network using resnet_stn.py by importing
the class SpatialTransformerNetwork which contains the architecture of
Resnet as proposed by the author, for Detection Network the filters [32,48,48]
and for Recognition Network the filters are [32,64,128]. The output of Localisation
Network is n theta(6 num_hidden unit), which is fed to SpatialTransformer network
along with initial input, It produces n sampled Image output which is fed to
Recognition Network.
"""
import tensorflow as tf
from src_code.models.stn_network import SpatialTransformerNetwork
from src_code.models.resnet_stn import StnOcr
from tensorflow.keras import layers

def stnOcrModel():
    num_steps = 1
    detection_filter = [32, 48, 48]
    recognition_filter = [32,64,128]
    stn_detection = StnOcr((600,150,1), 10, detection_filter,recognition_filter)

    flag = 'detection'
    theta = stn_detection.resnetDetRec(flag)  # localisation Network
    inp = stn_detection.input
    stn_obj = SpatialTransformerNetwork(inp,theta,num_steps) # Grid genrator
    sampled_image = stn_obj.image_sampling()  # sampled image from grid genrator

    flag = 'Recognition'
    out = stn_detection.resnetDetRec(sampled_image,flag)   # Recognition Model
    stn_model = tf.keras.Model(inp,out)
    stn_model.summary()


if __name__ == '__main__':
    stnOcrModel()
