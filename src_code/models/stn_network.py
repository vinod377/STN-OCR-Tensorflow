"""
Implementation of spatial Transformer Network with Bilinear Interpolation
"""
import tensorflow as tf
from tensorflow.keras import layers
from src_code.models.bilinear_interpolation import bilinear_sampler

class SpatialTransformerNetwork:
    def __init__(self, input, theta, num_steps, out=None):
        """
        :param input: Input feature
        :param theta: Localistaion network ouput with shape [None,1,6]
        :param num_steps: Num steps, it refers to word or lines we wish to localize
        :param out:output dimension of sampled image
        """
        self.input = input
        self.theta = theta
        self.num_steps = num_steps
        self.out_dims = out

    def grid_genrator(self):
        shape = self.input.shape
        batch_sz = 32  # shape[0]
        print(batch_sz)
        H = shape[1]
        W = shape[2]
        C = shape[3]
        theta = layers.Reshape((self.num_steps, 2, 3))(self.theta)

        grid_num = theta.shape[1]
        grid_h = H
        grid_w = W
        X = tf.linspace(1.0, -1.0, grid_w)
        Y = tf.linspace(1.0, -1.0, grid_h)
        grid_xt, grid_yt = tf.meshgrid(X, Y)
        grid_xt = tf.reshape(grid_xt, [-1])
        grid_yt = tf.reshape(grid_yt, [-1])
        ones = tf.ones_like(grid_xt)
        samplig_grid = tf.stack([grid_xt, grid_yt, ones])
        samplig_grid = tf.expand_dims(samplig_grid, axis=0)
        samplig_grid = tf.tile(samplig_grid, tf.stack([grid_num, 1, 1]))
        samplig_grid = tf.expand_dims(samplig_grid, axis=0)
        gen_grid = tf.matmul(theta, samplig_grid)
        gen_grid = layers.Reshape([self.num_steps, 2, 600, 150])(gen_grid)
        gen_grid_x = gen_grid[:, :, 0, :, :]
        gen_grid_y = gen_grid[:, :, 1, :, :]

        return gen_grid_x, gen_grid_y

    def image_sampling(self):
        gen_grid_x, gen_grid_y = self.grid_genrator()
        stacked_image = tf.expand_dims(self.input, axis=1)
        stacked_image = tf.tile(stacked_image, tf.constant([self.num_steps, 2, 1, 1, 1], tf.int32))
        output_feature_list = []
        for i in range(self.num_steps):
            output_feature_list.append(
                bilinear_sampler(stacked_image[:, i, :, :, :], gen_grid_x[:, i, :, :], gen_grid_y[:, i, :, :]))
        output_feature = tf.concat(output_feature_list, axis=1)
        return output_feature


if __name__ == "__main__":
    input = layers.Input(shape=(600, 150, 1))
    num_steps = 1
    theta = tf.ones((1,num_steps, 6), dtype=tf.float32)
    stn_obj = SpatialTransformerNetwork(input,theta,num_steps)
    stn_obj.image_sampling()


