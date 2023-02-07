import math
from typing import Union, Collection

import tensorflow as tf

TFData = Union[tf.Tensor, tf.Variable, float]

class GMModel:
    def __init__(self, K):
        self.K = K
        self.mean = tf.Variable(tf.random.normal(shape=[K]))
        self.logvar = tf.Variable(tf.random.normal(shape=[K]))
        self.logpi = tf.Variable(tf.zeros(shape=[K]))

    @property
    def variables(self) -> Collection[TFData]:
        return self.mean, self.logvar, self.logpi

    @staticmethod
    def neglog_normal_pdf(x: TFData, mean: TFData, logvar: TFData):
        var = tf.exp(logvar)

        return 0.5 * (tf.math.log(2 * math.pi) + logvar + (x - mean) ** 2 / var)

    @tf.function
    def loss(self, data: TFData):
        l_xzk = GMModel.neglog_normal_pdf(data, self.mean, self.logvar)
        l_zk =  tf.reduce_logsumexp(self.logpi) - self.logpi
        loss = -tf.reduce_logsumexp(-(l_xzk + l_zk), axis=1)
        return loss

    def p_xz(self, x: TFData, k: int) -> TFData:
        return tf.exp(-0.5 * (x - self.mean[k]) ** 2 / tf.exp(self.logvar[k])) \
            / tf.math.sqrt(2 * tf.constant(math.pi) * tf.exp(self.logvar[k]))
 
    def p_x(self, x: TFData) -> TFData:
        res = 0.0
        for i in range(self.K):
            res += (tf.exp(self.logpi[i]) / tf.reduce_sum(tf.exp(self.logpi))) \
            * self.p_xz(x, i)
        return res