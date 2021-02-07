"""Euclidean manifold."""

from manifolds.base import Manifold
import tensorflow as tf


class Euclidean(Manifold):
    def __init__(self):
        super(Euclidean, self).__init__()
        self.name = 'Euclidean'

    def minkowski_dot(self, x, y, c, dim=-1, keepdim=True):
        res = tf.reduce_sum(x * y, axis=dim)
        if keepdim:
            res = tf.expand_dims(res, -1)
        return res

    def sqdist(self, p1, p2, c):
        return tf.reduce_sum(tf.square(p1-p2), -1)

    def egrad2rgrad(self, p, dp, c):
        return dp

    def proj(self, p, c):
        return p

    def proj_tan(self, u, p, c):
        return u

    def proj_tan0(self, u, c):
        return u

    def expmap(self, u, p, c):
        return p + u

    def logmap(self, p1, p2, c):
        return p2 - p1

    def expmap0(self, u, c):
        return u

    def logmap0(self, p, c):
        return p

    def mobius_add(self, x, y, c, dim=-1):
        return x + y

    def mobius_matvec(self, x, y, c, sparse=False):
        if sparse:
            res = tf.sparse_tensor_dense_matmul(x, y)
        else:
            res = tf.matmul(x, y)
        return res

    def init_weights(self, w, c, irange=1e-5):
        w.data.uniform_(-irange, irange)
        return w

    def inner(self, p, c, u, v=None, keepdim=False):
        if v is None:
            v = u
        return (u * v).sum(dim=-1, keepdim=keepdim)

    def ptransp(self, x, y, v, c):
        return v

    def ptransp0(self, x, v, c):
        return x + v
