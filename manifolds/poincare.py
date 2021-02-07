"""Poincare ball manifold."""

import tensorflow as tf

from manifolds.base import Manifold
from math_utils import artanh, tanh


class PoincareBall(Manifold):
    def __init__(self, ):
        super(PoincareBall, self).__init__()
        self.name = 'PoincareBall'
        self.min_norm = 1e-15
        self.eps = {tf.float32: 4e-3, tf.float64: 1e-5}

    def safe_norm(self, x, eps=1e-12, axis=None, keep_dims=False):
        return tf.sqrt(tf.reduce_sum(x ** 2, axis=axis, keep_dims=keep_dims)+eps)

    def minkowski_dot(self, x, y, c, keepdim=True):
        hyp_norm_x = tf.sqrt(self.sqdist(x, tf.zeros_like(x), c))
        hyp_norm_y = tf.sqrt(self.sqdist(y, tf.zeros_like(y), c))
        norm_mul = tf.multiply(hyp_norm_x, hyp_norm_y)
        denominator = tf.multiply(tf.norm(x, axis=-1), tf.norm(y, axis=-1))
        denominator = tf.clip_by_value(denominator, clip_value_min=self.eps[denominator.dtype], clip_value_max=tf.reduce_max(denominator))
        cos_theta = tf.reduce_sum(tf.multiply(x,y), axis=-1) / denominator
        res = tf.multiply(norm_mul, cos_theta)
        res = tf.reshape(res, [-1])
        if keepdim:
            res = tf.expand_dims(res, -1)
        return res

    def minkowski_norm(self, u, c, keepdim=True):
        dot = self.minkowski_dot(u, u, c, keepdim=keepdim)
        return tf.sqrt(tf.clip_by_value(dot, clip_value_min=self.eps[u.dtype], clip_value_max=tf.reduce_max(dot)))

    def sqdist(self, p1, p2, c):
        sqrt_c = c ** 0.5
        dist_c = artanh(
            sqrt_c * self.safe_norm(self.mobius_add(-p1, p2, c, dim=-1), axis=-1, keep_dims=False)
        )
        dist = dist_c * 2 / sqrt_c
        return dist ** 2

    def _lambda_x(self, x, c):
        x_sqnorm = tf.reduce_sum(tf.pow(x, 2), axis=-1, keep_dims=True)
        result = 2 / (1. - c * x_sqnorm)
        max_value = tf.reduce_max(result)
        return tf.clip_by_value(result, clip_value_min=self.min_norm, clip_value_max=max_value)

    def egrad2rgrad(self, p, dp, c):
        lambda_p = self._lambda_x(p, c)
        dp /= tf.pow(lambda_p, 2)
        return dp

    def proj(self, x, c):
        norm_x = self.safe_norm(x, axis=-1, keep_dims=True)
        max_value = tf.reduce_max(norm_x)
        norm = tf.clip_by_value(norm_x, clip_value_min=self.min_norm, clip_value_max=max_value)
        maxnorm = (1 - 4e-3) / (c ** 0.5)
        cond = norm > maxnorm
        if len(x.shape) == 4:
            cond = tf.tile(cond, [1, 1, 1, x.shape[3]])
        elif len(x.shape) == 3:
            cond = tf.tile(cond, [1, 1, x.shape[2]])
        elif len(x.shape) == 2:
            cond = tf.tile(cond, [1, x.shape[1]])
        elif len(x.shape) == 1:
            cond = tf.tile(cond, [x.shape[0]])
        else:
            raise ValueError('invalid shape!')
        projected = x / norm * maxnorm
        result = tf.where(cond, projected, x)
        return result

    def proj_tan(self, u, p, c):
        return u

    def proj_tan0(self, u, c):
        return u

    def expmap(self, u, p, c):
        sqrt_c = c ** 0.5
        u_norm = self.safe_norm(u, axis=-1, keep_dims=True)
        max_value = tf.reduce_max(u_norm)
        u_norm = tf.clip_by_value(u_norm, clip_value_min=self.min_norm, clip_value_max=max_value)
        second_term = (
                tanh(sqrt_c / 2 * self._lambda_x(p, c) * u_norm)
                * u
                / (sqrt_c * u_norm)
        )
        gamma_1 = self.mobius_add(p, second_term, c)
        return gamma_1

    def logmap(self, p1, p2, c):
        sub = self.mobius_add(-p1, p2, c)
        sub_norm = self.safe_norm(sub, axis=-1, keep_dims=True)
        max_value = tf.reduce_max(sub_norm)
        sub_norm = tf.clip_by_value(sub_norm, clip_value_min=self.min_norm, clip_value_max=max_value)
        lam = self._lambda_x(p1, c)
        sqrt_c = c ** 0.5
        result = 2 / sqrt_c / lam * artanh(sqrt_c * sub_norm) * sub / sub_norm
        return result

    def expmap0(self, u, c):
        sqrt_c = c ** 0.5
        u_norm = self.safe_norm(u, axis=-1, keep_dims=True)
        max_value = tf.reduce_max(u_norm)
        u_norm = tf.clip_by_value(u_norm, clip_value_min=self.min_norm, clip_value_max=max_value)
        gamma_1 = tanh(sqrt_c * u_norm) * u / (sqrt_c * u_norm)
        return self.proj(gamma_1, c)

    def logmap0(self, p, c):
        sqrt_c = c ** 0.5
        p_norm = self.safe_norm(p, axis=-1, keep_dims=True)
        max_value = tf.reduce_max(p_norm)
        p_norm = tf.clip_by_value(p_norm, clip_value_min=self.min_norm, clip_value_max=max_value)
        scale = 1. / sqrt_c * artanh(sqrt_c * p_norm) / p_norm
        return scale * p

    def mobius_add(self, x, y, c, dim=-1):
        x2 = tf.reduce_sum(tf.pow(x, 2), axis=dim, keep_dims=True)
        y2 = tf.reduce_sum(tf.pow(y, 2), axis=dim, keep_dims=True)
        xy = tf.reduce_sum(x * y, axis=dim, keep_dims=True)
        num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
        denom = 1 + 2 * c * xy + c ** 2 * x2 * y2
        max_value = tf.reduce_max(denom)
        denom = tf.clip_by_value(denom, clip_value_min=self.min_norm, clip_value_max=max_value)
        result = num / denom
        return result

    def mobius_matvec(self, x, m, c, sparse=False):
        sqrt_c = c ** 0.5
        x_norm = self.safe_norm(x, axis=-1, keep_dims=True)
        max_value = tf.reduce_max(x_norm)
        x_norm = tf.clip_by_value(x_norm, clip_value_min=self.min_norm, clip_value_max=max_value)
        if sparse:
            mx = tf.sparse_tensor_dense_matmul(x, m)
        else:
            mx = tf.matmul(x, m)
        mx_norm = self.safe_norm(mx, axis=-1, keep_dims=True)
        max_value = tf.reduce_max(mx_norm)
        mx_norm = tf.clip_by_value(mx_norm, clip_value_min=self.min_norm, clip_value_max=max_value)
        res_c = tanh(mx_norm / x_norm * artanh(sqrt_c * x_norm)) * mx / (mx_norm * sqrt_c)
        cond = tf.reduce_all(tf.equal(mx, 0), axis=-1, keep_dims=True)
        cond = tf.tile(cond, [1, res_c.shape[1]])
        res_0 = tf.zeros_like(res_c)
        res = tf.where(cond, res_0, res_c)
        return self.proj(res, c)

    def _gyration(self, u, v, w, c, dim=-1):
        u2 = tf.reduce_sum(tf.pow(u, 2), axis=dim, keep_dims=True)
        v2 = tf.reduce_sum(tf.pow(v, 2), axis=dim, keep_dims=True)
        uv = tf.reduce_sum(u * v, axis=dim, keep_dims=True)
        uw = tf.reduce_sum(u * w, axis=dim, keep_dims=True)
        vw = tf.reduce_sum(v * w, axis=dim, keep_dims=True)
        c2 = c ** 2
        a = -c2 * uw * v2 + c * vw + 2 * c2 * uv * vw
        b = -c2 * vw * u2 - c * uw
        d = 1 + 2 * c * uv + c2 * u2 * v2
        max_value = tf.reduce_max(d)
        d = tf.clip_by_value(d, clip_value_min=self.min_norm, clip_value_max=max_value)
        result = w + 2 * (a * u + b * v) / d
        return result

    def inner(self, x, c, u, v=None, keepdim=False):
        if v is None:
            v = u
        lambda_x = self._lambda_x(x, c)
        uv = tf.reduce_sum(u * v, axis=-1, keep_dims=keepdim)
        return lambda_x ** 2 * uv

    def ptransp(self, x, y, u, c):
        lambda_x = self._lambda_x(x, c)
        lambda_y = self._lambda_x(y, c)
        result = self._gyration(y, -x, u, c) * lambda_x / lambda_y
        return result

    def ptransp_(self, x, y, u, c):
        lambda_x = self._lambda_x(x, c)
        lambda_y = self._lambda_x(y, c)
        result = self._gyration(y, -x, u, c) * lambda_x / lambda_y
        return result

    def ptransp0(self, x, u, c):
        lambda_x = self._lambda_x(x, c)
        max_value = tf.reduce_max(lambda_x)
        lambda_x = tf.clip_by_value(lambda_x, clip_value_min=self.min_norm, clip_value_max=max_value)
        result = 2 * u / lambda_x
        return result

    def to_hyperboloid(self, x, c):
        K = 1./ c
        sqrtK = K ** 0.5
        sqnorm = self.safe_norm(x, axis=1, keep_dims=True) ** 2
        return sqrtK * tf.concat([K + sqnorm, 2 * sqrtK * x], axis=1) / (K - sqnorm)

