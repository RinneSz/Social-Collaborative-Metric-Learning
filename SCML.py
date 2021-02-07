import functools
import numpy
import tensorflow as tf
import toolz
from evaluator import RecallEvaluator
from sampler import social_WarpSampler, rating_WarpSampler
import time
import argparse
import os
import pickle as pkl
import manifolds
import math


def doublewrap(function):
    """
    A decorator decorator, allowing to use the decorator to be used without
    parentheses if not arguments are provided. All arguments must be optional.
    """

    @functools.wraps(function)
    def decorator(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            return function(args[0])
        else:
            return lambda wrapee: function(wrapee, *args, **kwargs)

    return decorator


@doublewrap
def define_scope(function, scope=None, *args, **kwargs):
    """
    A decorator for functions that define TensorFlow operations. The wrapped
    function will only be executed once. Subsequent calls to it will directly
    return the result so that operations are added to the graph only once.
    The operations added by the function live within a tf.variable_scope(). If
    this decorator is used with arguments, they will be forwarded to the
    variable scope. The scope name defaults to the name of the wrapped
    function.
    """
    attribute = '_cache_' + function.__name__
    name = scope or function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            with tf.variable_scope(name, *args, **kwargs):
                setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator


class CML(object):
    def __init__(self,
                 manifold_name,
                 n_users,
                 n_items,
                 embed_dim=20,
                 rating_margin=1.5,
                 social_margin=1.5,
                 master_learning_rate=0.1,
                 clip_norm=1.0,
                 lambda_social=0.0,
                 center_init=False
                 ):
        """

        :param n_users: number of users i.e. |U|
        :param n_items: number of items i.e. |V|
        :param embed_dim: embedding size i.e. K (default 20)
        :param rating_margin: hinge loss threshold for rating part
        :param social_margin: hinge loss threshold for social part
        :param master_learning_rate: master learning rate for AdaGrad
        :param clip_norm: clip norm threshold (default 1.0)
        """

        self.center_init = center_init
        self.manifold_name = manifold_name
        self.manifold = getattr(manifolds, manifold_name)()
        self.c = tf.ones([1], dtype=tf.float32)
        self.lambda_social = lambda_social

        self.n_users = n_users
        self.n_items = n_items
        self.embed_dim = embed_dim

        self.clip_norm = clip_norm
        self.rating_margin = rating_margin
        self.social_margin = social_margin

        self.master_learning_rate = master_learning_rate

        self.user_positive_items_pairs = tf.placeholder(tf.int32, [None, 2])
        self.negative_samples = tf.placeholder(tf.int32, [None, None])

        self.positive_social_pairs = tf.placeholder(tf.int32, [None, 2])
        self.negative_social_samples = tf.placeholder(tf.int32, [None, None])

        self.score_user_ids = tf.placeholder(tf.int32, [None])
        self.score_neg_list = tf.placeholder(tf.int32, [None, None])

        self.user_embeddings
        self.item_embeddings
        self.embedding_loss
        self.social_loss
        self.loss
        self.optimize


    @define_scope
    def user_embeddings(self):
        if self.center_init:
            alpha = math.pow(3*0.0001/(2*self.embed_dim), 1/3)
            return tf.Variable(tf.random_uniform([self.n_users, self.embed_dim], minval=-alpha, maxval=alpha))
        else:
            return tf.Variable(tf.random_normal([self.n_users, self.embed_dim],
                                                stddev=1 / (self.embed_dim ** 0.5), dtype=tf.float32))

    @define_scope
    def item_embeddings(self):
        if self.center_init:
            alpha = math.pow(3*0.0001/(2*self.embed_dim), 1/3)
            return tf.Variable(tf.random_uniform([self.n_items, self.embed_dim], minval=-alpha, maxval=alpha))
        else:
            return tf.Variable(tf.random_normal([self.n_items, self.embed_dim],
                                                stddev=1 / (self.embed_dim ** 0.5), dtype=tf.float32))

    @define_scope
    def embedding_loss(self):
        """
        :return: the distance metric loss
        """
        # Let
        # N = batch size,
        # K = embedding size,
        # W = number of negative samples per a user-positive-item pair

        hyp_user_embeddings = self.manifold.expmap0(self.user_embeddings, self.c)
        hyp_item_embeddings = self.manifold.expmap0(self.item_embeddings, self.c)

        # user embedding (N, K)
        users = tf.nn.embedding_lookup(hyp_user_embeddings,
                                       self.user_positive_items_pairs[:, 0],
                                       name="users")

        # positive item embedding (N, K)
        pos_items = tf.nn.embedding_lookup(hyp_item_embeddings, self.user_positive_items_pairs[:, 1],
                                           name="pos_items")

        # positive item to user distance (N)
        pos_distances = self.manifold.sqdist(users, pos_items, self.c)

        # negative item embedding (N, K)
        neg_items = tf.nn.embedding_lookup(hyp_item_embeddings, self.negative_samples[:, 0])

        # distance to negative items (N)
        closest_negative_item_distances = self.manifold.sqdist(users, neg_items, self.c)

        # compute hinge loss (N)
        loss_per_pair = tf.maximum(pos_distances - closest_negative_item_distances + self.rating_margin, 0,
                                   name="rating_pair_loss")

        # the embedding loss
        loss = tf.reduce_sum(loss_per_pair, name="rating_loss")

        return loss


    @define_scope
    def social_loss(self):
        """
        :return: the distance metric loss
        """
        # Let
        # N = batch size,
        # K = embedding size,
        # W = number of negative samples per a user-positive-item pair

        hyp_user_embeddings = self.manifold.expmap0(self.user_embeddings, self.c)

        # user embedding (N, K)
        users = tf.nn.embedding_lookup(hyp_user_embeddings,
                                       self.positive_social_pairs[:, 0],
                                       name="social_u")

        # positive item embedding (N, K)
        pos_neighbors = tf.nn.embedding_lookup(hyp_user_embeddings, self.positive_social_pairs[:, 1],
                                           name="pos_neighbors")

        # positive item to user distance (N)
        pos_distances = self.manifold.sqdist(users, pos_neighbors, self.c)

        # negative item embedding (N, K)
        neg_neighbors = tf.nn.embedding_lookup(hyp_user_embeddings, self.negative_social_samples[:, 0])
        # distance to negative items (N)
        closest_negative_item_distances = self.manifold.sqdist(users, neg_neighbors, self.c)

        # compute hinge loss (N)
        loss_per_pair = tf.maximum(pos_distances - closest_negative_item_distances + self.social_margin, 0,
                                   name="social_pair_loss")

        # the embedding loss
        loss = tf.reduce_sum(loss_per_pair, name="social_loss")

        return loss

    @define_scope
    def loss(self):
        """
        :return: the total loss = embedding loss + feature loss
        """
        loss = self.embedding_loss + self.lambda_social*self.social_loss
        return loss

    @define_scope
    def clip_by_norm_op(self):
        return [tf.assign(self.user_embeddings, tf.clip_by_norm(self.user_embeddings, self.clip_norm, axes=[1])),
                tf.assign(self.item_embeddings, tf.clip_by_norm(self.item_embeddings, self.clip_norm, axes=[1]))]

    @define_scope
    def optimize(self):
        gds = []
        gds.append(tf.train
                   .AdamOptimizer(self.master_learning_rate)
                   .minimize(self.loss, var_list=[self.user_embeddings, self.item_embeddings]))

        with tf.control_dependencies(gds):
            return gds + [self.clip_by_norm_op]

    @define_scope
    def item_scores(self):
        hyp_user_embeddings = self.manifold.expmap0(self.user_embeddings, self.c)
        hyp_item_embeddings = self.manifold.expmap0(self.item_embeddings, self.c)

        # (N_USER_IDS, 1, K)
        user = tf.expand_dims(tf.nn.embedding_lookup(hyp_user_embeddings, self.score_user_ids), 1)
        # (N_USER_IDS, N_ITEM, K)
        item = tf.nn.embedding_lookup(hyp_item_embeddings, self.score_neg_list)
        return -self.manifold.sqdist(user, item, self.c)

    @define_scope
    def save_embeddings(self):
        hyp_users = self.manifold.expmap0(self.user_embeddings, self.c)
        hyp_items = self.manifold.expmap0(self.item_embeddings, self.c)
        return hyp_users, hyp_items


def optimize(model, rating_sampler, social_sampler, train, valid, test, neg_samples_list):
    """
    Optimize the model. TODO: implement early-stopping
    :param model: model to optimize
    :param rating_sampler: mini-batch sampler for rating part
    :param social_sampler: mini-batch sampler for social part
    :param train: train user-item matrix
    :param valid: validation user-item matrix
    :return: None
    """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    # sample some users to calculate recall validation
    valid_users = list(set(valid.nonzero()[0]))
    test_users = list(set(test.nonzero()[0]))

    best_val_hr_list = None
    best_val_ndcg_list = None
    test_hrs = None
    test_ndcgs = None
    best_val_ndcg_20 = -1
    epoch_count = 0
    endure_count = 0
    if manifold_name == 'Euclidean':
        num_user_chunck = 1000
    else:
        num_user_chunck = 100
    while True:
        epoch_count += 1
        endure_count += 1
        start = time.time()
        # create evaluator on validation set
        validation_recall = RecallEvaluator(model, train, valid)
        # compute hr and ndcg on validate set
        valid_hrs = [[],[],[],[],[],[]]
        valid_ndcgs = [[],[],[],[],[],[]]

        for user_chunk in toolz.partition_all(num_user_chunck, valid_users):
            hrs_1,hrs_5,hrs_10,hrs_15,hrs_20,hrs_50,ndcgs_1,ndcgs_5,ndcgs_10,ndcgs_15,ndcgs_20,ndcgs_50=validation_recall.eval(sess,user_chunk,neg_samples_list)
            valid_hrs[0].extend(hrs_1)
            valid_hrs[1].extend(hrs_5)
            valid_hrs[2].extend(hrs_10)
            valid_hrs[3].extend(hrs_15)
            valid_hrs[4].extend(hrs_20)
            valid_hrs[5].extend(hrs_50)
            valid_ndcgs[0].extend(ndcgs_1)
            valid_ndcgs[1].extend(ndcgs_5)
            valid_ndcgs[2].extend(ndcgs_10)
            valid_ndcgs[3].extend(ndcgs_15)
            valid_ndcgs[4].extend(ndcgs_20)
            valid_ndcgs[5].extend(ndcgs_50)
        valid_hrs[0] = numpy.mean(valid_hrs[0])
        valid_hrs[1] = numpy.mean(valid_hrs[1])
        valid_hrs[2] = numpy.mean(valid_hrs[2])
        valid_hrs[3] = numpy.mean(valid_hrs[3])
        valid_hrs[4] = numpy.mean(valid_hrs[4])
        valid_hrs[5] = numpy.mean(valid_hrs[5])
        valid_ndcgs[0] = numpy.mean(valid_ndcgs[0])
        valid_ndcgs[1] = numpy.mean(valid_ndcgs[1])
        valid_ndcgs[2] = numpy.mean(valid_ndcgs[2])
        valid_ndcgs[3] = numpy.mean(valid_ndcgs[3])
        valid_ndcgs[4] = numpy.mean(valid_ndcgs[4])
        valid_ndcgs[5] = numpy.mean(valid_ndcgs[5])

        val_ndcg_20 = valid_ndcgs[-2]
        if val_ndcg_20 > best_val_ndcg_20:
            endure_count = 0
            best_val_ndcg_20 = val_ndcg_20
            best_val_hr_list = valid_hrs
            best_val_ndcg_list = valid_ndcgs
            test_hrs = [[], [], [], [], [], []]
            test_ndcgs = [[], [], [], [], [], []]
            test_recall = RecallEvaluator(model, train, test)

            for user_chunk in toolz.partition_all(num_user_chunck, test_users):
                hrs_1, hrs_5, hrs_10, hrs_15, hrs_20, hrs_50, ndcgs_1, ndcgs_5, ndcgs_10, ndcgs_15, ndcgs_20, ndcgs_50 = test_recall.eval(
                    sess, user_chunk,neg_samples_list)
                test_hrs[0].extend(hrs_1)
                test_hrs[1].extend(hrs_5)
                test_hrs[2].extend(hrs_10)
                test_hrs[3].extend(hrs_15)
                test_hrs[4].extend(hrs_20)
                test_hrs[5].extend(hrs_50)
                test_ndcgs[0].extend(ndcgs_1)
                test_ndcgs[1].extend(ndcgs_5)
                test_ndcgs[2].extend(ndcgs_10)
                test_ndcgs[3].extend(ndcgs_15)
                test_ndcgs[4].extend(ndcgs_20)
                test_ndcgs[5].extend(ndcgs_50)
            test_hrs[0] = numpy.mean(test_hrs[0])
            test_hrs[1] = numpy.mean(test_hrs[1])
            test_hrs[2] = numpy.mean(test_hrs[2])
            test_hrs[3] = numpy.mean(test_hrs[3])
            test_hrs[4] = numpy.mean(test_hrs[4])
            test_hrs[5] = numpy.mean(test_hrs[5])
            test_ndcgs[0] = numpy.mean(test_ndcgs[0])
            test_ndcgs[1] = numpy.mean(test_ndcgs[1])
            test_ndcgs[2] = numpy.mean(test_ndcgs[2])
            test_ndcgs[3] = numpy.mean(test_ndcgs[3])
            test_ndcgs[4] = numpy.mean(test_ndcgs[4])
            test_ndcgs[5] = numpy.mean(test_ndcgs[5])
        else:
            if endure_count >= 10:
                break

        print(
            "\n[Epoch %d] val HR: [%.4f,%.4f,%.4f,%.4f,%.4f,%.4f], val NDCG: [%.4f,%.4f,%.4f,%.4f,%.4f,%.4f], best val HR: [%.4f,%.4f,%.4f,%.4f,%.4f,%.4f]"
            ", best val NDCG: [%.4f,%.4f,%.4f,%.4f,%.4f,%.4f], test HR: [%.4f,%.4f,%.4f,%.4f,%.4f,%.4f], test NDCG: [%.4f,%.4f,%.4f,%.4f,%.4f,%.4f]" %
            (epoch_count, valid_hrs[0], valid_hrs[1], valid_hrs[2], valid_hrs[3], valid_hrs[4], valid_hrs[5],
             valid_ndcgs[0], valid_ndcgs[1], valid_ndcgs[2], valid_ndcgs[3], valid_ndcgs[4], valid_ndcgs[5],
             best_val_hr_list[0], best_val_hr_list[1], best_val_hr_list[2], best_val_hr_list[3], best_val_hr_list[4],
             best_val_hr_list[5],
             best_val_ndcg_list[0], best_val_ndcg_list[1], best_val_ndcg_list[2], best_val_ndcg_list[3],
             best_val_ndcg_list[4], best_val_ndcg_list[5],
             test_hrs[0], test_hrs[1], test_hrs[2], test_hrs[3], test_hrs[4], test_hrs[5],
             test_ndcgs[0], test_ndcgs[1], test_ndcgs[2], test_ndcgs[3], test_ndcgs[4], test_ndcgs[5]))

        # train model
        losses = []
        # run n mini-batches
        time1 = time.time()
        for _ in range(EVALUATION_EVERY_N_BATCHES):
            user_pos, neg = rating_sampler.next_batch()
            social_pos, social_neg = social_sampler.next_batch()
            _, loss = sess.run((model.optimize, model.loss),
                               {model.user_positive_items_pairs: user_pos,
                                model.negative_samples: neg,
                                model.positive_social_pairs: social_pos,
                                model.negative_social_samples: social_neg})
            losses.append(loss)

        end = time.time()
        print('time1:',time1-start, ' time2:',end-time1)
        print("\nTraining loss {} finisded in {}s".format(numpy.mean(losses), end-start))
    print("\nFinished. Best val HR: [%.4f,%.4f,%.4f,%.4f,%.4f,%.4f]"
          ", best val NDCG: [%.4f,%.4f,%.4f,%.4f,%.4f,%.4f], test HR: [%.4f,%.4f,%.4f,%.4f,%.4f,%.4f], test NDCG: [%.4f,%.4f,%.4f,%.4f,%.4f,%.4f]" %
          (best_val_hr_list[0], best_val_hr_list[1], best_val_hr_list[2], best_val_hr_list[3], best_val_hr_list[4],
           best_val_hr_list[5],
           best_val_ndcg_list[0], best_val_ndcg_list[1], best_val_ndcg_list[2], best_val_ndcg_list[3],
           best_val_ndcg_list[4], best_val_ndcg_list[5],
           test_hrs[0], test_hrs[1], test_hrs[2], test_hrs[3], test_hrs[4], test_hrs[5],
           test_ndcgs[0], test_ndcgs[1], test_ndcgs[2], test_ndcgs[3], test_ndcgs[4], test_ndcgs[5]))
    # hyp_user_embeddings, hyp_item_embeddings = sess.run(model.save_embeddings)
    # pkl.dump(hyp_user_embeddings, open(), 'wb'))
    # pkl.dump(hyp_item_embeddings, open(), 'wb'))
    # print('Embeddings Saved.')
    rating_sampler.close()
    social_sampler.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifold', '-m', choices=['Euclidean', 'Hyperboloid', 'PoincareBall'], default='Euclidean',
                        help='embedding manifold')
    parser.add_argument('--dimension', '-d', type=int, default=10, help='number of dimensions of the latent factors')
    parser.add_argument('--iteration', '-i', type=int, default=1000, help='number of iterations')
    parser.add_argument('--lr', '-l', type=float, default=0.001, help='learning rate')
    parser.add_argument('--decay', '-ld', type=int, default=None,
                        help='the iteration where lr decreases by multiplying 0.1')
    parser.add_argument('--dataset', '-ds', choices=['ciao', 'epinions'], help='name of the dataset')
    parser.add_argument('--cuda', choices=['0', '1'], default='0')
    parser.add_argument('--random_seed', type=int, default=1000)
    parser.add_argument('--rating_margin', type=float, default=0.0, help='margin of the rating loss')
    parser.add_argument('--social_margin', type=float, default=0.0, help='margin of the social loss')
    parser.add_argument('--Lambda', type=float, default=0.0, help='lambda of the social loss')
    args = parser.parse_args()
    EMBED_DIM = args.dimension
    manifold_name = args.manifold
    dataset = args.dataset
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    # random_seed = args.random_seed
    # numpy.random.seed(random_seed)
    # tf.set_random_seed(random_seed)

    BATCH_SIZE = 50000
    N_NEGATIVE = 2
    EVALUATION_EVERY_N_BATCHES = 500
    if dataset == 'ciao':
        data_path = '/home/uqhche32/sixiao/datasets/ciao/topn'
    elif dataset == 'epinions':
        data_path = '/home/uqhche32/sixiao/datasets/epinions/topn'
    f = open(os.path.join(data_path, 'rating_test_adj.pkl'), 'rb')
    rating_test = pkl.load(f).todok()
    f.close()
    f = open(os.path.join(data_path, 'rating_train_adj.pkl'), 'rb')
    rating_train = pkl.load(f).todok()
    f.close()
    f = open(os.path.join(data_path, 'rating_val_adj.pkl'), 'rb')
    rating_valid = pkl.load(f).todok()
    f.close()
    f = open(os.path.join(data_path, 'social_adj.pkl'), 'rb')
    social_adj = pkl.load(f).todok()
    f.close()
    f = open(os.path.join(data_path, 'neg_samples_list.pkl'), 'rb')
    neg_samples_list = pkl.load(f)
    f.close()
    n_users, n_items = rating_train.shape

    start = time.time()

    # create warp sampler
    rating_sampler = rating_WarpSampler(rating_train, batch_size=BATCH_SIZE, n_negative=N_NEGATIVE, check_negative=True)
    social_sampler = social_WarpSampler(social_adj, batch_size=BATCH_SIZE, n_negative=N_NEGATIVE, check_negative=True)

    if manifold_name == 'Euclidean':
        CLIP_NORM = 1.0
    elif manifold_name == 'PoincareBall':
        CLIP_NORM = 3.0
    else:
        CLIP_NORM = 14.9
    CENTER_INIT = True
    RATING_MARGIN = args.rating_margin
    SOCIAL_MARGIN = args.social_margin
    LAMBDA = args.Lambda

    model = CML(manifold_name,
                n_users,
                n_items,
                # size of embedding
                embed_dim=EMBED_DIM,
                # the size of hinge loss margin.
                social_margin=SOCIAL_MARGIN,
                rating_margin=RATING_MARGIN,
                # clip the embedding so that their norm <= clip_norm
                clip_norm=CLIP_NORM,
                # learning rate for AdaGrad
                master_learning_rate=args.lr,
                lambda_social=LAMBDA,
                center_init=CENTER_INIT
                )

    end = time.time()
    print('preprocessing complete in %ds' % (end-start))
    optimize(model, rating_sampler, social_sampler, rating_train, rating_valid, rating_test, neg_samples_list)
