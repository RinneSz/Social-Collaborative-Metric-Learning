import tensorflow as tf
from scipy.sparse import lil_matrix
import math

'''negative sampling'''
class RecallEvaluator(object):
    def __init__(self, model, train_user_item_matrix, test_user_item_matrix):
        self.model = model
        self.train_user_item_matrix = lil_matrix(train_user_item_matrix)
        self.test_user_item_matrix = lil_matrix(test_user_item_matrix)
        n_users = train_user_item_matrix.shape[0]
        self.user_to_test_set = {u: set(self.test_user_item_matrix.rows[u])
                                 for u in range(n_users) if self.test_user_item_matrix.rows[u]}

        if self.train_user_item_matrix is not None:
            self.user_to_train_set = {u: set(self.train_user_item_matrix.rows[u])
                                      for u in range(n_users) if self.train_user_item_matrix.rows[u]}

    def eval(self, sess, users, neg_samples_list, k=[1,5,10,15,20,50]):
        # compute the HRs and NDCGs for each user
        user_neg_list = []
        for i in users:
            test_set = list(self.user_to_test_set.get(i, set()))
            user_neg_list.append(neg_samples_list[i]+test_set)
        _, user_tops = sess.run(tf.nn.top_k(self.model.item_scores, k[-1]+1),
                                {self.model.score_user_ids: users,
                                 self.model.score_neg_list: user_neg_list})
        hrs_1, hrs_5, hrs_10, hrs_15, hrs_20, hrs_50 = [], [], [], [], [], []
        ndcgs_1, ndcgs_5, ndcgs_10, ndcgs_15, ndcgs_20, ndcgs_50 = [], [], [], [], [], []
        for user_id, tops in zip(users, user_tops):
            test_set = self.user_to_test_set.get(user_id, set())
            top_n_items = 0
            hits_1, hits_5, hits_10, hits_15, hits_20, hits_50 = 0,0,0,0,0,0
            nd_1, nd_5, nd_10, nd_15, nd_20, nd_50 = 0, 0, 0, 0, 0, 0
            for i in tops:
                top_n_items += 1
                if i == 1000:
                    if top_n_items <= 50:
                        hits_50 += 1
                        nd_50 += 1 / math.log2(top_n_items + 1)
                    if top_n_items <= 20:
                        hits_20 += 1
                        nd_20 += 1 / math.log2(top_n_items + 1)
                    if top_n_items <= 15:
                        hits_15 += 1
                        nd_15 += 1 / math.log2(top_n_items + 1)
                    if top_n_items <= 10:
                        hits_10 += 1
                        nd_10 += 1 / math.log2(top_n_items + 1)
                    if top_n_items <= 5:
                        hits_5 += 1
                        nd_5 += 1 / math.log2(top_n_items + 1)
                    if top_n_items <= 1:
                        hits_1 += 1
                        nd_1 += 1 / math.log2(top_n_items + 1)
                if top_n_items == k[-1]:
                    break
            hrs_1.append(hits_1 / float(len(test_set)))
            hrs_5.append(hits_5 / float(len(test_set)))
            hrs_10.append(hits_10 / float(len(test_set)))
            hrs_15.append(hits_15 / float(len(test_set)))
            hrs_20.append(hits_20 / float(len(test_set)))
            hrs_50.append(hits_50 / float(len(test_set)))
            ndcgs_1.append(nd_1 / float(len(test_set)))
            ndcgs_5.append(nd_5 / float(len(test_set)))
            ndcgs_10.append(nd_10 / float(len(test_set)))
            ndcgs_15.append(nd_15 / float(len(test_set)))
            ndcgs_20.append(nd_20 / float(len(test_set)))
            ndcgs_50.append(nd_50 / float(len(test_set)))
        return hrs_1, hrs_5, hrs_10, hrs_15, hrs_20, hrs_50, ndcgs_1, ndcgs_5, ndcgs_10, ndcgs_15, ndcgs_20, ndcgs_50


'''without negative sampling'''
# class RecallEvaluator(object):
#     def __init__(self, model, train_user_item_matrix, test_user_item_matrix):
#         self.model = model
#         self.train_user_item_matrix = lil_matrix(train_user_item_matrix)
#         self.test_user_item_matrix = lil_matrix(test_user_item_matrix)
#         n_users = train_user_item_matrix.shape[0]
#         self.user_to_test_set = {u: set(self.test_user_item_matrix.rows[u])
#                                  for u in range(n_users) if self.test_user_item_matrix.rows[u]}
#
#         if self.train_user_item_matrix is not None:
#             self.user_to_train_set = {u: set(self.train_user_item_matrix.rows[u])
#                                       for u in range(n_users) if self.train_user_item_matrix.rows[u]}
#             self.max_train_count = max(len(row) for row in self.train_user_item_matrix.rows)
#         else:
#             self.max_train_count = 0
#
#     def eval(self, sess, users, k=[1,5,10,15,20,50]):
#         _, user_tops = sess.run(tf.nn.top_k(self.model.item_scores, k[-1] + self.max_train_count),
#                                 {self.model.score_user_ids: users})
#         hrs_1, hrs_5, hrs_10, hrs_15, hrs_20, hrs_50 = [], [], [], [], [], []
#         ndcgs_1, ndcgs_5, ndcgs_10, ndcgs_15, ndcgs_20, ndcgs_50 = [], [], [], [], [], []
#         for user_id, tops in zip(users, user_tops):
#             train_set = self.user_to_train_set.get(user_id, set())
#             test_set = self.user_to_test_set.get(user_id, set())
#             top_n_items = 0
#             hits_1, hits_5, hits_10, hits_15, hits_20, hits_50 = 0,0,0,0,0,0
#             nd_1, nd_5, nd_10, nd_15, nd_20, nd_50 = 0, 0, 0, 0, 0, 0
#             for i in tops:
#                 # ignore item in the training set
#                 if i in train_set:
#                     continue
#                 top_n_items += 1
#                 if i in test_set:
#                     if top_n_items <= 50:
#                         hits_50 += 1
#                         nd_50 += 1 / math.log2(top_n_items + 1)
#                     if top_n_items <= 20:
#                         hits_20 += 1
#                         nd_20 += 1 / math.log2(top_n_items + 1)
#                     if top_n_items <= 15:
#                         hits_15 += 1
#                         nd_15 += 1 / math.log2(top_n_items + 1)
#                     if top_n_items <= 10:
#                         hits_10 += 1
#                         nd_10 += 1 / math.log2(top_n_items + 1)
#                     if top_n_items <= 5:
#                         hits_5 += 1
#                         nd_5 += 1 / math.log2(top_n_items + 1)
#                     if top_n_items <= 1:
#                         hits_1 += 1
#                         nd_1 += 1 / math.log2(top_n_items + 1)
#                 if top_n_items == k[-1]:
#                     break
#             hrs_1.append(hits_1 / float(len(test_set)))
#             hrs_5.append(hits_5 / float(len(test_set)))
#             hrs_10.append(hits_10 / float(len(test_set)))
#             hrs_15.append(hits_15 / float(len(test_set)))
#             hrs_20.append(hits_20 / float(len(test_set)))
#             hrs_50.append(hits_50 / float(len(test_set)))
#             ndcgs_1.append(nd_1 / float(len(test_set)))
#             ndcgs_5.append(nd_5 / float(len(test_set)))
#             ndcgs_10.append(nd_10 / float(len(test_set)))
#             ndcgs_15.append(nd_15 / float(len(test_set)))
#             ndcgs_20.append(nd_20 / float(len(test_set)))
#             ndcgs_50.append(nd_50 / float(len(test_set)))
#         return hrs_1, hrs_5, hrs_10, hrs_15, hrs_20, hrs_50, ndcgs_1, ndcgs_5, ndcgs_10, ndcgs_15, ndcgs_20, ndcgs_50
