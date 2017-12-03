"""End-To-End Memory Networks.

The implementation is based on http://arxiv.org/abs/1503.08895 [1]
"""
from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
import numpy as np
from six.moves import range

def position_encoding(sentence_size, embedding_size):
    """
    Position Encoding described in section 4.1 [1]
    """
    encoding = np.ones((embedding_size, sentence_size), dtype=np.float32)
    ls = sentence_size+1
    le = embedding_size+1
    for i in range(1, le):
        for j in range(1, ls):
            encoding[i-1, j-1] = (i - (le-1)/2) * (j - (ls-1)/2)
    encoding = 1 + 4 * encoding / embedding_size / sentence_size
    return np.transpose(encoding)

def no_encoding(sentence_size, embedding_size):
    """
    No Encoding
    """
    encoding = np.ones((embedding_size, sentence_size), dtype=np.float32)
    return np.transpose(encoding)

def zero_nil_slot(t, name=None):
    """
    Overwrites the nil_slot (first row) of the input Tensor with zeros.

    The nil_slot is a dummy slot and should not be trained and influence
    the training algorithm.
    """
    with tf.op_scope([t], name, "zero_nil_slot") as name:
        t = tf.convert_to_tensor(t, name="t")
        s = tf.shape(t)[1]
        z = tf.zeros(tf.stack([1, s]))
        return tf.concat([z, tf.slice(t, [1, 0], [-1, -1])], 0, name=name)

def add_gradient_noise(t, stddev=1e-3, name=None):
    """
    Adds gradient noise as described in http://arxiv.org/abs/1511.06807 [2].

    The input Tensor `t` should be a gradient.

    The output will be `t` + gaussian noise.

    0.001 was said to be a good fixed value for memory networks [2].
    """
    with tf.op_scope([t, stddev], name, "add_gradient_noise") as name:
        t = tf.convert_to_tensor(t, name="t")
        gn = tf.random_normal(tf.shape(t), stddev=stddev)
        return tf.add(t, gn, name=name)

class MemN2N(object):
    """End-To-End Memory Network."""
    def __init__(self, batch_size, vocab_size, sentence_size, memory_size, embedding_size, word_vectors,
        hops=3,
        max_grad_norm=40.0,
        nonlin=None,
        initializer=tf.random_normal_initializer(stddev=0.1),
        #encoding=position_encoding,
        encoding=no_encoding,
        session=tf.Session(),
        l2 = 0.02,
        lr = 0.01,
        epsilon = 1e-8,
        restoreLoc = None,
        name='MemN2N',
        optimizer=None):
        """Creates an End-To-End Memory Network

        Args:
            batch_size: The size of the batch.

            vocab_size: The size of the vocabulary (should include the nil word). The nil word
            one-hot encoding should be 0.

            sentence_size: The max size of a sentence in the data. All sentences should be padded
            to this length. If padding is required it should be done with nil one-hot encoding (0).

            memory_size: The max size of the memory. Since Tensorflow currently does not support jagged arrays
            all memories must be padded to this length. If padding is required, the extra memories should be
            empty memories; memories filled with the nil word ([0, 0, 0, ......, 0]).

            embedding_size: The size of the word embedding.

            hops: The number of hops. A hop consists of reading and addressing a memory slot.
            Defaults to `3`.

            max_grad_norm: Maximum L2 norm clipping value. Defaults to `40.0`.

            nonlin: Non-linearity. Defaults to `None`.

            initializer: Weight initializer. Defaults to `tf.random_normal_initializer(stddev=0.1)`.

            optimizer: Optimizer algorithm used for SGD. Defaults to `tf.train.AdamOptimizer(learning_rate=1e-2)`.

            encoding: A function returning a 2D Tensor (sentence_size, embedding_size). Defaults to `position_encoding`.

            session: Tensorflow Session the model is run with. Defaults to `tf.Session()`.

            name: Name of the End-To-End Memory Network. Defaults to `MemN2N`.
        """

        self._batch_size = batch_size
        self._vocab_size = vocab_size
        self._sentence_size = sentence_size
        self._memory_size = memory_size
        self._embedding_size = embedding_size
        self._hops = hops
        self._max_grad_norm = max_grad_norm
        self._nonlin = nonlin
        self._init = initializer
        self._opt = tf.train.AdamOptimizer(learning_rate=lr, epsilon=epsilon)
        self._name = name
        self._l2 = l2
        self.word_vectors = word_vectors

        self._build_inputs()
        self._build_vars()
        self._encoding = tf.constant(encoding(self._sentence_size, self._embedding_size), name="encoding")

        # cross entropy
        logits = self._inference(self._stories, self._queries) # (batch_size, vocab_size)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf.cast(self._answers, tf.float32), name="cross_entropy")
        cross_entropy_sum = tf.reduce_sum(cross_entropy, name="cross_entropy_sum")

        # loss op
        reg_loss = self._l2 * tf.add_n(tf.get_collection('reg_loss'))
        loss_op = cross_entropy_sum + reg_loss

        loss_op_summary = tf.summary.scalar("loss", loss_op)

        ema = tf.train.ExponentialMovingAverage(decay=0.99)
        self.update_loss_ema = ema.apply([loss_op])
        loss_ema = ema.average(loss_op)
        self.loss_ema_op = tf.summary.scalar('batch_loss_ema', loss_ema)

        # gradient pipeline
        grads_and_vars = self._opt.compute_gradients(loss_op)
        grads_and_vars = [(tf.clip_by_norm(g, self._max_grad_norm), v) for g,v in grads_and_vars]
        grads_and_vars = [(add_gradient_noise(g), v) for g,v in grads_and_vars]
        nil_grads_and_vars = []
        for g, v in grads_and_vars:
            nil_grads_and_vars.append((g, v))
        train_op = self._opt.apply_gradients(nil_grads_and_vars, name="train_op")

        # predict ops
        predict_op = tf.argmax(logits, 1, name="predict_op")
        predict_proba_op = tf.nn.softmax(logits, name="predict_proba_op")
        predict_log_proba_op = tf.log(predict_proba_op, name="predict_log_proba_op")

        # validation accuracy ops
        self.val_acc_op = self._get_val_acc(predict_op, self._val_answers)
        self.val_acc_summary = tf.summary.scalar("val_acc", self.val_acc_op)

        # assign ops
        self.loss_op = loss_op
        self.predict_op = predict_op
        self.predict_proba_op = predict_proba_op
        self.predict_log_proba_op = predict_log_proba_op
        self.train_op = train_op
        self.loss_op_summary = loss_op_summary

        # Summaries
        self.merged = tf.summary.merge_all()

        self._sess = session

        if restoreLoc is not None:
            saver = tf.train.Saver()
            saver.restore(self._sess, restoreLoc)
        else:
            init_op = tf.initialize_all_variables()
            self._sess.run(init_op)


    def _build_inputs(self):
        self._stories = tf.placeholder(tf.int32, [None, self._memory_size, self._sentence_size], name="stories")
        self._queries = tf.placeholder(tf.int32, [None, self._sentence_size], name="queries")
        self._answers = tf.placeholder(tf.int32, [None, self._vocab_size], name="answers")
        self._val_answers = tf.placeholder(tf.int32, [None], name="val_answers")

    def _build_vars(self):
        with tf.variable_scope(self._name + str(1)):
            nil_word_slot = tf.zeros([1, self._embedding_size])
            A = tf.concat([ nil_word_slot, self._init([self._vocab_size-1, self._embedding_size]) ], 0)
            B = tf.concat([ nil_word_slot, self._init([self._vocab_size-1, self._embedding_size]) ], 0)
            C = tf.concat([ nil_word_slot, self._init([self._vocab_size-1, self._embedding_size]) ], 0)
            #self.A = tf.Variable(A, name="A")
            #self.B = tf.Variable(B, name="B")
            #self.C = tf.Variable(C, name="C")

            self.TA = tf.Variable(self._init([self._memory_size, self._embedding_size]), name='TA')
            self.TC = tf.Variable(self._init([self._memory_size, self._embedding_size]), name='TC')

            self.H = tf.Variable(self._init([self._embedding_size, self._embedding_size]), name="H")
            self.W = tf.Variable(self._init([self._embedding_size, self._vocab_size]), name="W")
        #self._nil_vars = set([self.A.name, self.B.name, self.C.name])

        #tf.add_to_collection('reg_loss', tf.nn.l2_loss(self.A))
        #tf.add_to_collection('reg_loss', tf.nn.l2_loss(self.B))
        #tf.add_to_collection('reg_loss', tf.nn.l2_loss(self.C))
        tf.add_to_collection('reg_loss', tf.nn.l2_loss(self.TA))
        tf.add_to_collection('reg_loss', tf.nn.l2_loss(self.TC))
        tf.add_to_collection('reg_loss', tf.nn.l2_loss(self.W))
        tf.add_to_collection('reg_loss', tf.nn.l2_loss(self.H))

    def _inference(self, stories, queries):
        with tf.variable_scope(self._name + str(2)):
            q_emb = tf.nn.embedding_lookup(self.word_vectors.vectors, queries)
            # print('Query Embedding')
            # print(q_emb)
            u_0 = tf.reduce_sum(q_emb * self._encoding, 1)
            u = [u_0]
            self.probs_hops = []
            for hop_iter in range(self._hops):
                m_emb = tf.nn.embedding_lookup(self.word_vectors.vectors, stories)
                # print('Story Embedding')
                # print(m_emb)
                m = tf.reduce_sum(m_emb * self._encoding, 2) + self.TA
                # print('m')
                # print(m)
                # hack to get around no reduce_dot
                u_temp = tf.transpose(tf.expand_dims(u[-1], -1), [0, 2, 1])
                dotted = tf.reduce_sum(m * u_temp, 2)
                # print('dotted')
                # print(dotted)

                # Calculate probabilities
                probs = tf.nn.softmax(dotted)
                self.probs_hops.append(probs)

                probs_temp = tf.transpose(tf.expand_dims(probs, -1), [0, 2, 1])

                c_emb = tf.nn.embedding_lookup(self.word_vectors.vectors, stories)
                c = tf.reduce_sum(c_emb * self._encoding, 2) + self.TC

                c_temp = tf.transpose(c, [0, 2, 1])
                o_k = tf.reduce_sum(c_temp * probs_temp, 2)

                u_k = tf.matmul(u[-1], self.H) + o_k
                # nonlinearity
                if self._nonlin:
                    u_k = self._nonlin(u_k)

                u.append(u_k)

            return tf.matmul(u_k, self.W)

    def save_model(self, location):
        saver = tf.train.Saver()
        saver.save(self._sess, location, write_meta_graph=False)

    def _get_val_acc(self, pred_op, val_labels):
        corr_pred = tf.equal(tf.cast(pred_op, tf.int32), val_labels)
        acc_op = tf.reduce_mean(tf.cast(corr_pred, tf.float32))
        return acc_op

    def batch_fit(self, stories, queries, answers):
        """Runs the training algorithm over the passed batch

        Args:
            stories: Tensor (None, memory_size, sentence_size)
            queries: Tensor (None, sentence_size)
            answers: Tensor (None, vocab_size)

        Returns:
            loss: floating-point number, the loss computed for the batch
        """
        feed_dict = {self._stories: stories, self._queries: queries, self._answers: answers}
        loss, loss_op_summary, _, _, loss_ema = self._sess.run([self.loss_op, self.loss_op_summary, self.train_op, self.update_loss_ema, self.loss_ema_op], feed_dict=feed_dict)
        return loss, loss_op_summary, loss_ema

    def predict(self, stories, queries):
        """Predicts answers as one-hot encoding.

        Args:
            stories: Tensor (None, memory_size, sentence_size)
            queries: Tensor (None, sentence_size)

        Returns:
            answers: Tensor (None, vocab_size)
        """
        feed_dict = {self._stories: stories, self._queries: queries}
        return self._sess.run(self.predict_op, feed_dict=feed_dict)

    def predict_test(self, stories, queries):
        """Predicts answers as one-hot encoding.

        Args:
            stories: Tensor (None, memory_size, sentence_size)
            queries: Tensor (None, sentence_size)

        Returns:
            answers, probabilities per hop: Tensor (None, vocab_size), Tensor (None, hops, memory_size)
        """
        feed_dict = {self._stories: stories, self._queries: queries}
        ops = [self.predict_op]
        ops.extend(self.probs_hops)

        return self._sess.run(ops, feed_dict=feed_dict)

    def predict_proba(self, stories, queries):
        """Predicts probabilities of answers.

        Args:
            stories: Tensor (None, memory_size, sentence_size)
            queries: Tensor (None, sentence_size)

        Returns:
            answers: Tensor (None, vocab_size)
        """
        feed_dict = {self._stories: stories, self._queries: queries}
        return self._sess.run([self.predict_op, self.probs_hops, self.predict_proba_op], feed_dict=feed_dict)

    def predict_wlogs(self, stories, queries):
        """Predicts probabilities of answers.

        Args:
            stories: Tensor (None, memory_size, sentence_size)
            queries: Tensor (None, sentence_size)

        Returns:
            answers: Tensor (None, vocab_size)
        """
        feed_dict = {self._stories: stories, self._queries: queries}
        return self._sess.run([self.A, self.B, self.C, self.TA, self.TC, self.H, self.W, self.predict_op, self.probs_hops, self.predict_proba_op], feed_dict=feed_dict)

    def get_val_acc_summary(self, stories, queries, answers):
        feed_dict = {self._stories: stories, self._queries: queries, self._val_answers: answers}
        return self._sess.run([self.val_acc_op, self.val_acc_summary], feed_dict=feed_dict)

    def predict_log_proba(self, stories, queries):
        """Predicts log probabilities of answers.

        Args:
            stories: Tensor (None, memory_size, sentence_size)
            queries: Tensor (None, sentence_size)
        Returns:
            answers: Tensor (None, vocab_size)
        """
        feed_dict = {self._stories: stories, self._queries: queries}
        return self._sess.run(self.predict_log_proba_op, feed_dict=feed_dict)
