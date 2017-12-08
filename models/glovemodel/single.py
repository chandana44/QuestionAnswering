"""Example running MemN2N on a single bAbI task.
Download tasks from facebook.ai/babi """
from __future__ import absolute_import
from __future__ import print_function

from data_utils import load_task, vectorize_data, read_word_embeddings, create_small_embedding
from sklearn import cross_validation, metrics
from memn2n import MemN2N
from itertools import chain
from six.moves import range
from sets import Set

import os
import tensorflow as tf
import numpy as np

tf.flags.DEFINE_float("learning_rate", 0.01, "Learning rate for Adam Optimizer.")
tf.flags.DEFINE_float("epsilon", 1e-8, "Epsilon value for Adam Optimizer.")
tf.flags.DEFINE_float("regularization", 0.02, "Regularization.")
tf.flags.DEFINE_float("max_grad_norm", 40.0, "Clip gradients to this norm.")
tf.flags.DEFINE_integer("evaluation_interval", 10, "Evaluate and print results every x epochs")
tf.flags.DEFINE_integer("batch_size", 64, "Batch size for training.")
tf.flags.DEFINE_integer("hops", 3, "Number of hops in the Memory Network.")
tf.flags.DEFINE_integer("epochs", 200, "Number of epochs to train for.")
tf.flags.DEFINE_integer("embedding_size", 50, "Embedding size for embedding matrices.")
tf.flags.DEFINE_integer("memory_size", 50, "Maximum size of memory.")
tf.flags.DEFINE_integer("task_id", 1, "bAbI task id, 1 <= id <= 20")
tf.flags.DEFINE_integer("random_state", None, "Random state.")
tf.flags.DEFINE_string("data_dir", "babi/tasks_1-20_v1-2/en/", "Directory containing bAbI tasks")
#tf.flags.DEFINE_string("data_dir", "../../data/tasks_1-20_v1-2/en/", "Directory containing bAbI tasks")
FLAGS = tf.flags.FLAGS

def get_log_dir_name():
    lr = FLAGS.learning_rate
    eps = FLAGS.epsilon
    mgn = FLAGS.max_grad_norm
    hp = FLAGS.hops
    es = FLAGS.embedding_size
    ms = FLAGS.memory_size
    ti = FLAGS.task_id
    reg = FLAGS.regularization

    log_dir_name = "lr={0}_eps={1}_mgn={2}_hp={3}_es={4}_ms={5}_reg={6}".format(lr, eps, mgn, hp, es, ms, reg)
    return os.path.join('./logs', str(ti), log_dir_name)

print("Started Task:", FLAGS.task_id)

# task data
train, test = load_task(FLAGS.data_dir, FLAGS.task_id)
data = train + test

#building embeddings
word_vectors = read_word_embeddings("babi/glove.6B.50d.txt")
#word_vectors = read_word_embeddings("../../data/glove.6B.50d.txt")
word_vectors.vectors = np.float32(word_vectors.vectors)
vocab = sorted(reduce(lambda x, y: x | y, (set(list(chain.from_iterable(s)) + q + a) for s, q, a in data)))
word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
reverse_lookup = {v:k for (k,v) in word_idx.items()}
lookup_vocab = ['nil']

print(reverse_lookup)
print(word_idx)
for i in range(1, len(reverse_lookup)+1):
    lookup_vocab.append(reverse_lookup[i])

max_story_size = max(map(len, (s for s, _, _ in data)))
mean_story_size = int(np.mean(map(len, (s for s, _, _ in data))))
sentence_size = max(map(len, chain.from_iterable(s for s, _, _ in data)))
query_size = max(map(len, (q for _, q, _ in data)))
#memory_size = min(FLAGS.memory_size, max_story_size)
memory_size = max_story_size
vocab_size = len(word_idx) + 1 # +1 for nil word
sentence_size = max(query_size, sentence_size) # for the position

print("Longest sentence length", sentence_size)
print("Longest story length", max_story_size)
print("Average story length", mean_story_size)

# train/validation/test sets
small_embedding = create_small_embedding(data, word_vectors, sentence_size)
small_embedding.vectors = np.float32(small_embedding.vectors)
S, Q, A = vectorize_data(train, small_embedding, word_idx, sentence_size, memory_size)
trainS, valS, trainQ, valQ, trainA, valA = cross_validation.train_test_split(S, Q, A, test_size=.1, random_state=FLAGS.random_state)
testS, testQ, testA = vectorize_data(test, small_embedding, word_idx, sentence_size, memory_size)

print(small_embedding)
print(testS[0])

print("Training set shape", trainS.shape)

# params
n_train = trainS.shape[0]
n_test = testS.shape[0]
n_val = valS.shape[0]

print("Training Size", n_train)
print("Validation Size", n_val)
print("Testing Size", n_test)

train_labels = np.argmax(trainA, axis=1)
test_labels = np.argmax(testA, axis=1)
val_labels = np.argmax(valA, axis=1)

tf.set_random_seed(FLAGS.random_state)
batch_size = FLAGS.batch_size
optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate, epsilon=FLAGS.epsilon)

print(small_embedding.word_indexer)
print(word_vectors.vectors.shape)
print(word_vectors.word_indexer.get_index('sandra'))
print(word_vectors.word_indexer.get_index('UNK'))
print(word_vectors.get_embedding('sandra'))
print(word_vectors.get_embedding('UNK'))
print(small_embedding.get_embedding('sandra'))
print(small_embedding.vectors.shape)


batches = zip(range(0, n_train-batch_size, batch_size), range(batch_size, n_train, batch_size))
with tf.Session() as sess:
    print('Network')
    model = MemN2N(batch_size, vocab_size, sentence_size, memory_size, FLAGS.embedding_size, small_embedding, session=sess,
                   hops=FLAGS.hops, max_grad_norm=FLAGS.max_grad_norm, optimizer=optimizer, l2=FLAGS.regularization, nonlin=tf.nn.relu)
    print('Network')
    #writer = tf.summary.FileWriter(get_log_dir_name(), sess.graph)

    for t in range(1, FLAGS.epochs+1):
        #print('Shuffle')
        np.random.shuffle(batches)
        total_cost = 0.0
        for start in range(0, n_train, batch_size):
            #print('Prefit')
            end = start + batch_size
            s = trainS[start:end]
            q = trainQ[start:end]
            a = trainA[start:end]
            #print('Fit start')
            cost_t, cost_summary, cost_ema = model.batch_fit(s, q, a)
            #print('Fit end')
            total_cost += cost_t

            # writer.add_summary(cost_summary, t*n_train+start)
            #writer.add_summary(cost_ema, t*n_train+start)

        if t % FLAGS.evaluation_interval == 0:
            train_preds = []
            for start in range(0, n_train, batch_size):
                end = start + batch_size
                s = trainS[start:end]
                q = trainQ[start:end]
                pred = model.predict(s, q)
                train_preds += list(pred)

#             val_preds = model.predict(valS, valQ)
            train_acc = metrics.accuracy_score(np.array(train_preds), train_labels)
            total_cost_summary = tf.summary.scalar("epoch_loss", total_cost)
            tcs = sess.run(total_cost_summary)
            #writer.add_summary(tcs, t)
#             val_acc = metrics.accuracy_score(val_preds, val_labels)

            val_acc, val_acc_summary = model.get_val_acc_summary(valS, valQ, val_labels)
            #writer.add_summary(val_acc_summary, t)

            print('-----------------------')
            print('Epoch', t)
            print('Total Cost:', total_cost)
            print('Training Accuracy:', train_acc)
            print('Validation Accuracy:', val_acc)
            print('-----------------------')

    test_preds, probs_hops, pred_proba = model.predict_proba(testS, testQ)
    A, B, C, TA, TC, H, W, test_preds, probs_hops, pred_proba = model.predict_wlogs(testS, testQ)
    pred_word = [reverse_lookup[i] for i in test_preds]
    true_word = [reverse_lookup[i] for i in test_labels]
    print(word_idx)
    print(reverse_lookup)
    print(lookup_vocab)
    
    np.save('attention_task'+str(FLAGS.task_id), probs_hops)
    np.save('predprob_task'+str(FLAGS.task_id), pred_proba)
    np.save('pred_task'+str(FLAGS.task_id), pred_word)
    np.save('truth_task'+str(FLAGS.task_id), true_word)
    np.save('A_task'+str(FLAGS.task_id), A)
    np.save('B_task'+str(FLAGS.task_id), B)
    np.save('C_task'+str(FLAGS.task_id), C)
    np.save('TA_task'+str(FLAGS.task_id), TA)
    np.save('TC_task'+str(FLAGS.task_id), TC)
    np.save('H_task'+str(FLAGS.task_id), H)
    np.save('W_task'+str(FLAGS.task_id), W)
    np.save('lookupvocab_task'+str(FLAGS.task_id), lookup_vocab)

    test_acc = metrics.accuracy_score(test_preds, test_labels)
    print("Testing Accuracy:", test_acc)
