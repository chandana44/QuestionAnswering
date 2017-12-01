from memn2n import MemN2N
import tensorflow as tf
import numpy as np
import os


"""
Stores the configuration for loading the model
max_memory_size is the maximum size allowed during training
memory_size is the actual memory size used based on max story size
"""
config = {
    'batch': 32,
    'vocab_size': 175,
    'sentence_size': 11,
    'max_memory_size': 50,
    'memory_size': 50,
    'embedding_size': 50,
    'hops': 3,
    'max_grad_norm': 40.0,
    'regularization': 1e-5,
    'epsilon': 1e-8,
    'lr': 0.001
}

restore_location = 'server/model/weights/wts_pe'
print(restore_location)

sess = tf.Session()

model = MemN2N(config["batch"],
               config["vocab_size"],
               config["sentence_size"],
               config["memory_size"],
               config["embedding_size"],
               session=sess,
               hops=config["hops"],
               max_grad_norm=config["max_grad_norm"],
               l2=config["regularization"],
               lr=config["lr"],
               epsilon=config["epsilon"],
               nonlin=tf.nn.relu,
               restoreLoc=restore_location)


# Uncomment to see if the weights were loaded correctly
# print(sess.run(model.A))

def get_pred(testS, testQ):
    ps = model.predict_proba(testS, testQ)
    op = model.predict_test(testS, testQ)

    answer = op[0][0]
    answer_probability = float(np.max(ps))
    mem_probs = np.vstack(op[1:]).T[testS[0].any(axis=1)]
    return answer, answer_probability, mem_probs
