from __future__ import division, print_function
from keras.layers import Dense, Merge, Dropout, Permute, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.models import Sequential

import os

import babi

BABI_DIR = "../data/tasks_1-20_v1-2/en"
EMBED_HIDDEN_SIZE = 64
LSTM_OUTPUT_SIZE = 32
BATCH_SIZE = 32
NBR_EPOCHS = 500

for TASK_NBR in range(1,21):
    print('************* Task ' + str(TASK_NBR) + ' ****************')
    train_file, test_file = babi.get_files_for_task(TASK_NBR, BABI_DIR)

    data_train = babi.get_stories(os.path.join(BABI_DIR, train_file))
    data_test = babi.get_stories(os.path.join(BABI_DIR, test_file))

    word2idx = babi.build_vocab([data_train, data_test])
    vocab_size = len(word2idx) + 1
    print("vocab_size=", vocab_size)

    story_maxlen, question_maxlen = babi.get_maxlens([data_train, data_test])
    print("story_maxlen=", story_maxlen, "question_maxlen=", question_maxlen)

    Xs_train, Xq_train, Y_train = babi.vectorize(data_train, word2idx,
                                                 story_maxlen, question_maxlen)
    Xs_test, Xq_test, Y_test = babi.vectorize(data_test, word2idx,
                                              story_maxlen, question_maxlen)
    print(Xs_train.shape, Xq_train.shape, Y_train.shape)
    print(Xs_test.shape, Xq_test.shape, Y_test.shape)

    # story encoder memory. Output dim: (None, story_maxlen, EMBED_HIDDEN_SIZE)
    story_encoder_m = Sequential()
    story_encoder_m.add(Embedding(input_dim=vocab_size,
                                  output_dim=EMBED_HIDDEN_SIZE,
                                  input_length=story_maxlen))
    story_encoder_m.add(Dropout(0.3))

    # question encoder. Output dim: (None, query_maxlen, EMBED_HIDDEN_SIZE)
    question_encoder = Sequential()
    question_encoder.add(Embedding(input_dim=vocab_size,
                                   output_dim=EMBED_HIDDEN_SIZE,
                                   input_length=question_maxlen))
    question_encoder.add(Dropout(0.3))

    # compute match between story and question.
    # Output dim: (None, story_maxlen, question_maxlen)
    match = Sequential()
    match.add(Merge([story_encoder_m, question_encoder],
                    mode="dot", dot_axes=[2, 2]))

    # encode story into vector space of question
    # output dim: (None, story_maxlen, query_maxlen)
    story_encoder_c = Sequential()
    story_encoder_c.add(Embedding(input_dim=vocab_size,
                                  output_dim=question_maxlen,
                                  input_length=story_maxlen))
    story_encoder_c.add(Dropout(0.3))

    # combine match and story vectors.
    # Output dim: (None, query_maxlen, story_maxlen)
    response = Sequential()
    response.add(Merge([match, story_encoder_c], mode="sum"))
    response.add(Permute((2, 1)))

    ## combine response and question vectors and do logistic regression
    answer = Sequential()
    answer.add(Merge([response, question_encoder], mode="concat", concat_axis=-1))
    answer.add(LSTM(LSTM_OUTPUT_SIZE))
    answer.add(Dropout(0.3))
    answer.add(Dense(vocab_size))
    answer.add(Activation("softmax"))

    answer.compile(optimizer="rmsprop", loss="categorical_crossentropy",
                   metrics=["accuracy"])

    answer.fit([Xs_train, Xq_train, Xs_train], Y_train,
               batch_size=BATCH_SIZE, nb_epoch=NBR_EPOCHS,
               validation_data=([Xs_test, Xq_test, Xs_test], Y_test))
