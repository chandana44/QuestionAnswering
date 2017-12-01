import re
import numpy as np
import pickle


with open('server/model/vocab_data.pickle', 'rb') as handle:
  vocab_data = pickle.load(handle)

decode_dict = {v:k for k,v in vocab_data['w_idx'].iteritems()}

def tokenize(sent):
    '''Return the tokens of a sentence including punctuation.
    >>> tokenize('Bob dropped the apple. Where is the apple?')
    ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
    '''
    return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]


def vectorize_data(data, word_idx, sentence_size, memory_size):
    """
    Vectorize stories and queries.

    If a sentence length < sentence_size, the sentence will be padded with 0's.

    If a story length < memory_size, the story will be padded with empty memories.
    Empty memories are 1-D arrays of length sentence_size filled with 0's.

    The answer array is returned as a one-hot encoding.
    """
    S = []
    Q = []
    A = []
    for story, query, answer in data:
        ss = []
        for i, sentence in enumerate(story, 1):
            ls = max(0, sentence_size - len(sentence))
            ss.append([word_idx[w] for w in sentence] + [0] * ls)

        # take only the most recent sentences that fit in memory
        ss = ss[::-1][:memory_size][::-1]

        # pad to memory_size
        lm = max(0, memory_size - len(ss))
        for _ in range(lm):
            ss.append([0] * sentence_size)

        lq = max(0, sentence_size - len(query))
        q = [word_idx[w] for w in query] + [0] * lq

        y = np.zeros(len(word_idx) + 1) # 0 is reserved for nil word
        for a in answer:
            y[word_idx[a]] = 1

        S.append(ss)
        Q.append(q)
        A.append(y)
    return np.array(S), np.array(Q), np.array(A)


def decode(index):
    return decode_dict.get(index, 'unknown')


def process_data(sentences, question):
    sent_t = [tokenize(s.lower()) for s in sentences]
    sent_t = [filter(lambda x: x != ".", s) for s in sent_t]

    q_t = tokenize(question.lower())
    if q_t[-1] == "?":
        q_t = q_t[:-1]

    data = [(sent_t, q_t, ['where'])]

    testS, testQ, testA = vectorize_data(data, vocab_data['w_idx'], vocab_data['sentence_size'], vocab_data['memory_size'])

    return testS, testQ, testA