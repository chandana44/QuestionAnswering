from __future__ import absolute_import

import os
import re
import numpy as np
from sets import Set

def load_task(data_dir, task_id, only_supporting=False):
    '''Load the nth task. There are 20 tasks in total.

    Returns a tuple containing the training and testing data for the task.
    '''
    assert task_id > 0 and task_id < 21

    files = os.listdir(data_dir)
    files = [os.path.join(data_dir, f) for f in files]
    s = 'qa{}_'.format(task_id)
    train_file = [f for f in files if s in f and 'train' in f][0]
    test_file = [f for f in files if s in f and 'test' in f][0]
    train_data = get_stories(train_file, only_supporting)
    test_data = get_stories(test_file, only_supporting)
    return train_data, test_data


def tokenize(sent):
    '''Return the tokens of a sentence including punctuation.
    >>> tokenize('Bob dropped the apple. Where is the apple?')
    ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
    '''
    return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]


def parse_stories(lines, only_supporting=False):
    '''Parse stories provided in the bAbI tasks format
    If only_supporting is true, only the sentences that support the answer are kept.
    '''
    data = []
    story = []
    for line in lines:
        line = str.lower(line)
        nid, line = line.split(' ', 1)
        nid = int(nid)
        if nid == 1:
            story = []
        if '\t' in line:  # question
            q, a, supporting = line.split('\t')
            q = tokenize(q)
            # a = tokenize(a)
            # answer is one vocab word even if it's actually multiple words
            a = [a]
            substory = None

            # remove question marks
            if q[-1] == "?":
                q = q[:-1]

            if only_supporting:
                # Only select the related substory
                supporting = map(int, supporting.split())
                substory = [story[i - 1] for i in supporting]
            else:
                # Provide all the substories
                substory = [x for x in story if x]

            data.append((substory, q, a))
            story.append('')
        else:  # regular sentence
            # remove periods
            sent = tokenize(line)
            if sent[-1] == ".":
                sent = sent[:-1]
            story.append(sent)
    return data


def get_stories(f, only_supporting=False):
    '''Given a file name, read the file, retrieve the stories, and then convert the sentences into a single story.
    If max_length is supplied, any stories longer than max_length tokens will be discarded.
    '''
    with open(f) as f:
        return parse_stories(f.readlines(), only_supporting=only_supporting)

def create_small_embedding(data, word_vectors, sentence_size):
    """
    Create relativized vectors from glove embedding
    """
    small_indexer = Indexer()
    small_vectors = []
    vocab_set = Set()

    for story, query, answer in data:
        # Get query vocab
        query_set = {w for w in query}
        vocab_set = vocab_set.union(query_set)
        ss = []

        # Get sentence vocab
        for i, sentence in enumerate(story, 1):
            ls = max(0, sentence_size - len(sentence))
            ss.append([word_vectors.word_indexer.get_index(w, False) for w in sentence] + [0] * ls)
            sentence_set = {w for w in sentence}
        vocab_set = vocab_set.union(sentence_set)

        # Get answer voca
        ans_set = {a for a in answer}
        vocab_set = vocab_set.union(ans_set)


    print vocab_set
    print len(vocab_set)
    small_indexer.get_index("UNK")
    small_vectors.append(word_vectors.get_embedding("UNK"))
    for w in vocab_set:
        print w
        small_indexer.get_index(w)
        small_vectors.append(word_vectors.get_embedding(w))

    # small_indexer.get_index("UNK")
    # small_vectors.append(np.zeros(vectors[0].shape[0]))

    small_embedding = WordEmbeddings(small_indexer, np.array(small_vectors))

    return small_embedding

def vectorize_data(data, word_vectors, word_idx, sentence_size, memory_size):
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
    # small_indexer = Indexer()
    # small_vectors = []
    # vocab_set = Set()

    for story, query, answer in data:
        lq = max(0, sentence_size - len(query))
        q = [word_vectors.word_indexer.get_index(w, False) for w in query] + [0] * lq
        # query_set = {w for w in query}
        # vocab_set = vocab_set.union(query_set)
        ss = []
        for i, sentence in enumerate(story, 1):
            ls = max(0, sentence_size - len(sentence))
            ss.append([word_vectors.word_indexer.get_index(w, False) for w in sentence] + [0] * ls)
        #     sentence_set = {w for w in sentence}
        # vocab_set = vocab_set.union(sentence_set)

        if len(ss) > memory_size:
            # Use Jaccard similarity to determine the most relevant sentences
            q_words = set(q)
            least_like_q = sorted(ss, cmp=lambda x, y: jaccard(set(x), q_words) < jaccard(set(y), q_words))[
                           :len(ss) - memory_size]
            for sent in least_like_q:
                # Remove the first occurrence of sent. A list comprehension as in [sent for sent in ss if sent not in least_like_q]
                # should not be used, as it would remove multiple occurrences of the same sentence, some of which might actually make the cutoff.
                ss.remove(sent)
        else:
            # pad to memory_size
            lm = max(0, memory_size - len(ss))
            for _ in range(lm):
                ss.append([0] * sentence_size)

        y = np.zeros(len(word_idx) + 1)  # 0 is reserved for nil word
        for a in answer:
            y[word_idx[a]] = 1
        # ans_set = {a for a in answer}
        # vocab_set = vocab_set.union(ans_set)

        S.append(ss)
        Q.append(q)
        A.append(y)

    # print vocab_set
    # print len(vocab_set)
    # for w in vocab_set:
    #     print w
    #     small_indexer.get_index(w)
    #     small_vectors.append(word_vectors.get_embedding(w))

    # small_indexer.get_index("UNK")
    # small_vectors.append(np.zeros(vectors[0].shape[0]))

    # small_embedding = WordEmbeddings(small_indexer, np.array(small_vectors))

    return np.array(S), np.array(Q), np.array(A)


def jaccard(a, b):
    '''
    Assumes that a and b are sets so that calling code only has to cast the question to set once.
    '''
    return len(a.intersection(b)) / float(len(a.union(b)))
    set(a).intersection(set(b))


# Loads the given embeddings (ASCII-formatted) into a WordEmbeddings object. Augments this with an UNK embedding
# that is the 0 vector. Reads in all embeddings with no filtering -- you should only use this for relativized
# word embedding files.
def read_word_embeddings(embeddings_file):
    f = open(embeddings_file)
    word_indexer = Indexer()
    vectors = []
    for line in f:
        if line.strip() != "":
            space_idx = line.find(' ')
            word = line[:space_idx]
            numbers = line[space_idx + 1:]
            float_numbers = [float(number_str) for number_str in numbers.split()]
            # print repr(float_numbers)
            vector = np.array(float_numbers)
            word_indexer.get_index(word)
            vectors.append(vector)
            # print repr(word) + " : " + repr(vector)
    f.close()
    print "Read in " + repr(len(word_indexer)) + " vectors of size " + repr(vectors[0].shape[0])
    # Add an UNK token at the end
    word_indexer.get_index("UNK")
    vectors.append(np.zeros(vectors[0].shape[0]))
    # Turn vectors into a 2-D numpy array
    return WordEmbeddings(word_indexer, np.array(vectors))


# Wraps an Indexer and a list of 1-D numpy arrays where each position in the list is the vector for the corresponding
# word in the indexer. The 0 vector is returned if an unknown word is queried.
class WordEmbeddings:
    def __init__(self, word_indexer, vectors):
        self.word_indexer = word_indexer
        self.vectors = vectors

    def get_embedding(self, word):
        word_idx = self.word_indexer.get_index(word, False)
        if word_idx != -1:
            return self.vectors[word_idx]
        else:
            return self.vectors[self.word_indexer.get_index("UNK")]


# Bijection between objects and integers starting at 0. Useful for mapping
# labels, features, etc. into coordinates of a vector space.
class Indexer(object):
    def __init__(self):
        self.objs_to_ints = {}
        self.ints_to_objs = {}

    def __repr__(self):
        return str([str(self.get_object(i)) for i in xrange(0, len(self))])

    def __len__(self):
        return len(self.objs_to_ints)

    def get_object(self, index):
        if (index not in self.ints_to_objs):
            return None
        else:
            #return self.objs_to_ints["UNK"]
            return self.ints_to_objs[index]

    def contains(self, object):
        return self.index_of(object) != -1

    # Returns -1 if the object isn't present, index otherwise
    def index_of(self, object):
        if (object not in self.objs_to_ints):
            return self.objs_to_ints["UNK"]
        else:
            return self.objs_to_ints[object]

    # Adds the object to the index if it isn't present, always returns a nonnegative index
    def get_index(self, object, add=True):
        if not add:
            return self.index_of(object)
        if (object not in self.objs_to_ints):
            new_idx = len(self.objs_to_ints)
            self.objs_to_ints[object] = new_idx
            self.ints_to_objs[new_idx] = object
        return self.objs_to_ints[object]


# Reads sentiment examples in the format [0 or 1]<TAB>[raw sentence]; tokenizes and indexes the sentence according
# to the vocabulary in indexer. If add_to_indexer is False, replaces unseen words with UNK, otherwise grows the
# indexer. word_counter optionally keeps a tally of how many times each word is seen (mostly for logging purposes).
def read_and_index_sentiment_examples(infile, indexer, add_to_indexer=False, word_counter=None):
    f = open(infile)
    exs = []
    for line in f:
        if len(line.strip()) > 0:
            fields = line.split("\t")
            label = int(fields[0])
            sent = fields[1]
            tokenized_cleaned_sent = filter(lambda x: x != '', clean_str(sent).rstrip().split(" "))
            if word_counter is not None:
                for word in tokenized_cleaned_sent:
                    word_counter.increment_count(word, 1.0)
            indexed_sent = [
                indexer.get_index(word) if indexer.contains(word) or add_to_indexer else indexer.get_index("UNK")
                for word in tokenized_cleaned_sent]
            exs.append(SentimentExample(indexed_sent, label))
    f.close()
    return exs
