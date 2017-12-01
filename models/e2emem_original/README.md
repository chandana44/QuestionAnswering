# MemN2N

Implementation of [End-To-End Memory Networks](http://arxiv.org/abs/1503.08895) with sklearn-like interface using Tensorflow. Tasks are from the [bAbl](http://arxiv.org/abs/1502.05698) dataset.

![MemN2N picture](https://www.dropbox.com/s/3rdwfxt80v45uqm/Screenshot%202015-11-19%2000.57.27.png?dl=1)

This is a fork of the original repository https://github.com/domluna/memn2n written by Dominique Luna.

Changes from the original implementation include -
1. L2 regularizations for weight matrices
2. Jaccard similarity for sentence selection to form memories
3. Per task early stopping during joint training
4. Web demo (see below)

## Our results

Task  |  Testing Accuracy
------|------------------
1     |  99.6
2     |  67.4
3     |  57.6
4     |  98.4
5     |  83.1
6     |  99.3
7     |  85.8
8     |  91.8
9     |  99.3
10    |  95.7
11    |  97.5
12    |  99.2
13    |  98.3
14    |  87.7
15    |  100
16    |  48
17    |  61.3
18    |  92.1
19    |  10.8
20    |  100

## Demo
We added a web demo allowing us to test the model and visualize the memory probabilities in each hop (episode). Below is an example that demonstrate it -

![Demo picture](https://github.com/priyank87/memn2n/blob/master/memn2n_web_demo.png)
