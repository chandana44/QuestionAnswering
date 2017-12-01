import numpy as np 
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import collections

# attn_dir = 'attention_vis/attn/'
# pred_dir = 'attention_vis/pred/'
# truth_dir = 'attention_vis/truth/'
attn_dir = 'tmp/single_sharing_no_encode/'
pred_dir = 'tmp/single_sharing_no_encode/'
truth_dir = 'tmp/single_sharing_no_encode/'
data_dir = '../sujitpal/data/tasks_1-20_v1-2/en'
task_id = 2

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
            # num_questions = 0
        if '\t' in line: # question
            q, a, supporting = line.split('\t')
            supporting = map(int, supporting.split())
            supporting = [i-1 for i in supporting]
            q = tokenize(q)
            #a = tokenize(a)
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
                #substory = [x for x in story if x]
                #print substory
                substory = [x for x in story]

            data.append((substory, q, a, supporting))
            # num_questions += 1
            story.append('')
        else: # regular sentence
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

def heatmap_attention(mat_in, data):
	"""mat_in must be num_sentences x num_layers
		data is of the form (story, question, answer, supporting_fact)"""
	f = []
	ylabel = data[0]
	for sent in ylabel:
		if sent:
			s = ' '.join(sent)
			f.append(s)
	question = ' '.join(data[1]) + '?'
	answer = ' '.join(data[2])
	plt.title(question + ' ' + answer)
	ax = sns.heatmap(mat_in, vmin=0, vmax=1, yticklabels=f, annot=True)
	plt.tight_layout()
	plt.show()

if __name__ == '__main__':
	attn_mat = np.load(os.path.join(attn_dir,'attention_task'+str(task_id)+'.npy'))
	predicted_mat = np.load(os.path.join(pred_dir,'pred_task'+str(task_id)+'.npy'))
	truth_mat = np.load(os.path.join(truth_dir,'truth_task'+str(task_id)+'.npy'))
	idx = predicted_mat!=truth_mat
	idx = [c for c,v in enumerate(idx) if v == True]
	_, test_data = load_task(data_dir, task_id)
	misclassified = [c for (c,(i,j,k,l)) in enumerate(test_data) if c in idx]
	print len(misclassified)
	if len(misclassified) > 0:

		# Cause of error
		imagination, okay_memory, poor_memory, misinterpret = 0,0,0,0
		total_wrong = float(len(misclassified))
		for i in misclassified:
			attn_win_per_hop = np.argmax(attn_mat[:,i,:], axis=1)
			attn_win = attn_win_per_hop[-1]
			# print 'Length: '+ str(len(test_data[i][0]))
			# print  'attn: ' + str(attn_win)
			if attn_win >= len(test_data[i][0]):
				imagination += 1
			elif attn_win not in test_data[i][3]:
				poor_memory += 1
			elif truth_mat[i] not in test_data[i][0][attn_win]:
				okay_memory += 1
			elif attn_win in test_data[i][3]:
				misinterpret += 1
		# print(imagination+poor_memory+okay_memory+misinterpret)
		# print(total_wrong)
		print('Imagine: '+str((imagination/total_wrong)*100))
		print('PoorMemory: '+str((poor_memory/total_wrong)*100))
		print('OkayMemory: '+str((okay_memory/total_wrong)*100))
		print('Misinterpret: '+str((misinterpret/total_wrong)*100))

		# Error frequency by sentence length
		sl_wrong_freq = collections.defaultdict(int)
		for i in misclassified:
			f = []
			for s in test_data[i][0]:
				if s:
					f.append(s)
				#print t
			sl_wrong_freq[len(f)] += 1

		# sl_corr_freq = collections.defaultdict(int)
		# corr_classified = [c for (c,(i,j,k,l)) in enumerate(test_data) if c not in idx]
		# for i in corr_classified:
		# 	for s in test_data[i]:
		# 		t = [i for i in s if i]
		# 	sl_corr_freq[len(t)] += 1
		x_axis = []
		y_axis = []
		for key in sorted(sl_wrong_freq.iterkeys()):
			x_axis.append(key)
			y_axis.append(sl_wrong_freq[key])
		plt.plot(x_axis, y_axis)
		plt.show()

		# Visualize errors
		print_lim = 10
		visualize_wrong= [c for (c,(i,j,k,l)) in enumerate(test_data) if len(i)<=print_lim and c in idx]
		#print(visualize_wrong)
		print(test_data[visualize_wrong[3]])
		mat_in = attn_mat[:,visualize_wrong[3],:min(print_lim, len(test_data[visualize_wrong[3]][0]))].transpose()
		heatmap_attention(mat_in, test_data[visualize_wrong[3]])

	print_lim = 10
	visualize_correct = [c for (c,(i,j,k,l)) in enumerate(test_data) if len(i)<=print_lim and c not in idx]
	#print(visualize_correct)
	print(test_data[visualize_correct[4]])
	mat_in = attn_mat[:,visualize_correct[4],:min(print_lim, len(test_data[visualize_correct[4]][0]))].transpose()
	heatmap_attention(mat_in, test_data[visualize_correct[4]])