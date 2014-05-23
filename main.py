import os
from sklearn import svm

print 'Adding stopwords'
path_stopwords = '../corpora/stopwords/'
file_stopwords = os.listdir(path_stopwords)
list_stopwords = []
for i in file_stopwords:
	f = open(path_stopwords+i,'r')
	content = f.read()
	content = content.split('\n')
	content.remove('')
	list_stopwords = list_stopwords + content

print 'Adding positive training data'	
path_train = '../corpora/tweets_train/'
list_train = []
list_train_target = []
# Positive
file_train = os.listdir(path_train+'pos/')
for i in file_train:
	f = open(path_train+'pos/'+i,'r')
	content = f.read()
	content = content.lower()
	list_train.append(content.split('\n')[0])
	list_train_target.append(1)

# Negative
print 'Adding negative training data'
file_train = os.listdir(path_train+'neg/')
for i in file_train:
	f = open(path_train+'neg/'+i,'r')
	content = f.read()
	content = content.lower()
	list_train.append(content.split('\n')[0])
	list_train_target.append(0)

print 'Remove stopwords and replace url/mention on tweets training'	
for i in range(len(list_train)):
	temp = list_train[i].split(' ')
	temp1 = [w for w in temp if not w in list_stopwords]
	j = 0
	z = len(temp1)
	while j<len(temp1):
		if temp1[j].startswith('@'):
			temp1.remove(temp1[j])
			temp1.append('||T||')
		elif temp1[j].startswith('http'):
			temp1.remove(temp1[j])
			temp1.append('||U||')
		else:
			j = j+1
	list_train[i] = temp1

print 'Determine unigram attribute'
attrib = []
for i in range(len(list_train)):
	for j in list_train[i]:
		if j in attrib:
			pass
		else:
			attrib.append(j)

print 'Generate unigram for tweets training'
unigram_train = []
i = 0
while i<len(list_train):
	list = [0 for k in range(len(attrib))]
	for j in list_train[i]:
		idx = attrib.index(j)
		list[idx] = 1
	unigram_train.append(list)
	i = i+4

print 'Train with SVM'
svc_rbf = svm.SVC(kernel='rbf',C=1)
svc_rbf.fit(unigram_train,list_train_target[::4])