import os
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer

class L1LinearSVC(LinearSVC):
    def fit(self, X, y):
        # The smaller C, the stronger the regularization.
        # The more regularization, the more sparsity.
        self.transformer_ = LinearSVC(penalty="l1", dual=False, tol=1e-3)
        X = self.transformer_.fit_transform(X, y)
        return LinearSVC.fit(self, X, y)

    def predict(self, X):
        X = self.transformer_.transform(X)
        return LinearSVC.predict(self, X)

'''print 'Adding stopwords'
path_stopwords = '../corpora/stopwords/'
file_stopwords = os.listdir(path_stopwords)
list_stopwords = []
for i in file_stopwords:
	f = open(path_stopwords+i,'r')
	content = f.read()
	content = content.split('\n')
	content.remove('')
	list_stopwords = list_stopwords + content'''

print 'Adding positive training data'	
path_train = '../corpora/tweets_train/'
list_train = []
list_train_target = []
# Positive
file_train = os.listdir(path_train+'pos/')
for i in file_train:
	f = open(path_train+'pos/'+i,'r')
	content = f.read()
	list_train.append(content)
	list_train_target.append(1)

# Negative
print 'Adding negative training data'
file_train = os.listdir(path_train+'neg/')
for i in file_train:
	f = open(path_train+'neg/'+i,'r')
	content = f.read()
	list_train.append(content)
	list_train_target.append(0)

print 'Adding positive testing data'	
path_test = '../corpora/tweets_test/'
list_test = []
list_test_target = []
# Positive
file_test = os.listdir(path_test+'pos/')
for i in file_test:
	f = open(path_test+'pos/'+i,'r')
	content = f.read()
	list_test.append(content)
	list_test_target.append(1)

# Negative
print 'Adding negative testing data'
file_test = os.listdir(path_test+'neg/')
for i in file_test:
	f = open(path_test+'neg/'+i,'r')
	content = f.read()
	list_test.append(content)
	list_test_target.append(0)

print 'Replace URL and mention'
for i in range(len(list_train)):
	temp = list_train[i]
	temp = temp.split(' ')
	for j in range(len(temp)):
		if temp[j].startswith('@'):
			temp[j]='||T||'
		elif (temp[j].lower()).startswith('http'):
			temp[j]='||U||'
	list_train[i] = ' '.join(temp)

for i in range(len(list_test)):
	temp = list_test[i]
	temp = temp.split(' ')
	for j in range(len(temp)):
		if temp[j].startswith('@'):
			temp[j]='||T||'
		elif (temp[j].lower()).startswith('http'):
			temp[j]='||U||'
	list_test[i] = ' '.join(temp)

print 'Extracting feature from training and testing data'
vectorizer = TfidfVectorizer(stop_words='english',token_pattern='([^\\s]+)')
X_train = vectorizer.fit_transform(list_train)
X_test = vectorizer.transform(list_test)
y_train = list_train_target
y_test = list_test_target

print 'Training with linear kernel SVM'
svc_linear = L1LinearSVC()
svc_linear.fit(X_train,y_train)

print 'Predict test data'
pred = svc_linear.predict(X_test)

'''print 'Input testing data'
true = 0
false = 0
path_test = '../corpora/tweets_test/'
# Positive
file_test = os.listdir(path_test+'pos/')
for i in file_test:
	f = open(path_test+'pos/'+i,'r')
	content = f.read()
	content = content.split('\n')[0]
	ans = vectorizer.transform([content]).toarray()
	ans = svc_linear.predict(ans)[0]
	if ans==1:
		true = true+1
	else:
		false = false+1

# Negative
file_test = os.listdir(path_test+'neg/')
for i in file_test:
	f = open(path_test+'neg/'+i,'r')
	content = f.read()
	content = content.split('\n')[0]
	ans = vectorizer.transform([content]).toarray()
	ans = svc_linear.predict(ans)[0]
	if ans==0:
		true = true+1
	else:
		false = false+1'''

true = 0
false = 0
for i in range(len(pred)):
	if pred[i]==y_test[i]:
		true=true+1
	else:
		false=false+1
	
print 'Accuracy: ',true*100.0/(true+false)
'''print 'Remove stopwords and replace url/mention on tweets training'	
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
svc_rbf.fit(unigram_train,list_train_target[::4])'''