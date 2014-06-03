import random
import os
from sklearn.svm import LinearSVC, SVC
from sklearn.feature_extraction.text import TfidfVectorizer

chromosomes = []
kernels = ["linear","poly","rbf","sigmoid"]
attributes_used = ["kernel","gamma","degree","C"]

population = 4
mutation_rate = 0.2
iteration = 50

mins = {}
maks = {}
step = {}
mins["gamma"] = 0.0
maks["gamma"] = 10.0
step["gamma"] = 0.5
mins["degree"] = 1
maks["degree"] = 5
step["degree"] = 1
mins["C"] = 0.5
maks["C"] = 10.5
step["C"] = 0.5
mins["kernel"] = 0
maks["kernel"] = 3
step["kernel"] = 1

X_train = []
X_test = []
y_train = []
y_test = []

def print_chromosomes():
    for i in range(0,population):
        print chromosomes[i]
def extract_dataset():
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
    return (X_train, X_test, y_train, y_test)

def calculate_accuracy_chromosome(chromosome):
    #print 'Training with SVM'
    svc = SVC(C=chromosome["C"], kernel=chromosome["kernel"], degree=chromosome["degree"], gamma=chromosome["gamma"], coef0=0.0, 
              shrinking=True, probability=False, 
              tol=1e-3, cache_size=200, 
              class_weight=None, verbose=False, 
              max_iter=1000, random_state=None)
    #print svc_linear
    svc.fit(X_train,y_train)

    #print 'Predict test data'
    pred = svc.predict(X_test)

    true = 0
    false = 0
    for i in range(len(pred)):
        if pred[i]==y_test[i]:
            true=true+1
        else:
            false=false+1
    #print 'Accuracy: ',true*100.0/(true+false)
    print "calculate accuracy chromosome" + str(chromosome) + " " + str(true*100.0/(true+false))
    return (true*100.0/(true+false))

def calculate_accuracy():
    for i in range(0,len(chromosomes)):
        #print "calculating chromosome " + str(i)
        chromosomes[i]["accuracy"] = calculate_accuracy_chromosome(chromosomes[i]["chromosome"])

def generate_populasi_awal():
    for i in range(0,population):
        chromosomes.append({"accuracy": -1, "chromosome": generate_random_chromosome() })

def generate_random_chromosome():
    chromosome = {}
    for attribute in attributes_used:
        chromosome[attribute] = convert_attribute(attribute, random.randint(0,(maks[attribute]-mins[attribute])/step[attribute])*step[attribute] + mins[attribute])
    return chromosome

def convert_attribute(attr_name, attr_value):
    if (attr_name == "kernel"):
        return kernels[attr_value]
    else:
        return attr_value


def generate_crossover_matrix():
    crossover = {}
    for attribute in attributes_used:
        crossover[attribute] = random.randint(0,1)
    return crossover

def mating(crossover_matrix):
    for dad in range(0,population-1):
        for mom in range(dad+1,population):
            chromosome = { "accuracy":-1, "chromosome":{} }
            parent = [dad, mom]
            for attribute in crossover_matrix:
                chromosome["chromosome"][attribute] = chromosomes[parent[crossover_matrix[attribute]]]["chromosome"][attribute]
            chromosome["accuracy"] = calculate_accuracy_chromosome(chromosome["chromosome"])
            chromosomes.append(chromosome)

def mutasi_atribut(chromosome, attribute):
    chromosome[attribute] = convert_attribute(attribute, random.randint(0,(maks[attribute]-mins[attribute])/step[attribute])*step[attribute] + mins[attribute])

def mutasi():
    index_populasi = [i for i in range(0,len(chromosomes))]
    random.shuffle(index_populasi)
    new_chromosomes = []
    for i in range(0,int(mutation_rate*len(chromosomes))):
        new_chromosome = {"accuracy":chromosomes[index_populasi[i]]["accuracy"], "chromosome":{}}
        for attribut in attributes_used:
            new_chromosome["chromosome"][attribut] = chromosomes[index_populasi[i]]["chromosome"][attribut]
        index_atribut = random.randint(0,len(attributes_used)-1)
        #mutasi_atribut(chromosomes[index_populasi[i]]["chromosome"],attributes_used[index_atribut])
        #chromosomes[index_populasi[i]]["accuracy"] = calculate_accuracy_chromosome(chromosomes[index_populasi[i]]["chromosome"])
        mutasi_atribut(new_chromosome["chromosome"], attributes_used[index_atribut])
        new_chromosome["accuracy"] = calculate_accuracy_chromosome(new_chromosome["chromosome"])
        new_chromosomes.append(new_chromosome)
    chromosomes.extend(new_chromosomes)

X_train,X_test,y_train,y_test = extract_dataset()

random.seed()

generate_populasi_awal()
calculate_accuracy()

chromosomes.sort(reverse=True, key=lambda chromosome: chromosome["accuracy"])

for i in range(0,iteration):
    print "generasi " + str(i) + " size " + str(len(chromosomes))
    print_chromosomes()
    mating(generate_crossover_matrix())
    mutasi()
    #calculate_accuracy()
    chromosomes.sort(reverse=True)