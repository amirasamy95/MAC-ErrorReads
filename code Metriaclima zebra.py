from sklearn import preprocessing
import numpy as np
from itertools import product
from scipy.optimize import linear_sum_assignment
import pandas as pd
import collections
from collections import OrderedDict
from matplotlib import pyplot as plt
from matplotlib import cm
import pylab
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,classification_report


f=open("/home/citc/data/fish/fir1.fastq")
h=open("/home/citc/data/fish/fif1.fastq")
f1=open("/home/citc/data/fish/fir2.fastq")
h1=open("/home/citc/data/fish/fif2.fastq")



def parse(km):
    reads=[]
    while True:
        firstline=km.readline()
        if(len(firstline)==0):
            break
        name=firstline[1:].rstrip()
        seq=km.readline().rstrip()
        km.readline()
        qual=km.readline().rstrip()
       
        reads.append(seq)
        
    return reads



q=parse(h)
q1=parse(h1)
w=parse(f)
w1=parse(f1)
print(len(q))
print(len(q1))
print(len(w))
print(len(w1))


for i in q1:
    q.append(i)
print(len(q))
for j in w1:
    w.append(j)  
    
    
ytain1= np.zeros(len(q), dtype = int) 
print(len(ytain1))

ytain2= np.ones(len(w), dtype = int)
print(len(ytain2))

train1=q[:350000]
train2=w[:350000]

test1=q[350000:]
test2=w[350000:]
print(len(test1))
print(len(test2))


for x in train2:
    train1.append(x)
print(len(train1))  
for y in test2:
    test1.append(y)
print(len(test1)) 

yta=ytain1[:350000]
print(len(yta))
yta1=ytain1[350000:]
print(len(yta1))

ytaa1=ytain2[:350000]
print(len(ytaa1))
ytaa2=ytain2[350000:]   
print(len(ytaa2))

ytrain = np.concatenate((yta,ytaa1))
print(len(ytrain))    
ytest= np.concatenate((yta1,ytaa2))
print(len(ytest)) 



def kmers(read, k):
    counts = []
    num_kmers = len(read) - k + 1
    for i in range(num_kmers):
       kmer = read[i:i+k]
       counts.append(kmer)
    return counts 



train=train1[:700000]
kme=[]
for i in train:
    x= kmers(i,11)
    kme.append(x)
    
print(len(kme))    

w=[]
for i in kme:
    o=  ' '.join(i) 
    w.append(o)


tes=[]
test=test1[:299944]
for i in test:
    x= kmers(i,11)
    tes.append(x) 
    
(print(len(tes)))   

r=[]
for i in tes:
    o=  ' '.join(i) 
    r.append(o)
print(len(r))  


real=open("/home/citc/data/fish/datafish2.fastq")
realdata=parse(real)
print(len(realdata))

reals=[]
for i in realdata:
    x= kmers(i,11)
    reals.append(x)         
print(len(reals))
print(len(reals[0]))

re=[]
for i in reals:
    o=  ' '.join(i) 
    re.append(o)

print(len(realdata[0]))





tr_idf_model  = TfidfVectorizer()
tf_idf_vector = tr_idf_model.fit_transform(w)
#print(tf_idf_vector)
print(type(tf_idf_vector), tf_idf_vector.shape)




tf_idf_vec = tr_idf_model.transform(re)
print(type(tf_idf_vec), tf_idf_vec.shape)



tf_idf_vectortest = tr_idf_model.transform(r)
print(type(tf_idf_vectortest), tf_idf_vectortest.shape)



from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB(alpha=0.1)
nb.fit(tf_idf_vector,ytrain)




#to save the model 
import joblib
joblib.dump(nb, "./naive_bayes11fish101.joblib")

import sklearn.metrics as metrics
y_pred = nb.predict(tf_idf_vectortest)
print('accuracy %s' % metrics.accuracy_score(y_pred, ytest))
      
import seaborn as sns
print(classification_report(ytest,y_pred))
from sklearn.metrics import confusion_matrix,classification_report
cnf_matrix = confusion_matrix(ytest,y_pred)
group_names = ['TN','FP','FN','TP']
group_counts = ["{0:0.0f}".format(value) for value in cnf_matrix.flatten()]
labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_names,group_counts)]
labels = np.asarray(labels).reshape(2,2)
sns.heatmap(cnf_matrix, annot=labels, fmt='', cmap='Blues')


from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
precision = precision_score(y_pred, ytest)
print(precision)

recall = recall_score(y_pred, ytest)
print(recall)


f1 = f1_score(y_pred, ytest)
print('F1 score: %f' % f1)

from sklearn.metrics import matthews_corrcoef
matthews_corrcoef(ytest,y_pred)


from sklearn import metrics
from sklearn.metrics import roc_curve, roc_auc_score
fpr1 , tpr1, thresholds1 = roc_curve(ytest,y_pred)
print('roc_auc_score for nb: ', roc_auc_score(ytest,y_pred))


plt.plot([0,1],[0,1], 'k--')
plt.plot(fpr1, tpr1, label= "NB")
plt.legend()
plt.xlabel("False Positive Rate")  
plt.ylabel("True Positive Rate")
plt.title('ROC curve')
#plt.legend(('NB','SVM','LR','XGB','RF'),loc='center right',borderpad=0.25)
plt.show()



import joblib
loaded_rf1 = joblib.load( './naive_bayes11fish101.joblib')


per1=loaded_rf1.predict(tf_idf_vec)
print(len(per1))



def parse(km):
    reads=[]
    while True:
        firstline=km.readline()
        if(len(firstline)==0):
            break
        name=firstline[:].rstrip()
        seq=km.readline().rstrip()
        km.readline()
        qual=km.readline().rstrip()
       
        reads.append(name)
        
    return reads


real=open("/home/citc/data/fish/datafish2.fastq")
realdata=parse(real)
print(len(realdata))
listper1 = per1.tolist()
print(realdata[99998:100000])


d1=zip(realdata,listper1)
d2=(dict(d1))


from string import Template
tru=open("corfishdata2.fastq","w")
fal=open("falfishdata2.fastq","w")
real=open("/home/citc/data/fish/datafish2.fastq")
def parse(km):
    reads=[]
    while True:
        firstline=km.readline()
        if(len(firstline)==0):
            break
        name=firstline[:].rstrip()
        seq=km.readline().rstrip()
        kk=km.readline().rstrip()
        qual=km.readline().rstrip()
        
        fastq_template = Template(f'$name\n$seq\n$kk\n$qual\n')
     
        for i,j in d2.items():
           
            if j==0 and i==name:
                tru.write(fastq_template.substitute(name=name,
                                         seq=seq,
                                         kk=kk,
                                         qual=qual))
                     
            elif j==1 and i==name : 
                fal.write(fastq_template.substitute(name=name,
                                         seq=seq,
                                         kk=kk,
                                         qual=qual))



xxx=parse(real)




