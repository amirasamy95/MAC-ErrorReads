from sklearn import preprocessing
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from xgboost.sklearn import XGBClassifier 
from sklearn.pipeline import Pipeline

#read the simulated dataset files to use it in train and test
errorfile1=open("/home/citc/data/erer1.fastq")
errorfile2=open("/home/citc/data/erer2.fastq")
errorfreefile1=open("/home/citc/data/eeff1.fastq")
errorfreefile2=open("/home/citc/data/eeff2.fastq")


# to pares fastq files reads
def parse(fastqfile):
    reads=[]
    while True:
        firstline=fastqfile.readline()
        if(len(firstline)==0):
            break
        name=firstline[1:].rstrip()
        seq=fastqfile.readline().rstrip()
        fastqfile.readline()
        qual=fastqfile.readline().rstrip()
        reads.append(seq)
    return reads

# to pares fastq files reads name 
def parsename(fastqfile1):
    readsname=[]
    while True:
        firstline=fastqfile1.readline()
        if(len(firstline)==0):
            break
        name=firstline[:].rstrip()
        seq=fastqfile1.readline().rstrip()
        fastqfile1.readline()
        qual=fastqfile1.readline().rstrip()
       
        readsname.append(name)
        
    return readsname


errorreadsfile1=parse(errorfreefile1)
errorreadsfile2=parse(errorfreefile2)
erfreereadsfile1=parse(errorfile1)
erfreereadsfile2=parse(errorfile2)


for i in errorreadsfile2:
    errorreadsfile1.append(i)
for j in erfreereadsfile2:
    erfreereadsfile1.append(j)  
    
# create two vectors to labeled the simulated data    
label1= np.zeros(len(errorreadsfile1), dtype = int) 
label2= np.ones(len(erfreereadsfile1), dtype = int)

#split the simulated data to train and test
train1=errorreadsfile1[:300000]
train2=erfreereadsfile1[:300000]
for x in train2:
    train1.append(x)
    
test1=errorreadsfile1[300000:]
test2=erfreereadsfile1[300000:]
for y in test2:
    test1.append(y)

#split the labeld data to train and test
yta=label1[:300000]
ytaa1=label2[:300000]
ytrain = np.concatenate((yta,ytaa1)) 

yta1=label1[300000:]
ytaa2=label2[300000:]   
ytest= np.concatenate((yta1,ytaa2))


# to compute the k-mer of all data using specific k-mer size
def kmers(read, k):
    count = []
    num_kmers = len(read) - k + 1
    for i in range(num_kmers):
       kmer = read[i:i+k]
       counts.append(kmer)
    return counts 


#compute the k-mer of training dataset using k-mer size 15
kmertrain=[]
for i in train1:
    x= kmers(i,15)
    kmertrain.append(x)
    
kmertrain1=[]
for i in kmertrain:
    o=  ' '.join(i) 
    kmertrain1.append(o) 


# compute k-mer of testing dataset using k-mer size 15
kmertest=[]
for i in test1:
    x= kmers(i,15)
    kmertest.append(x) 

kmertest1=[]
for i in kmertest:
    o=  ' '.join(i) 
    kmertest1.append(o)


# read the real dataset and parse the data
real=open("/home/citc/data/frag.fastq")
realdata=parse(real)

#compute the k-mer for real dataset using k-mer size 15
reals=[]
for i in realdata:
    x= kmers(i,15)
    reals.append(x)         

realdata1=[]
for i in reals:
    o=  ' '.join(i) 
    realdata1.append(o)


tr_idf_model  = TfidfVectorizer()
# compute TF-IDF for train, test and real dataset
tf_idf_vectortrain = tr_idf_model.fit_transform(kmertrain1)
tf_idf_vetortest = tr_idf_model.transform(kmertest1)
tf_idf_vecctorreal = tr_idf_model.transform(realdata1)


#after classification step in real dataset we need to put the errorfree reads in file and error reads in another file  
def parserealdata(fastqfile2):
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




# train our dataset using naive base classifier 
nb = MultinomialNB(alpha=0.1)
nb.fit(tf_idf_vectortrain,ytrain)

#to test our simulated dataset
y_pred = nb.predict(tf_idf_vectortest)

#test the performance of simulated test dataset
classification_report(ytest,y_pred)

#to test our real dataset
per1=nb.predict(tf_idf_vecctorreal)

#after classification step in real dataset we need to put the errorfree reads in file and error reads in another file to take the error free file and make the assembly process using velvet
realdataname=parsename(real)
listper1 = per1.tolist()
d1=zip(realdataname,listper1)
d2=(dict(d1))
tru=open("truereadsnb.txt","w")
fal=open("faleereadsnb.txt","w")
result=parserealdata(real)


# train our dataset using support vector machine classifier 
model_SVC = LinearSVC()
model_SVC.fit(tf_idf_vectortrain,ytrain)
#to test our simulated dataset
y_pred = model_SVC.predict(tf_idf_vectortest)

#test the performance of simulated test dataset
classification_report(ytest,y_pred)

#to test our real dataset
per1=model_SVC.predict(tf_idf_vecctorreal)

#after classificatio step in real dataset we need to put the errorfree reads in file and error reads in another file 
realdataname=parsename(real)
listper1 = per1.tolist()
d1=zip(realdataname,listper1)
d2=(dict(d1))
tru=open("truereadssvm.txt","w")
fal=open("faleereadssvm.txt","w")
real=open("/home/citc/data/frag.fastq")
result=parserealdata(real)


# train our dataset using random forest classifier 
classifier = RandomForestClassifier()
classifier.fit(tf_idf_vectortrain,ytrain)
#to test our  simulated dataset
y_pred = classifier.predict(tf_idf_vectortest)

#test the performance of simulated test dataset
classification_report(ytest,y_pred)

#to test our real dataset
per1=classifier.predict(tf_idf_vecctorreal)

#after classificatio step in real dataset we need to put the errorferr reads in file and error reads to file 
realdataname=parsename(real)
listper1 = per1.tolist()
d1=zip(realdataname,listper1)
d2=(dict(d1))
tru=open("truereadsrf.txt","w")
fal=open("faleereadsrf.txt","w")
result=parserealdata(real)

# train our dataset using logistic regration
classifier_tfidf = LogisticRegression()
classifier_tfidf.fit(tf_idf_vectortrain,ytrain)
#to test our simulated dataset
y_pred = classifier_tfidf.predict(tf_idf_vectortest)

#test the performance of simulated test dataset
classification_report(ytest,y_pred)

#to test our real dataset
per1=classifier_tfidf.predict(tf_idf_vecctorreal)

#after classificatio step in real dataset we need to put the errorfree reads in file and error reads in another file 
realdataname=parsename(real)
listper1 = per1.tolist()
d1=zip(realdataname,listper1)
d2=(dict(d1))
tru=open("truereadslr.txt","w")
fal=open("faleereadslr.txt","w")
result=parserealdata(real)


#train our dataset using xgboost
clf=XGBClassifier(max_depth=3, learning_rate=0.1, n_estimators=500, objective='binary:logistic', booster='gbtree')
classifier = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('clf', XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.1)),

    ])
classifier.fit(kmertrain1, ytrain)

#to test our simulated dataset
y_pred =classifier.predict(kmertest1)

#test the performance of simulated test dataset
print(classification_report(ytest,y_pred))

#to test our real dataset
per1=classifier.predict(realdata1)

#after classificatio step in real dataset we need to put the errorfree reads in file and error reads in another file 
realdataname=parsename(real)
listper1 = per1.tolist()
d1=zip(realdataname,listper1)
d2=(dict(d1))
tru=open("truereadsxgb.fastq","w")
fal=open("faleereadsxgb.fastq","w")
result=parserealdata(real)
