#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn import preprocessing
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import csv
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from xgboost.sklearn import XGBClassifier 
from sklearn.pipeline import Pipeline


# In[ ]:


#read the simulated dataset files to use it in train and test
errorfile1=open("/home/citc/data/e1.fastq")
errorfile2=open("/home/citc/data/e2.fastq")
errorfreefile1=open("/home/citc/data/eef1.fastq")
errorfreefile2=open("/home/citc/data/eef2.fastq")


# In[ ]:


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


# In[ ]:


errorreadsfile1=parse(errorfreefile1)
errorreadsfile2=parse(errorfreefile2)
erfreereadsfile1=parse(errorfile1)
erfreereadsfile2=parse(errorfile2)


# In[ ]:


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


# In[ ]:


# to compute the k-mer of all data using specific k-mer size
def kmers(read, k):
    count = []
    num_kmers = len(read) - k + 1
    for i in range(num_kmers):
       kmer = read[i:i+k]
       counts.append(kmer)
    return counts 


# In[ ]:


#compute the k-mer of training dataset using k-mer size 15(we used different k-mer size)
kmertrain=[]
for i in train1:
    x= kmers(i,15)
    kmertrain.append(x)
    
kmertrain1=[]
for i in kmertrain:
    o=  ' '.join(i) 
    kmertrain1.append(o) 


# In[ ]:


# compute k-mer of testing dataset using k-mer size 15
kmertest=[]
for i in test1:
    x= kmers(i,15)
    kmertest.append(x) 

kmertest1=[]
for i in kmertest:
    o=  ' '.join(i) 
    kmertest1.append(o)


# In[ ]:


# read the real dataset and parse the data
real=open("/home/citc/data/data")
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


# In[ ]:


#to parse the sam file 
def parsesam(sa):
    reads=[]
    while True:
        firstline=sa.readline()
        if(len(firstline)==0):
            break
        name=firstline[0:].rstrip()
        reads.append(name)
        
    return reads

sam=open("/home/citc/bwa-0.7.17/results.sam")
datafromsam=parsesam(sam)
datafromsam1=(datafromsam[2:])


# In[1]:


#samm contain information about each read (read ID,alignment score)
def samdata (file):    
    samdat=[]
    for st in file:
        stri=st.split('\t')
        samdat.append(stri)
    
    sam1=[]
    sam2=[]
    sam3=[]
#we take only some information from each sam read  
    for aa in samdat:
        x=aa[0]   #read ID
        ID=x1[:15]    
        MD=aa[12]  #MD
        AS=aa[-2]  #AS
        sam1.append(ID)
        sam2.append(MD)
        sam3.append(AS)
       
    sammdata=[]
    allsammdata=[]
    for i in range(len(sam1)):
        q=sam1[i]
        q1=sam2[i]
        q2=sam3[i]
        
        sammdata.append(q)
        sammdata.append(q1)
        sammdata.append(q2)
        allsammdata.append(sammdata) 
        ss=[] 
    return allsammdata

sss1=samdata(datafromsam1) #list cotain lists of information about sam file data for each read


# In[ ]:


realdataname=parsename(real)
realda=[]
for i in realdataname:
        x=i[:15] 
        realda.append(x)


# In[8]:


def writecsv (filename):   
    fields = ['Name','MD','AS'] 
      # writing to csv file 
    with open(filename, 'w') as csvfile:
      # creating a csv writer object
         csvwriter = csv.writer(csvfile) 
     # writing the fields 
         csvwriter.writerow(fields) 
     # writing the data rows 
         x=csvwriter.writerows(rows) 


# In[ ]:


tr_idf_model  = TfidfVectorizer()
# compute TF-IDF for train, test and real dataset
tf_idf_vectortrain = tr_idf_model.fit_transform(kmertrain1)
tf_idf_vetortest = tr_idf_model.transform(kmertest1)
tf_idf_vecctorreal = tr_idf_model.transform(realdata1)


# In[ ]:


# train our dataset using naive base classifier 
nb = MultinomialNB(alpha=0.1)
nb.fit(tf_idf_vectortrain,ytrain)

#to test our simulated dataset
y_pred = nb.predict(tf_idf_vectortest)

#test the performance of simulated test dataset
print(classification_report(ytest,y_pred))

#to test our real dataset
per1=nb.predict(tf_idf_vecctorreal)

#after test the real dataset we need to evaluate the performace so we used bwa to align the dataset to the refrence genome and take the result sam file that determind the alignment score(AS) to detrmind the quality of the alighnment of each read and then compute the performance metrics
#read the sam file resulted from the bwa aligner
        
listper1 = per1.tolist()        
d1=zip(realda,listper1)
d2=(dict(d1))  #dictioary cotain the key is reads ID and value is the predited value from the NB classifier      

#we create a csv file to show each read ID ,MD and AS to compute the performance metric 
rows=[]
for i,j in d2.items():
    if j == 1:
        x=i 
        for h in range(len(sss1)):
            if(x==sss1[h][0]):
                rows.append(sss1[h])
#this for reads that classified as error                 
filename = "f_naive_bayes.csv" 
# writing to csv file 
writecsv(filename)

    
rows=[]
for i,j in d2.items():
    if j == 0:
        x=i 
        for h in range(len(sss1)):
            if(x==sss1[h][0]): #check if the name of read in key of the dictionary = the name of read in the data from sam file then add in formation of all data in list 
                #print(ss1[h][1])
                rows.append(sss1[h])
                
#this for reads that classified as errorfree                  
filename = "t_naive_bayes.csv"
# writing to csv file 
writecsv(filename )


# In[ ]:


# train our dataset using support vector machine classifier 
model_SVC = LinearSVC()
model_SVC.fit(tf_idf_vectortrain,ytrain)
#to test our simulated dataset
y_pred = model_SVC.predict(tf_idf_vectortest)

#test the performance of simulated test dataset
print(classification_report(ytest,y_pred))

#to test our real dataset
per1=model_SVC.predict(tf_idf_vecctorreal)

#after test the real dataset we need to evaluate the performace so we used bwa to align the dataset to the refrence genome and take the result sam file that determind the alignment score(AS) to detrmind the quality of the alighnment of each read and then compute the performance metrics
#read the sam file resulted from the bwa aligner
        
listper1 = per1.tolist()        
d1=zip(realda,listper1)
d2=(dict(d1))  #dictioary cotain the key is reads ID and value is the predited value from the NB classifier      

#we create a csv file to show each read ID ,MD and AS to compute the performance metric 
rows=[]
for i,j in d2.items():
    if j == 1:
        x=i 
        for h in range(len(sss1)):
            if(x==sss1[h][0]):
                rows.append(sss1[h])
#this for reads that classified as error                 
filename = "f_support_vector_machine.csv" 
# writing to csv file 
writecsv(filename)

    
rows=[]
for i,j in d2.items():
    if j == 0:
        x=i 
        for h in range(len(sss1)):
            if(x==sss1[h][0]): #check if the name of read in key of the dictionary = the name of read in the data from sam file then add in formation of all data in list 
                #print(ss1[h][1])
                rows.append(sss1[h])
                
#this for reads that classified as errorfree                  
filename = "t_support_vector_machine.csv"
# writing to csv file 
writecsv(filename )


# In[ ]:


# train our dataset using random forest classifier 
classifier = RandomForestClassifier()
classifier.fit(tf_idf_vectortrain,ytrain)
#to test our  simulated dataset
y_pred = classifier.predict(tf_idf_vectortest)

#test the performance of simulated test dataset
print(classification_report(ytest,y_pred))

#to test our real dataset
per1=classifier.predict(tf_idf_vecctorreal)

listper1 = per1.tolist()        
d1=zip(realda,listper1)
d2=(dict(d1))  #dictioary cotain the key is reads ID and value is the predited value from the NB classifier      

#we create a csv file to show each read ID ,MD and AS to compute the performance metric 
rows=[]
for i,j in d2.items():
    if j == 1:
        x=i 
        for h in range(len(sss1)):
            if(x==sss1[h][0]):
                rows.append(sss1[h])
#this for reads that classified as error                 
filename = "f_random_forest.csv" 
# writing to csv file 
writecsv(filename)

    
rows=[]
for i,j in d2.items():
    if j == 0:
        x=i 
        for h in range(len(sss1)):
            if(x==sss1[h][0]): #check if the name of read in key of the dictionary = the name of read in the data from sam file then add in formation of all data in list 
                #print(ss1[h][1])
                rows.append(sss1[h])
                
#this for reads that classified as errorfree                  
filename = "t_random_forest.csv"
# writing to csv file 
writecsv(filename )


# In[ ]:


# train our dataset using logistic regration
classifier_tfidf = LogisticRegression()
classifier_tfidf.fit(tf_idf_vectortrain,ytrain)
#to test our simulated dataset
y_pred = classifier_tfidf.predict(tf_idf_vectortest)

#test the performance of simulated test dataset
print(classification_report(ytest,y_pred))

#to test our real dataset
per1=classifier_tfidf.predict(tf_idf_vecctorreal)

listper1 = per1.tolist()        
d1=zip(realda,listper1)
d2=(dict(d1))  #dictioary cotain the key is reads ID and value is the predited value from the NB classifier      

#we create a csv file to show each read ID ,MD and AS to compute the performance metric 
rows=[]
for i,j in d2.items():
    if j == 1:
        x=i 
        for h in range(len(sss1)):
            if(x==sss1[h][0]):
                rows.append(sss1[h])
#this for reads that classified as error                 
filename = "f_logistic regration.csv" 
# writing to csv file 
writecsv(filename)

    
rows=[]
for i,j in d2.items():
    if j == 0:
        x=i 
        for h in range(len(sss1)):
            if(x==sss1[h][0]): #check if the name of read in key of the dictionary = the name of read in the data from sam file then add in formation of all data in list 
                #print(ss1[h][1])
                rows.append(sss1[h])
                
#this for reads that classified as errorfree                  
filename = "t_logistic_regration.csv"
# writing to csv file 
writecsv(filename )


# In[ ]:


#train our dataset using xgboost
clf=XGBClassifier(max_depth=3, learning_rate=0.1, n_estimators=500, objective='binary:logistic', booster='gbtree')
classifier = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('clf', XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.1)),

    ])
classifier.fit(w, ytrain)

#to test our simulated dataset
y_pred =classifier.predict(r)

#test the performance of simulated test dataset
print(classification_report(ytest,y_pred))

#to test our real dataset
per1=classifier.predict(re)

listper1 = per1.tolist()        
d1=zip(realda,listper1)
d2=(dict(d1))  #dictioary cotain the key is reads ID and value is the predited value from the NB classifier      

#we create a csv file to show each read ID ,MD and AS to compute the performance metric 
rows=[]
for i,j in d2.items():
    if j == 1:
        x=i 
        for h in range(len(sss1)):
            if(x==sss1[h][0]):
                rows.append(sss1[h])
#this for reads that classified as error                 
filename = "f_XGB.csv" 
# writing to csv file 
writecsv(filename)

    
rows=[]
for i,j in d2.items():
    if j == 0:
        x=i 
        for h in range(len(sss1)):
            if(x==sss1[h][0]): #check if the name of read in key of the dictionary = the name of read in the data from sam file then add in formation of all data in list 
                #print(ss1[h][1])
                rows.append(sss1[h])
                
#this for reads that classified as errorfree                  
filename = "t_XGB.csv"
# writing to csv file 
writecsv(filename )

