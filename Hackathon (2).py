#!/usr/bin/env python
# coding: utf-8

# In[38]:


import pandas as pd
import numpy as np
from textblob import TextBlob as tb
from string import punctuation
from nltk.corpus import wordnet as wn
import operator
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import mpld3
from mpld3 import plugins


# In[70]:


# Data cleaning

resume = pd.read_csv('resume_dataset.csv')
print(resume['Category'].isnull())
print(resume['Resume'].isnull())
resume['Resume'] = resume['Resume'].str.replace('[^\w\s\n#@/:%.,_-]', '')
resume['Resume'] = resume['Resume'].str.replace('[â\*]', '')
resume['Resume'] = resume['Resume'].str.replace('[ª]', '')
# print(resume.head())


# In[71]:


#filter by category = Data Science
df = resume[resume['Category'] == 'Data Science']
df['Resume']


# In[94]:


# get the commonly used word in ntlk and create a list of unwanted words
UW=set(stopwords.words('english'))
UWs=['Experience','months','company','year','exprience','24','details','year','january','pvt','ltd']
#create list to store words
ngram_2=[]
#for every description in resume
for i in df['Resume']:
    #set a = spread all words in that review into 2 words by using ngram
    a= tb(i).ngrams(n=2)
    #for number between 0 to lenth of a
    for j in range(0,len(a)):
        #lower case the 1st word
        word=a[j][0].lower()
        #lower case the 2nd word
        word1=a[j][1].lower()
        #if the word not in commonly used word in ntlk and not in the list we created earlier
        if word not in UW and word1 not in UW and word not in UWs and word1 not in UWs:
            #lemmatize the words and append to the list 
            b=a[j].lemmatize()
            ngram_2.append(b)


# In[95]:


ngram_2


# In[96]:


#create list
wordlist = []
#for every value in test1
for i in ngram_2:
    #combine 2 words tgt
    wordlist.append(i[0]+ " "+i[1])

#another list
wordlist1=[]
#for number between 0 to length of wordlist
for i in range(0,len(wordlist)):
    #lower case
    a=wordlist[i].lower()
    #append to list
    wordlist1.append(a)

#create a dict
word_dict = {}
#for every value in wordlist1 
for i in wordlist1:
    #if it is in the dict alr
    if i in word_dict:
        #the count +1
        word_dict[i] +=1
    # if not
    else:
        #insert the word and let count =1
        word_dict[i] =1

#sort them in decscending order 
skilllist = sorted(word_dict.items(),key=operator.itemgetter(1), reverse = True)
#take the 1st 50 only
skilllist1=skilllist[:50]
skilllist1


# In[109]:


import spacy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.manifold import TSNE

from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer 

from imblearn.over_sampling import SMOTE
get_ipython().run_line_magic('matplotlib', 'inline')


# In[110]:


# get unique words and sentence count
def get_sentence_word_count(text_list):
    sent_count = 0
    word_count = 0
    vocab = {}
    for text in text_list:
        sentences=sent_tokenize(str(text).lower())
        sent_count = sent_count + len(sentences)
        for sentence in sentences:
            words=word_tokenize(sentence)
            for word in words:
                if(word in vocab.keys()):
                    vocab[word] = vocab[word] +1
                else:
                    vocab[word] =1 
    word_count = len(vocab.keys())
    return sent_count,word_count


# In[137]:


resume = resume[resume['Resume'].notna()]
sent_count,word_count= get_sentence_word_count(resume['Resume'].tolist())
print("Number of sentences in Resume column: "+ str(sent_count))
print("Number of unique words in Resume column: "+str(word_count))
data_categories  = resume.groupby(resume['Category'])
i = 1
print('=========== Categories =======================')
for catName,dataCategory in data_categories:
    print('Cat:'+str(i)+' '+catName + ' : '+ str(len(dataCategory)) )
    i = i+1
print('==================================')


# In[143]:


print('============ Categories Resume example======================')
i=1
for catName,dataCategory in data_categories:
    print('Cat:'+str(i)+' '+catName + ' : '+ str(len(dataCategory)) )
    dataList = dataCategory['Resume'].tolist()
    print('====================================================')
    print('Sample Resume:'+str(dataList[1]))
    print('====================================================')
    i = i+1

print('============ Categories Resume example======================')


# In[113]:


# text cleanning function

def clean_text(text ): 
    text = text.translate(str.maketrans('', '', string.punctuation))
    text1 = ''.join([w for w in text if not w.isdigit()]) 
    REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
    #BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
    
    text2 = text1.lower()
    text2 = REPLACE_BY_SPACE_RE.sub('', text2) # replace REPLACE_BY_SPACE_RE symbols by space in text
    #text2 = BAD_SYMBOLS_RE.sub('', text2)
    return text2

def lemmatize_text(text):
    wordlist=[]
    lemmatizer = WordNetLemmatizer() 
    sentences=sent_tokenize(text)
    
    intial_sentences= sentences[0:1]
    final_sentences = sentences[len(sentences)-2: len(sentences)-1]
    
    for sentence in intial_sentences:
        words=word_tokenize(sentence)
        for word in words:
            wordlist.append(lemmatizer.lemmatize(word))
    for sentence in final_sentences:
        words=word_tokenize(sentence)
        for word in words:
            wordlist.append(lemmatizer.lemmatize(word))       
    return ' '.join(wordlist)


# In[147]:


data = resume[['Category', 'Resume']]
data.shape


# In[149]:


print('Sample Resume 1:'+data.iloc[1]['Resume']+'\n')
print('Sample Resume 2:'+data.iloc[25]['Resume']+'\n')
print('Sample Resume 3:'+data.iloc[100]['Resume'])


# In[150]:


resume['Resume'] = resume['Resume'].apply(lemmatize_text)
resume['Resume'] = resume['Resume'].apply(clean_text)


# In[151]:


print('Sample Resume 1:'+data.iloc[1]['Resume']+'\n')
print('Sample Resume 2:'+data.iloc[25]['Resume']+'\n')
print('Sample Resume 3:'+data.iloc[100]['Resume'])


# In[152]:


vectorizer = TfidfVectorizer(analyzer='word', stop_words='english',ngram_range=(1,3), max_df=0.75, use_idf=True, smooth_idf=True, max_features=1000)
tfIdfMat  = vectorizer.fit_transform(data['Resume'].tolist() )
feature_names = sorted(vectorizer.get_feature_names())
print(feature_names)


# In[166]:


import gc
gc.collect()
tfIdfMatrix = tfIdfMat.todense()
labels = data['Category'].tolist()
tsne_results = TSNE(n_components=2,init='random',random_state=0, perplexity=40).fit_transform(tfIdfMatrix)
plt.figure(figsize=(16,10))
palette = sns.hls_palette(21, l=.6, s=.9)
sns.scatterplot(
    x=tsne_results[:,0], y=tsne_results[:,1],
    hue=labels,
    legend="full",
    alpha=0.3
)
plt.show()


# In[175]:



gc.collect()
pca = PCA(n_components=0.95)
tfIdfMat_reduced = pca.fit_transform(tfIdfMat.toarray())
labels = data['Category'].tolist()
category_list = data.Category.unique()
X_train, X_test, y_train, y_test = train_test_split(tfIdfMat_reduced, labels, test_size=0.2,random_state=1)


# In[176]:


print('Train_Set_Size:'+str(X_train.shape))
print('Test_Set_Size:'+str(X_test.shape))


# In[177]:


clf = LogisticRegression(penalty= 'elasticnet', solver= 'saga', l1_ratio=0.5, random_state=1).fit(X_train, y_train)
y_test_pred= clf.predict(X_test)


# In[178]:


labels = category_list
cm = confusion_matrix(y_test, y_test_pred, labels)


# In[179]:


fig = plt.figure(figsize=(20,20))
ax= fig.add_subplot(1,1,1)
sns.heatmap(cm, annot=True, cmap="Greens",ax = ax,fmt='g'); #annot=True to annotate cells

# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix'); 
ax.xaxis.set_ticklabels(labels); ax.yaxis.set_ticklabels(labels);
plt.setp(ax.get_yticklabels(), rotation=30, horizontalalignment='right')
plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')     
plt.show()


# In[180]:


print(classification_report(y_test,y_test_pred,labels=category_list))


# In[ ]:




