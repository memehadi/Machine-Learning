#!/usr/bin/env python
# coding: utf-8

# In[2]:


import nltk
sentence_data = "The First sentence is about Python. The Second: about Django. You can learn Python,Django and Data Ananlysis here. "
nltk_tokens = nltk.sent_tokenize(sentence_data)
print (nltk_tokens)


# In[7]:



from translate import Translator
translator= Translator(to_lang="German")
translation = translator.translate("Good Morning!")
print (translation)


# In[8]:


from nltk.tokenize import sent_tokenize,word_tokenize
from googletrans import Translator
#gs=goslate.Goslate()
file=open("KMeans.docx","r")
x=file.read()
x=x.lower()
tokens=sent_tokenize(x)
for i in  tokens:
    print (i)
translator=Translator()

for i in tokens:
    translation=translator.translate(i,dest='de')

print(translation.x,end=" ")


# In[ ]:





# In[ ]:




