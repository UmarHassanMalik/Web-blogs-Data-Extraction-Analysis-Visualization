#!/usr/bin/env python
# coding: utf-8


# # IMPORTING ALL THE LIBRARIES WHICH WE APPLY IN PROJECT

# In[32]:


import requests
import bs4
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from pandas import DataFrame
import string
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.probability import FreqDist
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer


# # EXTRACTING LINK OF BLOGS IN food_url

# In[33]:


food_url = ['https://minimalistbaker.com/vegan-coffee-cake-gluten-free-oil-free/', 
       'https://minimalistbaker.com/easy-holiday-recipes/', 
       'https://minimalistbaker.com/easy-instant-pot-chili-vegan-oil-free/', 
       'https://minimalistbaker.com/instant-pot-wild-rice-fast-tender-no-soaking/',
       'https://minimalistbaker.com/blueberry-maple-protein-shake/', 
       'https://minimalistbaker.com/pumpkin-scones-with-maple-and-molasses-glaze/', 
       'https://minimalistbaker.com/vegan-everything-breakfast-cookies/',
       'https://minimalistbaker.com/baked-sweet-potato-chips/',
       'https://minimalistbaker.com/chewy-double-chocolate-peppermint-cookies/',
       'https://minimalistbaker.com/vegan-chocolate-coffee-ice-cream-sandwiches/',
       'https://minimalistbaker.com/vegan-berry-pop-tarts/',
       'https://minimalistbaker.com/date-sweetened-horchata/',
       'https://minimalistbaker.com/perfect-grilled-corn-salsa/',
       'https://minimalistbaker.com/rainbow-chard-hummus-wraps/',
       'https://minimalistbaker.com/lemon-blueberry-waffles-vegan-gluten-free/',
       'https://minimalistbaker.com/peach-oat-smoothie/',
       'https://minimalistbaker.com/gluten-free-strawberry-nectarine-crisp/',
       'https://minimalistbaker.com/15-minute-miso-soup-with-greens-and-tofu/',
       'https://minimalistbaker.com/homemade-pumpkin-pasta/',
       'https://minimalistbaker.com/pbj-graham-cracker-thumbprints/',
       'https://minimalistbaker.com/warm-roasted-butternut-squash-salad/',
       'https://minimalistbaker.com/creamy-eggplant-caramelized-onion-dip/',
       'https://minimalistbaker.com/vegan-pumpkin-spice-pancakes/',
       'https://minimalistbaker.com/tom-kha-gai-butternut-squash-soup/',
       'https://minimalistbaker.com/fudgy-vegan-double-chocolate-beet-muffins/',
       'https://minimalistbaker.com/the-best-damn-vegan-mashed-potatoes/',
       'https://minimalistbaker.com/coconut-cream-pie-french-toast/',
       'https://minimalistbaker.com/7-ingredient-vegan-key-lime-pies/',
       'https://minimalistbaker.com/vegan-chai-ice-cream/',
       'https://minimalistbaker.com/crispy-peanut-tofu-cauliflower-rice-stir-fry/'
      ]

food_url


# # TAKING CLASS FORM BLOGS

# In[34]:


def script(link):
    page=requests.get(link).text
    soup=BeautifulSoup(page,'lxml')
    text=[p.text for p in soup.find(class_='entry-header').find_all('h1')]
    text2=[p.text for p in soup.find(class_='wprm-recipe-summary wprm-block-text-normal').find_all('span')]
    return text,text2


# # DATA EXTRACTION

# In[35]:


result=[]
for i in food_url:
    a=script(i)
    title=a[0]
    content=a[1]
    link=i
    result.append((title,content,link))
    data=pd.DataFrame(result)
    

data


# # PRE-PROCESSING (DATA CLEANSING)

# In[36]:


result=[]
stemmer=WordNetLemmatizer()
for i in food_url:
    a=script(i)
    d=a[0]
    title= re.sub(r'\W', ' ', str(d))
    
    c=a[1]
    content= re.sub(r'\W', ' ', str(c))
    
    link=i
    
    result.append((title,content,link))
    data=pd.DataFrame(result)
data


# # RENAME COLUMN

# In[37]:


new_data=data.rename(columns={0:'Title',1:'Content',2:'URL'})
new_data


# # READ CSV FILE

# In[38]:


new_data.to_csv('tics.csv', index=False )
d=pd.read_csv('tics.csv')
d


# # IMPORT LIBRARY

# In[39]:


from textblob import TextBlob


# # SENTIMENTAL ANALYSIS

# In[40]:


TextBlob(d["Content"].iloc[0]).sentiment


# In[41]:


tfidfconverter= TfidfVectorizer(max_features=1500, min_df=5, max_df=0.7, ngram_range=(1,3), stop_words=stopwords.words('english'))
converted=tfidfconverter.fit_transform(d['Content'].values.astype('U'))
converted.toarray()


# In[42]:


tfidfconverter.get_feature_names()


# In[43]:


len(tfidfconverter.get_feature_names())


# In[44]:


d['Content'][:]


# In[45]:


fdist=FreqDist()


# In[46]:


for x in str(d['Content'][:]).split():
    fdist[x]+=1
len(fdist)


# In[47]:


fdist.most_common(30)


# In[48]:


d['Polarity']=d['Content'].apply(lambda x: TextBlob(x).sentiment[0])


# In[49]:


d['Subectivity']=d['Content'].apply(lambda x: TextBlob(x).sentiment[1])


# In[50]:


d['Length']=d['Content'].apply(lambda x: len(x.split()))


# In[51]:


d


# In[52]:


d.to_csv('tics.csv', index=False )
pd.read_csv('tics.csv')


# # IMPORT LIBRARY

# In[53]:


from matplotlib import pyplot as V


# # VISUALIZATION

# In[54]:


import seaborn as S
S.set(style="ticks", color_codes=True)


# In[58]:


S.histplot(data=d, x='Polarity',hue='Subectivity',multiple='stack')


# In[59]:


S.displot(data=d, x='Polarity',hue='Subectivity',color='variety')


# In[60]:


S.boxplot(x='Polarity', hue='Subectivity',data=d)
S.despine(offset=10, trim=True)


# In[61]:


H = S.PairGrid(d)
H.map_diag(S.histplot)
H.map_offdiag(S.scatterplot)


# In[ ]:




