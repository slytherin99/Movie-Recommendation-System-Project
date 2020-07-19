#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import warnings
    


# In[3]:


warnings.filterwarnings('ignore')


# In[4]:


column_names= ["user_id", "item_id", "rating", "timestamp"]
df= pd.read_csv('u.data', sep='\t', names= column_names)


# In[5]:


df.head()


# In[6]:


df.shape


# In[7]:


#number of unique users
df['user_id'].nunique()


# In[8]:


#number of unique movies
df['item_id'].nunique()


# In[9]:


movie_titles= pd.read_csv('u.item', sep= '\|', header= None)


# In[10]:


movie_titles= movie_titles[[0,1]]


# In[11]:


movie_titles.columns= ['item_id', 'title']


# In[12]:


movie_titles


# In[13]:


df= pd.merge(df,movie_titles, on="item_id")


# In[14]:


df.head()


# ## Exploratory Data Analysis 

# In[15]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[16]:


df.groupby('title').mean()['rating'].sort_values(ascending=False)


# In[17]:


df.groupby('title').count()['rating'].sort_values(ascending=False)


# In[18]:


ratings = pd.DataFrame(df.groupby('title').mean()['rating'])


# In[19]:


ratings['num_of_ratings']= pd.DataFrame(df.groupby('title').count()['rating'])


# In[20]:


ratings


# In[21]:


ratings.sort_values(by= 'rating', ascending= False)


# In[22]:


plt.figure(figsize=(10,6))
plt.hist(ratings['num_of_ratings'], bins=70)
plt.show()


# In[23]:


plt.hist(ratings['rating'], bins=70)
plt.show()


# In[24]:


sns.jointplot(x= 'rating', y= 'num_of_ratings', data=ratings, alpha=0.5)


# ## Creating Movie Recommendation

# In[25]:


df.head()


# In[26]:


moviematrix= df.pivot_table(index= 'user_id', columns='title', values= 'rating')


# In[27]:


moviematrix


# In[28]:


ratings.sort_values('num_of_ratings', ascending=False).head()


# In[29]:


starwars_user_rating = moviematrix['Star Wars (1977)']
starwars_user_rating.head()


# In[30]:


similar_to_starwars= moviematrix.corrwith(starwars_user_rating)


# In[31]:


similar_to_starwars
#this is a series


# In[32]:


corr_starwars= pd.DataFrame(similar_to_starwars, columns=['Correlation'])


# In[33]:


corr_starwars.dropna(inplace=True)


# In[34]:


corr_starwars.sort_values('Correlation', ascending= False).head(15)


# In[35]:


corr_starwars= corr_starwars.join(ratings['num_of_ratings'])
corr_starwars.head()


# In[36]:


corr_starwars[corr_starwars['num_of_ratings']>100].sort_values('Correlation', ascending= False)


# ### Predict Function

# In[37]:


def predict_movie(movie_name):
    movie_user_ratings= moviematrix[movie_name]
    similar_to_movie=moviematrix.corrwith(movie_user_ratings)
    
    corr_movie= pd.DataFrame(similar_to_movie, columns=['Correlation'])
    corr_movie.dropna(inplace=True)
    
    corr_movie= corr_movie.join(ratings['num_of_ratings'])
    predictions= corr_movie[corr_movie['num_of_ratings']>100].sort_values('Correlation', ascending= False)
    
    return predictions


# In[38]:


#Example
predictions= predict_movie('Titanic (1997)')


# In[39]:


predictions.head(10)
#top 10 recommendations


# In[40]:


predictions= predict_movie('Die Hard (1988)')


# In[41]:


predictions.head()
#top 5 recommendations


# In[42]:


#TREND: Franchise movies tend to show their sequels/prequels as top recommendations.


# In[ ]:




