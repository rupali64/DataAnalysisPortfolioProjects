#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib
plt.style.use('ggplot')
from matplotlib.pyplot import figure

get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.rcParams['figure.figsize'] = (12,8)

pd.options.mode.chained_assignment = None



# Now we need to read in the data
df = pd.read_csv(r'H:\Data Analysis\Projects\movies.csv')


# In[7]:


df


# In[8]:


#Missing values
for col in df.columns:
    pct_missing = np.mean(df[col].isnull())
    print('{} - {}%'.format(col, round(pct_missing*100)))


# In[9]:


#DataTypes
print(df.dtypes)


# In[13]:


df.boxplot(column=['gross'])


# In[14]:


df.drop_duplicates()


# In[15]:


df.sort_values(by=['gross'], inplace=False, ascending=False)


# In[16]:


sns.regplot(x="gross", y="budget", data=df)


# In[17]:


sns.regplot(x="score", y="gross", data=df)


# In[18]:


df.corr(method ='pearson')


# In[19]:


df.corr(method ='kendall')


# In[20]:


df.corr(method ='spearman')


# In[21]:


correlation_matrix = df.corr()

sns.heatmap(correlation_matrix, annot = True)

plt.title("Correlation matrix for Numeric Features")

plt.xlabel("Movie features")

plt.ylabel("Movie features")

plt.show()


# In[22]:


df.apply(lambda x: x.factorize()[0]).corr(method='pearson')


# In[23]:


correlation_matrix = df.apply(lambda x: x.factorize()[0]).corr(method='pearson')

sns.heatmap(correlation_matrix, annot = True)

plt.title("Correlation matrix for Movies")

plt.xlabel("Movie features")

plt.ylabel("Movie features")

plt.show()


# In[24]:


correlation_mat = df.apply(lambda x: x.factorize()[0]).corr()

corr_pairs = correlation_mat.unstack()

print(corr_pairs)


# In[25]:


sorted_pairs = corr_pairs.sort_values(kind="quicksort")

print(sorted_pairs)


# In[26]:


strong_pairs = sorted_pairs[abs(sorted_pairs) > 0.5]

print(strong_pairs)


# In[27]:


CompanyGrossSum = df.groupby('company')[["gross"]].sum()

CompanyGrossSumSorted = CompanyGrossSum.sort_values('gross', ascending = False)[:15]

CompanyGrossSumSorted = CompanyGrossSumSorted['gross'].astype('int64') 

CompanyGrossSumSorted


# In[28]:


df['Year'] = df['released'].astype(str).str[:4]
df


# In[29]:


df.groupby(['company', 'year'])[["gross"]].sum()


# In[30]:


CompanyGrossSum = df.groupby(['company', 'year'])[["gross"]].sum()

CompanyGrossSumSorted = CompanyGrossSum.sort_values(['gross','company','year'], ascending = False)[:15]

CompanyGrossSumSorted = CompanyGrossSumSorted['gross'].astype('int64') 

CompanyGrossSumSorted


# In[31]:


CompanyGrossSum = df.groupby(['company'])[["gross"]].sum()

CompanyGrossSumSorted = CompanyGrossSum.sort_values(['gross','company'], ascending = False)[:15]

CompanyGrossSumSorted = CompanyGrossSumSorted['gross'].astype('int64') 

CompanyGrossSumSorted


# In[32]:


plt.scatter(x=df['budget'], y=df['gross'], alpha=0.5)
plt.title('Budget vs Gross Earnings')
plt.xlabel('Gross Earnings')
plt.ylabel('Budget for Film')
plt.show()


# In[33]:


df


# In[34]:


df_numerized = df


for col_name in df_numerized.columns:
    if(df_numerized[col_name].dtype == 'object'):
        df_numerized[col_name]= df_numerized[col_name].astype('category')
        df_numerized[col_name] = df_numerized[col_name].cat.codes
        
df_numerized


# In[35]:


df_numerized.corr(method='pearson')


# In[36]:


correlation_matrix = df_numerized.corr(method='pearson')

sns.heatmap(correlation_matrix, annot = True)

plt.title("Correlation matrix for Movies")

plt.xlabel("Movie features")

plt.ylabel("Movie features")

plt.show()


# In[39]:


df_numerized.corr()


# In[40]:


correlation_mat=df_numerized.corr()
corr_pairs= correlation_mat.unstack()
corr_pairs


# In[41]:


sorted_pairs = corr_pairs.sort_values()
sorted_pairs


# In[42]:


high_corr=sorted_pairs[(sorted_pairs)>0.5]


# In[43]:


high_corr


# In[ ]:


#Votes and budget have the highest correlation to gross earnings

#Company has low correlation

