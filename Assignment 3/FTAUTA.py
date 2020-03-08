#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
import streamlit as st

from sklearn.utils import resample
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import keras

plt.style.use('ggplot')



def main():
   
    readme_text = st.markdown(get_file_content_as_string("instructions.md"))

    
    for filename in EXTERNAL_DEPENDENCIES.keys():
        download_file(filename)

    
data_file = 'criteo_attribution_dataset.tsv.gz'
df0 = pd.read_csv(data_file, sep='\t', compression='gzip')
df0.head()


st.sidebar.title("What to do") 
add_selectbox = st.sidebar.selectbox( 'Which model do you want to analyze?', 
                ('Show instructions', 'Show the source code', 'Run the app'))

if add_selectbox == "Show instructions" :
    st.sidebar.success('To continue select "Run the app".')
elif add_selectbox == "Show the source code":
        readme_text.empty()
        st.code(get_file_content_as_string("tests.py"))
elif add_selectbox == "Run the app":
        readme_text.empty()
       

        run_the_app()

column = st.selectbox("What column do you want to display?",df0.columns)


n_campaigns = 2


# In[7]:


def add_derived_columns(df):
    df_ext = df.copy()
    df_ext['jid'] = df_ext['uid'].map(str) + '_' + df_ext['conversion_id'].map(str)
    
    min_max_scaler = MinMaxScaler()
    for cname in ('timestamp', 'time_since_last_click'):
        x = df_ext[cname].values.reshape(-1, 1) 
        df_ext[cname + '_norm'] = min_max_scaler.fit_transform(x)
    
    return df_ext
df1 = add_derived_columns(df0)

if st.checkbox('Concatenated dataframe(UID + Conversion ID)'):
    st.write(df1)
    st.dataframe(df1.style.highlight_max(axis=0))


# In[8]:


def sample_campaigns(df, n_campaigns):    
    campaigns = np.random.choice( df['campaign'].unique(), n_campaigns, replace = False )
    return df[ df['campaign'].isin(campaigns) ]
df2 = sample_campaigns(df1, n_campaigns)


# In[9]:


def filter_journeys_by_length(df, min_touchpoints):
    if min_touchpoints <= 1:
        return df
    else:
        grouped = df.groupby(['jid'])['uid'].count().reset_index(name="count")
        return df[df['jid'].isin( grouped[grouped['count'] >= min_touchpoints]['jid'].values )]
df3 = filter_journeys_by_length(df2, 2)


# In[10]:


def balance_conversions(df):
    df_minority = df[df.conversion == 1]
    df_majority = df[df.conversion == 0]
    
    df_majority_jids = np.array_split(df_majority['jid'].unique(), 100 * df_majority.shape[0]/df_minority.shape[0] )
    
    df_majority_sampled = pd.DataFrame(data=None, columns=df.columns)
    for jid_chunk in df_majority_jids:
        df_majority_sampled = pd.concat([df_majority_sampled, df_majority[df_majority.jid.isin(jid_chunk)]])
        if df_majority_sampled.shape[0] > df_minority.shape[0]:
            break
    
    return pd.concat([df_majority_sampled, df_minority]).sample(frac=1).reset_index(drop=True)
df4 = balance_conversions(df3)


# In[11]:


def map_one_hot(df, column_names, result_column_name):
    mapper = {} 
    for i, col_name in enumerate(column_names):
        for val in df[col_name].unique():
            mapper[str(val) + str(i)] = len(mapper)
         
    df_ext = df.copy()
    
    def one_hot(values):
        v = np.zeros( len(mapper) )
        for i, val in enumerate(values): 
            v[ mapper[str(val) + str(i)] ] = 1
        return v    
    
    df_ext[result_column_name] = df_ext[column_names].values.tolist()
    df_ext[result_column_name] = df_ext[result_column_name].map(one_hot)
    
    return df_ext
df5 = map_one_hot(df4, ['cat1', 'cat2', 'cat3', 'cat4', 'cat5', 'cat6', 'cat8'], 'cats')
df6 = map_one_hot(df5, ['campaign'], 'campaigns').sort_values(by=['timestamp_norm'])


# In[12]:


print(df6.shape[0])

print([df6[df6.conversion == 0].shape[0], df6[df6.conversion == 1].shape[0]])


# In[7]:


#df0.head()


# # First Touch Attribution

# In[13]:

st.title('First Touch Attribution')
st.text('Streamlit showing First Touch Attribution')

def first_touch_attribution(df):
    
    def count_by_campaign(df):
        counters = np.zeros(n_campaigns)
        for campaign_one_hot in df['campaigns'].values:
            campaign_id = np.argmax(campaign_one_hot)
            counters[campaign_id] = counters[campaign_id] + 1
        return counters
        
    campaign_impressions = count_by_campaign(df)
    
    df_converted = df[df['conversion'] == 1]
    idx = df_converted.groupby(['jid'])['timestamp_norm'].transform(min) == df_converted['timestamp_norm']

    campaign_conversions = count_by_campaign(df_converted[idx])
        
    return (campaign_conversions) / campaign_impressions
    
fta = first_touch_attribution(df6)


# In[14]:


fta


# In[22]:


campaign_idx = range(0, 2)

fig = plt.figure(figsize=(15,4))
ax = fig.add_subplot(111)
plt.bar( range(len(fta[campaign_idx])), fta[campaign_idx], color = 'purple',label='FTA' )
plt.xlabel('Campaign ID')
plt.ylabel('Return per impression')
plt.legend(loc='upper left')
plt.show()
st.pyplot()

#uta

st.title('U Touch Attribution')
st.text('Streamlit showing U Touch Attribution')

def U_touch_attribution(df):
    def count_by_campaign(df):
        counters = np.zeros(n_campaigns)
        for campaign_one_hot in df['campaigns'].values:
            campaign_id = np.argmax(campaign_one_hot)
            counters[campaign_id] = counters[campaign_id] + 1
        return counters
    campaign_impressions = count_by_campaign(df)
    
    
    df_converted = df[df['conversion'] == 1]
    temp_jid=0
    campaign_conversions_nrml=0
    campaign_conversions_first=0
    campaign_conversions_last=0
    campaign_conversions_intermediate=0
    for jid in df_converted.jid.unique():
        if jid != temp_jid:
            temp_jid = jid
            print(temp_jid)
            if df_converted[df_converted['jid'] == temp_jid]['timestamp_norm'].max() == df_converted[df_converted['jid'] == temp_jid]['timestamp_norm'].min():
                idx_nrml = df_converted.groupby(['jid'])['timestamp_norm'].transform(min) == df_converted['timestamp_norm']
                campaign_conversions_nrml = count_by_campaign(df_converted[idx_nrml])
            if df_converted[df_converted['jid'] == temp_jid]['timestamp_norm'].max() != df_converted[df_converted['jid'] == temp_jid]['timestamp_norm'].min():
                idx_min = df_converted.groupby(['jid'])['timestamp_norm'].transform(min) == df_converted['timestamp_norm']
                idx_max = df_converted.groupby(['jid'])['timestamp_norm'].transform(max) == df_converted['timestamp_norm']
                campaign_conversions_first = count_by_campaign(df_converted[idx_min])
                campaign_conversions_last = count_by_campaign(df_converted[idx_max])
                campaign_conversions_intermediate = (campaign_impressions - (campaign_conversions_first + campaign_conversions_last ))
    return (((campaign_conversions_nrml + campaign_conversions_first + campaign_conversions_last ) * .7) + (campaign_conversions_intermediate * .3)) / campaign_impressions
uta = U_touch_attribution(df6)

uta

campaign_idx = range(0, 2)

fig = plt.figure(figsize=(15,4))
ax = fig.add_subplot(111)
plt.bar( range(len(uta[campaign_idx])), uta[campaign_idx],color = 'navajowhite', label='UTA' )
plt.xlabel('Campaign ID')
plt.ylabel('Return per impression')
plt.legend(loc='upper left')
plt.show()
st.pyplot()
