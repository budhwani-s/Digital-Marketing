#!/usr/bin/env python
# coding: utf-8

# In[71]:


#get_ipython().system('pip install Faker')


# In[131]:

import streamlit as st
import matplotlib
matplotlib.use('TkAgg')
import time
import pandas as pd
#from faker import Faker 
#import matplotlib.pyplot as plt 

#fake = Faker('')    


# In[73]:


import random


# In[74]:


snack = ['protein bar', 'chips', 'cookeis', 'mixed nuts', 'fruit snacks',
                'crackers', 'muffins', 'popcorns','David Sunflower Seeds',
         'Raisinets','Welch Fruit Snacks','Premium Saltines','Corn Nuts',
        'Popchips','Trolli','Donettes','Chicken in a Biskit','Swedish Fish',
         'Twizzlers','Hostess Pies','Honey Maid Grahams'
        'Zebra Cakes','Kettle Chips','Milanos','Starburst']
len(snack)

st.sidebar.title("What to do") 
add_selectbox = st.sidebar.selectbox( 'Analyzing SVD algorithm with simple protein dataset?', 
                ('Show instructions', 'Show the source code', 'Choose what you want!','Graph 1 - Actual value Vs Predicted Value'))

if add_selectbox == "Show instructions" :
        with st.echo():
            st.write('Created Protein dataset')
   # st.sidebar.success('To continue select "Run the app".')
elif add_selectbox == "Show the source code":
        with st.echo():
            st.write('snack_comp = list(zip(snack_id,snack,snack_weight,snack_rate))')
elif add_selectbox == "Choose what you want!":
            column = st.selectbox("What bar do you want?",snack)
        
my_bar = st.progress(0)
for percent_complete in range(100):
        time.sleep(0.1)
        my_bar.progress(percent_complete + 1)

# In[75]:


snack_id = ['100001','100002','100003','100004','100005','100006','100007','100008','100009','100010','100011','100012','100013','100014','100015','100016','100017','100018','100019','100020','100021','100022','100023','100024']


# In[76]:


snack_weight=['25 gm','30 gm','20 gm','25 gm','40 gm','45 gm','40 gm','20 gm','10 gm','20 gm','30 gm','50 gm','60 gm','40 gm','25 gm','35 gm','35 gm','20 gm','20 gm','70 gm','10 gm','30 gm','25 gm','10 gm']


# In[77]:


snack_rate=['$ 2.00','$ 2.00','$ 2.00','$ 2.00','$ 2.00','$ 2.00','$ 2.00','$ 1.99','$ 2.99','$ 2.49','$ 2.99','$ 2.49','$ 2.99','$ 2.99','$ 4.99','$ 3.49','$ 2.89','$ 2.49','$ 3.99','$ 2.79','$ 2.29','$ 2.49','$ 2.99','$ 3.00',]


# In[78]:


snack_comp = list(zip(snack_id,snack,snack_weight,snack_rate))
snack_comp_1 = list(zip(snack_id,snack,snack_weight,snack_rate))
snack_comp = pd.DataFrame(snack_comp, columns =('Snack Id','Snack Name', 'Weight', 'Rate'))
snack_comp


# In[79]:


snack_comp['Snack Id'] = snack_comp['Snack Id'].astype(str).astype(int)
snack_comp['Snack Id']


# In[80]:


def user_id_gen(how_many):
  user_id = []
  for _ in range(0,how_many):
    user_id.append(random.randint(500001,500500))
  return user_id


# In[81]:


user_id = []
for i in user_id_gen(5000):
    user_id.append(i)
len(user_id)


# In[82]:


x = len(user_id)
x


# In[83]:


def snack_dataset(how_many):
  s_all = []
  for _ in range(0,how_many):
    s_all.append(random.choice(snack_comp_1))
  return s_all


# In[84]:


s_name = []
for i in snack_dataset(5000):
    s_name.append(i)
len(s_name)


# In[85]:


snack_complete = pd.DataFrame(s_name, columns =('Snack Id','Snack Name', 'Weight', 'Rate'))
snack_complete['User-Id'] = user_id
snack_complete['Snack Id']


# In[86]:


snack_complete['Snack Id'] = snack_complete['Snack Id'].astype(str).astype(int)
snack_complete['Snack Id']


df = snack_complete[['User-Id','Snack Id','Review']]

df.plot(x='Snack Id', y='User-Id',kind='bar')
#plt.show()
st.pyplot
# In[284]:

# In[87]:


def review_generate(how_many):
  review = []
  for _ in range(0,how_many):
    review.append(random.randint(0,5))
  return review


# In[88]:


review = []
for i in review_generate(5000):
    review.append(i)
review
snack_complete['Review'] = review
snack_complete


# In[89]:


df = snack_complete[['User-Id','Snack Id','Review']]


# In[90]:


import sys
sys.path.append("../../")
import os
import surprise
import papermill as pm
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_squared_error


from reco_utils.common.timer import Timer
from reco_utils.dataset import movielens
from reco_utils.dataset.python_splitters import python_random_split
from reco_utils.evaluation.python_evaluation import (rmse, mae, rsquared, exp_var, map_at_k, ndcg_at_k, precision_at_k, 
                                                     recall_at_k, get_top_k_items)
from reco_utils.recommender.surprise.surprise_utils import predict, compute_ranking_predictions

print("System version: {}".format(sys.version))
print("Surprise version: {}".format(surprise.__version__))


# In[91]:


df['Review']


# In[92]:


train, test = python_random_split(df, 0.75)


# In[93]:


from surprise import Dataset
from surprise import Reader


# In[94]:


reader = Reader(rating_scale=(1, 5))


# In[95]:


train_set = Dataset.load_from_df(train, reader).build_full_trainset()
train_set


# In[96]:


svd = surprise.SVD(random_state=0, n_factors=200, n_epochs=30, verbose=True)


# In[97]:


with Timer() as train_time:
    svd.fit(train_set)

print("Took {} seconds for training.".format(train_time.interval))


# In[98]:


predictions = [svd.predict(row._1, row._2)
              for row in test.itertuples()]
predictions = pd.DataFrame(predictions)
predictions


# In[99]:


predictions = predictions.rename(index=str, columns={"uid": 'User-Id', "iid": 'Snack Id', "est": 'Review'})
predictions = predictions.drop(["details", "r_ui"], axis="columns")


# In[100]:


predictions


# In[101]:


with Timer() as test_time:
    all_predictions = compute_ranking_predictions(svd, train, usercol='User-Id', itemcol='Snack Id', remove_seen=True)
    
print("Took {} seconds for prediction.".format(test_time.interval))


# In[102]:


suffixes = ["_true", "_pred"]
rating_true_pred = pd.merge(test, predictions, on=["User-Id", 'Snack Id'], suffixes=suffixes)


# In[103]:


rating_true_pred


# In[104]:


def merge_rating_true_pred(
    rating_true,
    rating_pred,
    col_user,
    col_item,
    col_rating,
    col_prediction,
):
    if col_rating in rating_pred.columns:
        col_rating = col_rating + suffixes[0]
    if col_prediction in rating_true.columns:
        col_prediction = col_prediction + suffixes[1]
    return rating_true_pred[col_rating], rating_true_pred[col_prediction]


# In[105]:


y = merge_rating_true_pred(test,predictions,'User-Id','Snack Id','Review_true','Review_pred');
y_act = y[0]
y_pred = y[1]


# In[106]:


eval_rmse = np.sqrt(mean_squared_error(y_act, y_pred))
eval_rmse


# In[107]:
if st.checkbox('Snack Dataframe - Snack details'):
    st.write(snack_comp)
    st.dataframe(snack_comp.style.highlight_max(axis=0))

def mae(
    rating_true,
    rating_pred,
    col_user,
    col_item,
    col_rating,
    col_prediction,
):
    y_true, y_pred = merge_rating_true_pred(
        rating_true=rating_true,
        rating_pred=rating_pred,
        col_user=col_user,
        col_item=col_item,
        col_rating=col_rating,
        col_prediction=col_prediction,
    )
    return mean_absolute_error(y_true, y_pred)


# In[108]:


eval_mae = mean_absolute_error(y_act, y_pred)
eval_mae 


# In[109]:


eval_rsquared = r2_score(y_act, y_pred)
eval_rsquared


# In[110]:


eval_exp_var = explained_variance_score(y_act, y_pred)
eval_exp_var


# In[111]:


print("RMSE:\t\t%f" % eval_rmse,
      "MAE:\t\t%f" % eval_mae,
      "rsquared:\t%f" % eval_rsquared,
      "exp var:\t%f" % eval_exp_var, sep='\n')


# In[112]:


common_users = set(test['User-Id']).intersection(set(all_predictions['User-Id']))
common_users


# In[113]:


rating_true_common = test[test['User-Id'].isin(common_users)]
rating_pred_common = all_predictions[all_predictions['User-Id'].isin(common_users)]
n_users = len(common_users)


# In[114]:


def merge_ranking_true_pred(
    rating_true,
    rating_pred,
    col_user,
    col_item,
    col_rating,
    col_prediction,
    relevancy_method,
    k,
    threshold,
):
    # Return hit items in prediction data frame with ranking information. This is used for calculating NDCG and MAP.
    # Use first to generate unique ranking values for each item. This is to align with the implementation in
    # Spark evaluation metrics, where index of each recommended items (the indices are unique to items) is used
    # to calculate penalized precision of the ordered items.
    if relevancy_method == "top_k":
        top_k = k
    elif relevancy_method == "by_threshold":
        top_k = threshold
    else:
        raise NotImplementedError("Invalid relevancy_method")
    df_hit = get_top_k_items(
        dataframe=rating_pred_common,
        col_user=col_user,
        col_rating=col_prediction,
        k=top_k,
    )
    df_hit = pd.merge(df_hit, rating_true_common, on=[col_user, col_item])[
        [col_user, col_item, "rank"]
    ]

    # count the number of hits vs actual relevant items per user
    df_hit_count = pd.merge(
        df_hit.groupby(col_user, as_index=False)[col_user].agg({"hit": "count"}),
        rating_true_common.groupby(col_user, as_index=False)[col_user].agg(
            {"actual": "count"}
        ),
        on=col_user,
    )

    return df_hit, df_hit_count, n_users


# In[115]:


df_hit, df_hit_count, n_users = merge_ranking_true_pred(
        rating_true=test,
        rating_pred=all_predictions,
        col_user='User-Id',
        col_item='Snack Id',
        col_rating='Review',
        col_prediction='prediction',
        relevancy_method='top_k',
        k=10,
        threshold=10,
    )


# In[116]:


df_hit_sorted = df_hit.copy()
df_hit_sorted["rr"] = (df_hit_sorted.groupby(df_hit_sorted["User-Id"]).cumcount() + 1) / df_hit_sorted["rank"]
df_hit_sorted = df_hit_sorted.groupby(df_hit_sorted["User-Id"]).agg({"rr": "sum"}).reset_index()
df_merge = pd.merge(df_hit_sorted, df_hit_count, on=df_hit_sorted["User-Id"])


# In[117]:


eval_map =  (df_merge["rr"] / df_merge["actual"]).sum() / n_users
eval_map


# In[118]:


# calculate discounted gain for hit items
df_dcg = df_hit.copy()
# relevance in this case is always 1
df_dcg["dcg"] = 1 / np.log1p(df_dcg["rank"])


# In[119]:


# sum up discount gained to get discount cumulative gain
df_dcg = df_dcg.groupby(df_dcg['User-Id'], as_index=False, sort=False).agg({"dcg": "sum"})
# calculate ideal discounted cumulative gain
df_ndcg = pd.merge(df_dcg, df_hit_count, on=df_dcg['User-Id'])
df_ndcg["idcg"] = df_ndcg["actual"].apply(
    lambda x: sum(1 / np.log1p(range(1, min(x, 10) + 1)))
)

# DCG over IDCG is the normalized DCG
eval_ndcg = (df_ndcg["dcg"] / df_ndcg["idcg"]).sum() / n_users
eval_ndcg


# In[120]:


eval_precision = (df_hit_count["hit"] / 10).sum() / n_users
eval_precision


# In[121]:


eval_recall = (df_hit_count["hit"] / df_hit_count["actual"]).sum() / n_users
eval_recall


# In[122]:


print("RMSE:\t\t%f" % eval_rmse,
      "MAE:\t\t%f" % eval_mae,
      "rsquared:\t%f" % eval_rsquared,
      "exp var:\t%f" % eval_exp_var, sep='\n')

print('----')

print("MAP:\t%f"          % eval_map,
      "NDCG:\t%f"         % eval_ndcg,
      "Precision@10:\t%f" % eval_precision,
      "Recall@10:\t%f"    % eval_recall, sep='\n')


# In[123]:


#get_ipython().system(' pip install sklearn_evaluation')


# In[126]:


y_true = y_act.values.tolist()
y_predict = y_pred.values.tolist()

#str(y_true


# In[127]:


y_true = np.asarray(y_true)
y_true
y_predict = np.asarray(y_predict)
y_predict


# In[128]:


import seaborn as sns
sns.distplot((y_act-y_pred))
st.pyplot


# In[129]:


import numpy as np
acc_test = np.array(y_act)
pred_test = np.array(y_pred)


df1 = pd.DataFrame({'Actual': acc_test.flatten(), 'Predicted': pred_test.flatten()})
df1


# In[132]:


df2 = df1.head(n=25)

df2.plot(kind='bar',figsize=(10,8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()
st.pyplot

# In[133]:


df.hist(figsize=(17,17), color='#86bf91', zorder=2, rwidth=0.5)


# In[134]:


sns.jointplot(x="User-Id", y="Snack Id", data=df, kind = 'reg', height = 10)
plt.show()
st.pyplot


# In[135]:


from scipy.stats import norm
from scipy import stats

sns.distplot(df['Review'], fit=norm);
fig = plt.figure()
res = stats.probplot(df['Review'], plot=plt)
st.pyplot

# In[136]:


k = 3
#number of variables for heatmap
cols = corr.nlargest(k, 'Review')['Review'].index
cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale=1.25)
f, ax = plt.subplots(figsize=(15, 15))
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()
st.pyplot

# In[137]:


sns.set()
# cols = ['price', 'sqft_living', 'sqft_above', 'sqft_living15', 'bathrooms', 'view', 'sqft_basement', 'bedrooms', 'lat', 'waterfront']
sns.pairplot(df[cols], height = 2.5)
plt.show();


# In[ ]:





# In[ ]:




