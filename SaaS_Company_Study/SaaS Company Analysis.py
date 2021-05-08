#!/usr/bin/env python
# coding: utf-8

# # SaaS Company Analysis
# ***

# ### Introduction
# 
# We are data analysts from a SaaS Company that works with a subscription business model, where the customer can buy monthly or yearly plans, and have full access to our service.
# In this study, we're going to take a look at some important business metrics of the company, and make a cohort analysis to provide insights to the growth team to increase the retention of our customers
# 
# 
# ### Steps
# 
# ##### 1) Imports <br>2) Defining functions <br> 3) Treating and understandig our data <br> 4) Calculating the revenue and MRR by breaking the dataframe in monthly records <br> 5) Calculating active customers, churn and  new customers monthly <br> 6) Cohort Analysis <br> 7) Studying acquisition channels and marketing campaign
# 

# ### 1) Imports <a id=’section_1’></a>

# In[1]:


import pandas as pd
import random
from datetime import datetime, timedelta, date
from dateutil.relativedelta import *
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
from matplotlib import ticker
import seaborn as sns
import warnings
from scipy.stats import chisquare

warnings.filterwarnings('ignore')


# ### 2) Defining Functions

# In[2]:


def add_month(data):
    novadata = data + relativedelta(months=1)
    return novadata

def sub_month(data):
    novadata = data + relativedelta(months=-1)
    return novadata

def get_month_diff(date_series1, date_series2, time:str):
    date1=pd.to_datetime(date_series1)
    date2=pd.to_datetime(date_series2)
    
    diff = (date1-date2) / np.timedelta64(1,time)
    diff= round(diff/30)
    diff = diff.astype(int)
    return diff

def get_first_day_of_month(data):
    return (data.replace(day=1))

def get_difference_between_lists (list1, list2):
   list_dif = [i for i in list1 + list2 if i not in list1 or i not in list2]
   return list_dif


# ### 3) Treating and understandig our data

# In[3]:


df = pd.read_csv('all_subscriptions.csv', parse_dates = ['start_date', 'end_date'])
display(df.sample(5))
display(df.shape)


# #### Checking for data quality. Do we have null values?

# In[4]:


df.isna().sum()


# #### Modifying the dataframe: Truncate the purchases by month, rank the customer's subscriptions in order of purchase
# 

# In[5]:


df['purchase_no'] = df.sort_values(['customer_id', 'start_date'], ascending=True).groupby(['customer_id']).cumcount() + 1
df['start_month']=df['start_date'].apply(get_first_day_of_month)
df


# #### Possible values for subscription types, price and media

# In[6]:


display(df[['subscription_type', 'price','media']].sort_values(['subscription_type', 'price', 'media']).drop_duplicates().reset_index(drop=True))


# #### Plotting the number of sales per month

# In[7]:


sales_per_month = df.groupby(['start_month']).size().reset_index(name='sales')

plt.figure(figsize=(10,6))
ax = sns.lineplot(data = sales_per_month, x='start_month', y='sales', label = 'sales')
plt.ylim(0,1200)
plt.style.use('seaborn-white')
plt.xlabel('Date', fontsize = 14)
plt.ylabel('Subscriptions Sold', fontsize = 14)
plt.title('Monthly Sales', fontsize = 18, fontweight = 'bold')
plt.grid(linestyle = '-')
plt.xticks(rotation=30)
plt.annotate('big increase', xy=(datetime(2018,7,1), 550), xytext = (datetime(2018,7,1), 650), backgroundcolor = "white", arrowprops = {"arrowstyle":"->", "color":"red"})
plt.show()

## Note that the chart contains not only data from new sales, but from renewals too


# The huge increase in sales in july-18 is probably related to a big marketing campaign. Checking our data, we could confirm that.

# In[8]:


df.query('start_month == datetime(2018,7,1)').groupby(['subscription_type','price','start_date']).size().head(10)


# #### The original dataframe shows records of every time the subscription is renewed. It would be interesting to analyse our data, considering only the first purchases of the customers.
# 

# In[9]:


first_purchases = df[df['purchase_no'] == 1]


# #### Plotting the acquisitions per media channel and subscription type

# In[10]:


sales_per_media = first_purchases.groupby(['media']).size().reset_index(name='acquisitions')
plt.style.use('seaborn-white')
plt.figure(figsize = (10,5))
sns.barplot(data=sales_per_media, x="media", y="acquisitions")
sns.set(font_scale=1.5)
plt.title('Acquisition per Channels', fontweight = 'bold')
plt.xlabel("Channel")
#plt.xticks(rotation=90)
plt.show()


# In[11]:


sales_per_subscription_type = first_purchases.groupby(['subscription_type','start_month']).size().reset_index(name='acquisitions')
plt.style.use('seaborn-white')
plt.figure(figsize = (10,5))
sns.lineplot(data=sales_per_subscription_type, x="start_month", y="acquisitions", hue = 'subscription_type')
sns.set(font_scale=1.5)
plt.title('Acquisitions per Subscription Type', fontweight = 'bold')
plt.grid(linestyle = '--')
plt.xticks(rotation=45)
plt.xlabel('Month')
plt.ylim(0,300)
plt.show()


# ### 4) Calculating the Revenue and MRR by breaking the dataframe in monthly records

# #### Since we want to understand our customers activity monthly, it makes sense to have a dataframe with the all the yearly subscriptions divided in 12 records, instead of one.
# 

# In[12]:


df['month_difference'] = get_month_diff(df['end_date'], df['start_date'],'D')
df['def_month'] = df.apply(lambda row: pd.date_range(row['start_date'], row['end_date'], freq='M', closed = "left"), axis=1)
df_deffered = df.explode('def_month')
df_deffered['price_def'] = df_deffered['price']/df_deffered['month_difference']
df_deffered['def_month']=df_deffered['def_month'].apply(get_first_day_of_month)
df_deffered.drop('month_difference', axis = 1, inplace = True)
df_deffered.reset_index(drop=True, inplace=True)
df.drop('def_month', axis = 1, inplace = True)


# #### Example of a YEARLY subscription, before and after our change
# 

# In[13]:


print("\n")
print("Original Dataframe")
display(df[df['subscription_id'] == 24])

print("\n")
print("New Dataframe")
display(df_deffered[df_deffered['subscription_id'] == 24])


# #### Calculating MRR (Monthly Recurring Revenue) and the amount actually received per Month

# In[14]:


revenue_per_month = df.groupby(['start_month'])['price'].apply(sum).reset_index()
mrr = df_deffered.groupby(['def_month'])['price_def'].apply(sum).reset_index()

plt.style.use('seaborn-white')
plt.figure(figsize=(10,6))
ax = sns.lineplot(data = mrr, x='def_month', y='price_def', label = 'MRR')
ax = sns.lineplot(data = revenue_per_month, x='start_month', y='price', label = 'Amount actually received')

ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("R${x:,.2f}")) 
ax.axvline(x=datetime(2019,12,1),linestyle='--', color = 'gray', label = 'end of the analyzed period')
ax.tick_params(axis='both', which='major', labelsize=12)
ax.tick_params(axis='both', which='minor', labelsize=12)
plt.ylim(0,60000)
plt.style.use('seaborn-white')
plt.xlabel('\n Date', fontsize = 16, fontweight = 'bold')
plt.ylabel('Revenue \n', fontsize = 16, fontweight = 'bold')
plt.title("Revenue per Month", fontsize = 20, fontweight = 'bold')
plt.xticks(rotation=0)
plt.legend(fontsize = 12)
plt.grid(linestyle="--")
plt.show()


# The difference between these series is basically the amount from yearly subscriptions. In the "Amount actually received", it is considered the full price of these plans, and in the MRR, this value is deffered between the months that the subscription is active. That's why the MRR is being considered in future months of the analysis.

# ### 5) Calculating Active Customers, Churn and New customers monthly

# #### Checking if the customers have only one active subscription per month

# In[15]:


display(df_deffered.groupby(['def_month','customer_id'])['subscription_id'].count().value_counts(normalize = True)*100)


# #### Making changes in our dataframe to understand our churn and new customers per month

# In[16]:


df_deffered = df_deffered.sort_values(by=['customer_id', 'def_month'], ascending=True).drop_duplicates()
df_deffered.reset_index(inplace=True, drop = True)
df_deffered['data_lag'] = df_deffered.groupby(['customer_id'])['def_month'].shift(1)
df_deffered['data_lead'] = df_deffered.groupby(['customer_id'])['def_month'].shift(-1)
df_deffered['next_month'] = df_deffered['def_month'].apply(add_month)
df_deffered['new'] = (df_deffered['def_month'].apply(sub_month)!=df_deffered['data_lag'])#*-1
df_deffered['churn'] = (df_deffered['def_month'].apply(add_month)!=df_deffered['data_lead'])#*-1
df_deffered.head(10)


# #### Plotting our active customers monthly, highlighting the churn and new customers per month.

# In[17]:


total = df_deffered.groupby('def_month')['customer_id'].count().to_frame(name='total').reset_index()
total['def_month_str'] = total['def_month'].dt.strftime('%Y-%m-%d')

churn_aux = df_deffered[(df_deffered['churn'] == True) & (df_deffered['def_month'] < '2019-12-01')]
churn = churn_aux.groupby('next_month')['customer_id'].count().to_frame(name='churn').reset_index()
churn['next_month_str'] = churn['next_month'].dt.strftime('%Y-%m-%d')

new_aux = df_deffered[(df_deffered['new'] == True) & (df_deffered['def_month'] < '2020-01-01')]
new = new_aux.groupby('def_month')['customer_id'].count().to_frame(name='novos').reset_index()
new['def_month_str'] = new['def_month'].dt.strftime('%Y-%m-%d')


fig = make_subplots(specs=[[{"secondary_y": True}]])

fig.add_trace(go.Bar(x=churn['next_month'],
                           y=churn['churn'], 
                           name = "Churn", 
                           marker_color='rgb(255,17,7)'))
fig.add_trace(go.Bar(x=new['def_month'], 
                               y=new['novos'], 
                               name = "New", 
                               marker_color='rgb(102,166,30)'
                        ), 
                               secondary_y = False)

fig.add_trace(go.Scatter(x=new['def_month'], 
                               y=total['total'], 
                               name = "Total", 
                               marker_color='rgb(242,183,1)'
                        ), 
                               secondary_y = False)

fig.update_layout(yaxis_title = "Customers", 
                        showlegend = False,
                        title = "Active Customers",
                        template = "ggplot2",
                        plot_bgcolor='rgba(0,0,0,0)',
                        font = dict(family= "Tahoma",
                                    size = 20,
                                    color = "#222A2A"))

fig.update_layout(showlegend=True)
fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor= 'rgb(179,179,179)')
fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgb(179,179,179)')
fig.update_yaxes(zeroline=True, zerolinewidth=2, zerolinecolor='Black')
fig.show()


# The company is acquiring more new customers than those who leave. Thus, there is an increase in the number of active customers monthly.

# ### 6) Cohort Analysis

# Since the acquiring new customers is usually more expensive than retaining/reactivating the one's we alredy have, it's important to understand how our retention rates are changing over time.

# #### Adding 2 columns in our dataframe to make it easier to create a cohort.  The first one indicates the month of the customer's first purchase, and the other the cohort index (the number of months between that record and the customers first purchase) 
# 

# In[18]:


list_cohort_index = []
list_customers_first_month = []
loop_count = 0
cohort_index = 0
customer_first_month = 0

# loop through the records, adding the cohort_index and customer first month to the corresponding list at the end of each loop. 
for i in df_deffered['new']:

    if i == True:
        cohort_index = 0
        customer_first_month = df_deffered.loc[loop_count,'def_month']
    else:
        cohort_index = cohort_index + 1
        
    list_cohort_index.append(cohort_index)
    list_customers_first_month.append(customer_first_month)
    loop_count = loop_count+1

df_deffered.loc[:,'month_cohort'] = list_cohort_index
df_deffered.loc[:,'entry_month'] = list_customers_first_month
df_deffered['entry_month_str'] = df_deffered['entry_month'].dt.strftime('%Y-%m-%d')
df_deffered.head(10)


# In[19]:


cohort_user = df_deffered[df_deffered['def_month']<'2020-01-01'] ## deleting future data from yearly subscriptions

# pivot table customers shows the number of customers in each cohort, pivot table retention, the % that remained from each cycle
pivot_table_customers = pd.pivot_table(cohort_user, index = 'entry_month_str',columns='month_cohort', values = 'customer_id', aggfunc=len)
pivot_table_retention = (pivot_table_customers.divide(pivot_table_customers[0],axis=0)*100).round(decimals=1) 


# #### Plotting the cohorts

# In[20]:


plt.figure(figsize=(30, 15))
sns.set(font_scale=1.5)
heatmap_plot = sns.heatmap(pivot_table_customers, mask=pivot_table_customers.isnull(), annot=True, fmt='', cmap='YlGnBu')
plt.title('Cohort: Number of Customers', fontsize = 30)
plt.xlabel('Months since entry', fontsize = 25)
plt.ylabel('Entry Month', fontsize = 25)

plt.show()


# In[21]:


plt.figure(figsize=(30, 15))
sns.set(font_scale=1.5)
heatmap_plot = sns.heatmap(pivot_table_retention, mask=pivot_table_retention.isnull(), annot=True, fmt='', cmap='YlGn')
plt.title('Cohort: Customers retained (%)', fontsize = 30)
plt.xlabel('Months since entry', fontsize = 25)
plt.ylabel('Entry Month', fontsize = 25)

plt.show()


# At first glance:
# 1) it doesn't seem like we have a big change in the retention over the entry months. 
# 2) As expected, the longer the customer is in the system, it is less likely he will churn.

# #### Two possible analysis to help increasing the retention: <br> 1) how different is the behavior from the customers that joined from different media channels? <br> 2) the customers who joined at the promotion remained as long as the others?

# ### 7) Studying acquisition channels and marketing campaign

# ### a) Acquisition Channels

# #### Facebook_Ads:

# In[22]:


facebook_ads = df_deffered[df_deffered['media']=='FACEBOOK_ADS']
cohort_user_facebook = facebook_ads[facebook_ads['def_month']<'2020-01-01']
pivot_table_customers_facebook = pd.pivot_table(cohort_user_facebook, index = 'entry_month_str',columns='month_cohort', values = 'customer_id', aggfunc=len)
pivot_table_retention_facebook = (pivot_table_customers_facebook.divide(pivot_table_customers_facebook[0],axis=0)*100).round(decimals=1) 

plt.figure(figsize=(30, 15))
sns.set(font_scale=1.5)
heatmap_plot = sns.heatmap(pivot_table_retention_facebook, mask=pivot_table_retention_facebook.isnull(), annot=True, fmt='', cmap='YlGn')
plt.title('Cohort: Facebook Customers retained (%)', fontsize = 30)
plt.xlabel('Months since entrance', fontsize = 25)
plt.ylabel('Entry Month', fontsize = 25)

plt.show()


# #### Google Ad Words

# In[23]:


google_ad_words = df_deffered[df_deffered['media']=='GOOGLE_AD_WORDS']
cohort_user_google = google_ad_words[google_ad_words['def_month']<'2020-01-01']
pivot_table_customers_google = pd.pivot_table(cohort_user_google, index = 'entry_month_str',columns='month_cohort', values = 'customer_id', aggfunc=len)
pivot_table_retention_google = (pivot_table_customers_google.divide(pivot_table_customers_google[0],axis=0)*100).round(decimals=1) 

plt.figure(figsize=(30, 15))
sns.set(font_scale=1.5)
heatmap_plot = sns.heatmap(pivot_table_retention_google, mask=pivot_table_retention_google.isnull(), annot=True, fmt='', cmap='YlGn')
plt.title('Cohort: Google Customers retained (%)', fontsize = 30)
plt.xlabel('Months since entrance', fontsize = 25)
plt.ylabel('Entry Month', fontsize = 25)

plt.show()


# #### Statistics about customers who remained more than 1 month

# In[24]:


print("Facebook")
display(pivot_table_retention_facebook[1].describe())
print("\n")
print("Google")
display(pivot_table_retention_google[1].describe())


# #### Plotting a histogram with the retention per channel after the first month

# In[25]:


facebook_ads = pivot_table_retention_facebook[1]
google_ad_words = pivot_table_retention_google[1]

plt.figure(figsize = (8,5))
plt.style.use('seaborn-white')
plt.title("Retention from different channels", fontweight = 'bold')
plt.hist(facebook_ads, bins=5, alpha=0.6, label='facebook_ads')
plt.hist(google_ad_words, bins=5, alpha=0.6, label='google_ad_words')
plt.xlabel("Retention in the first month (%)")
plt.ylabel("Number of months")
plt.legend(loc='upper left')
plt.show()


# The results above strongly suggests that people who joined via facebook ads are more likely to remain longer than the ones who joined via google ad words. Let's run a chi-squared test, to ensure that we have statistical evidence to make such an important call. Since it doesn't looks that there's an unusual change in the behaviour across channels over the months, let's run this test ignoring the month of the event.

# In[26]:


fexp = cohort_user_facebook[cohort_user_facebook['month_cohort'] == 0]['churn'].value_counts()
fobs = cohort_user_google[cohort_user_google['month_cohort'] == 0]['churn'].value_counts()

chi_2, p_value = chisquare(f_obs=fobs, f_exp=fexp)
print("The value of chi_2 is:", chi_2)
print("The value is", p_value)


# Since the p value is much lower than the 0.05, we have statistical evidence to confirm what we expected. The customers from facebook_ads have a higher retention after one month.
# 
# Therefore, assuming that the cost to acquire a customer (CAC) are similar for each acquisition channel, **it makes sense to prioritize facebook ads rather than google ad words, as they would bring more money to the company in the long term.**

# ### b) Marketing Campaign

# Since the campaign was giving discounts only to monthly subscriptions, it's fair to compare the behavior of the customers who joined in the campaign vs the other customers, excluding the one's who have a yearly subscription 

# In[27]:


all_customers_list = df_deffered['customer_id'].unique().tolist()
customers_promotion_list = df_deffered[df_deffered['price']==25]['customer_id'].unique().tolist()
all_sales_customers_promotion = df_deffered[df_deffered['customer_id'].isin(customers_promotion_list)]

customers_with_monthly_subscription_not_promotion_list = df_deffered[(df_deffered['month_cohort']== 0) & (df_deffered['price']==50)]['customer_id'].to_list()
all_sales_customers_with_monthly_subscription_not_promotion = df_deffered[df_deffered['customer_id'].isin(customers_with_monthly_subscription_not_promotion_list)]



# In[28]:


# Customers who joined in the campaign
cohort_user_promotion = all_sales_customers_promotion[all_sales_customers_promotion['def_month']<'2020-01-01']
pivot_table_customers_promotion = pd.pivot_table(cohort_user_promotion, index = 'entry_month_str',columns='month_cohort', values = 'customer_id', aggfunc=len)
pivot_table_retention_promotion = (pivot_table_customers_promotion.divide(pivot_table_customers_promotion[0],axis=0)*100).round(decimals=1) 

plt.figure(figsize=(30, 1))
sns.set(font_scale=1.5)
heatmap_plot = sns.heatmap(pivot_table_retention_promotion, mask=pivot_table_retention_promotion.isnull(), annot=True, fmt='', cmap='YlGn')
plt.title('Cohort: Customers retained (%)', fontsize = 30)
plt.xlabel('Months since entrance', fontsize = 25)
plt.ylabel('Entry Month', fontsize = 25)

plt.show()


# Note: 50,4% of the customers of the campaign were still active after 1 month. Now, let's compare this results to the customers that entered without the campaign.

# In[29]:


# Other customers, who joined buying a monthly subscription

cohort_user_not_promotion = all_sales_customers_with_monthly_subscription_not_promotion[all_sales_customers_with_monthly_subscription_not_promotion['def_month']<'2020-01-01']
pivot_table_customers_not_promotion = pd.pivot_table(cohort_user_not_promotion, index = 'entry_month_str',columns='month_cohort', values = 'customer_id', aggfunc=len)
pivot_table_retention_not_promotion = (pivot_table_customers_not_promotion.divide(pivot_table_customers_not_promotion[0],axis=0)*100).round(decimals=1) 

plt.figure(figsize=(30, 15))
sns.set(font_scale=1.5)
heatmap_plot = sns.heatmap(pivot_table_retention_not_promotion, mask=pivot_table_retention_not_promotion.isnull(), annot=True, fmt='', cmap='YlGn')
plt.title('Cohort: Customers retained (%)', fontsize = 30)
plt.xlabel('Months since entrance', fontsize = 25)
plt.ylabel('Entry Month', fontsize = 25)

plt.show()


# In[30]:


## Information about the retention after 1 month of customers who didn't join in the campaign
print("Main Statistics about retention after 1 month")
display(pivot_table_retention_not_promotion[1].describe())
print("\n")

plt.style.use('seaborn-white')

fig1, ax1 = plt.subplots()
ax1.set_title('Retention in the First Month', fontweight = 'bold')

ax1 = sns.boxplot(x=pivot_table_retention_not_promotion[1], orient = 'h')#.set(xlabel='Retention (%)')
ax1.axvline(x=int(pivot_table_retention_promotion[1]),linestyle='--', color = 'red', label = 'Retention from campaign users')
plt.xlabel("Retention (%)")
ax1.legend(bbox_to_anchor=(0.85, -0.2));


# The boxplot shows that **the customers who joined via the campaign don't have the same retention as the others. 
# <br> These results do not lead to an immediate conclusion about the sucess of the campaign or not, since the campaign might be important to increase the cash flow in a short term basis, expand the brand, etc. But it's important to study and consider the retention rate when planning the next marketing campaign.**
