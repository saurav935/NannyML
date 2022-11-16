#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nannyml as nml
import pandas as pd
from IPython.display import display


# In[2]:


data = pd.read_csv('car_loan_default_data.csv')
display(data)


# In[3]:


reference = data.loc[data['partition'] == 'reference'].copy()
analysis = data.loc[data['partition'] == 'analysis'].copy()


# In[4]:


reference.head()


# In[5]:


analysis.head()


# # Performance estimation

# In[6]:


estimator = nml.CBPE(
    y_pred_proba='y_pred_proba',
    y_pred='y_pred',
    y_true='repaid',
    timestamp_column_name = 'timestamp',
    metrics=['roc_auc'],
    chunk_size=5000,
    problem_type='classification_binary',
)
estimator = estimator.fit(reference)
estimated_performance = estimator.estimate(analysis)

# Show results
figure = estimated_performance.plot(kind='performance', metric='roc_auc', plot_reference=True)
figure.show()


# # Univariate feature drift

# In[7]:


reference.head()


# In[8]:


analysis.head()


# In[9]:


column_names = [col for col in reference.columns if col not in ['timestamp']]
calc = nml.UnivariateDriftCalculator(
    column_names=column_names,
    timestamp_column_name='timestamp',
    continuous_methods=['kolmogorov_smirnov', 'jensen_shannon'],
    categorical_methods=['chi2', 'jensen_shannon'],
)


# In[10]:


calc.fit(reference)
uni_results = calc.calculate(analysis)
uni_results.to_df()


# In[11]:


for column_name in uni_results.continuous_column_names:
    drift_fig = uni_results.plot(
        kind='drift',
        column_name=column_name,
        method='jensen_shannon',
        plot_reference=True
    )
    drift_fig.show()


# In[12]:


for column_name in uni_results.categorical_column_names:
    drift_fig = uni_results.plot(
        kind='drift',
        column_name=column_name,
        method='chi2',
        plot_reference=True
    )
    drift_fig.show()


# # Distribution of continuous variables

# In[13]:


for column_name in uni_results.continuous_column_names:
    figure = uni_results.plot(
        kind='distribution',
        column_name=column_name,
        method='jensen_shannon',
        plot_reference=True
    )
    figure.show()


# # Distribution of categorical variables

# In[14]:


for column_name in uni_results.categorical_column_names:
    figure = uni_results.plot(
        kind='distribution',
        column_name=column_name,
        method='chi2',
        plot_reference=True)
    figure.show()


# # Multivariate drift

# In[15]:


# Define feature columns
feature_column_names = [
    col for col in reference.columns if col not in ['timestamp']]

from sklearn.impute import SimpleImputer

calc = nml.DataReconstructionDriftCalculator(
    column_names=feature_column_names,
    timestamp_column_name='timestamp',
    chunk_size=5000,
    imputer_categorical=SimpleImputer(strategy='constant', fill_value='missing'),
    imputer_continuous=SimpleImputer(strategy='median')
)
calc.fit(reference)
results = calc.calculate(analysis)


# In[16]:


results.to_df()


# In[17]:


figure = results.plot(plot_reference=True)
figure.show()


# # Ranking

# In[18]:


ranker = nml.Ranker.by('alert_count')
ranked_features = ranker.rank(uni_results, only_drifting = False)
ranked_features


# In[ ]:




