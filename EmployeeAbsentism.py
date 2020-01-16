# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 14:58:05 2019

@author: Abhishek Mandal
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
import os
import seaborn as sns


#Importing data
os.getcwd()
os.chdir('D:\\Data Science\\Datasets\\EmployeeAbsentism_Edwisor')
data = pd.read_excel("Absenteeism_at_work_Project.xls")


#Data wrangling
data['Month of absence'].head()
data.dtypes

data['Month of absence'].unique() 



#Sorting by month as its a cyclic data 
data = data.sort_values('Month of absence')

##Analysing the data
#Missing value analysis

data.isnull().sum().sort_values(ascending=False)

#Replacing 0 with modefor Month of absence column
month_mode = data.loc[:,"Month of absence"].mode()
month_mode = pd.to_numeric(month_mode, errors ='ignore')
data=data.replace({'Month of absence': {0: 3.0}}) 


#Replace all missing values with mode
cols = list(data.columns)
print(cols)
for col in cols:
    data[col] = data[col].fillna(data[col].mode().iloc[0])
    print(col)

#Dropping ID and saving it to variable
data_id = data['ID']
data = data.drop('ID', axis=1)


#Outlier analysis
#from scipy import stats
#data = data[(np.abs(stats.zscore(data)) < 3).all(axis=1)] #Remove all rows which have outlier in atleast 1 column

##Plotting diagrams
corrmat = data.corr(method='pearson')
#Absenteesm time in hours correlation matrix
corr_num = 20 #number of variables for heatmap
cols_corr = corrmat.nlargest(corr_num, 'Absenteeism time in hours')['Absenteeism time in hours'].index
corr_mat_sales = np.corrcoef(data[cols_corr].values.T)
sns.set(font_scale=1.25)
f, ax = plt.subplots(figsize=(12, 9))
hm = sns.heatmap(corr_mat_sales, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 7}, yticklabels=cols_corr.values, xticklabels=cols_corr.values)
plt.show()

# Pairplot for the most intresting parameters
# pair plots for variables with largest correlation
var_num = 8
vars = cols_corr[0:var_num]

sns.set()
sns.pairplot(data[vars], size = 2.5)
plt.show();

#Bar chart
cols.remove('Absenteeism time in hours')
for col in cols:
    data.groupby(col).sum().plot(y='Absenteeism time in hours', kind='bar') 
    
#Removing overfit
# Removes colums where the threshold of zero's is (> 99.95), means has only zero values 
X = data
overfit = []
for i in X.columns:
    counts = X[i].value_counts()
    zeros = counts.iloc[0]
    if zeros / len(X) * 100 > 99.95:
        overfit.append(i)

overfit = list(overfit)
print(overfit)          #Overfit is empty i.e. no overfitting

##Feature Engineering
#Calculating correlation coeff for highly correlated values
np.corrcoef(data['Height'], data['Body mass index'])  #-0.1202946
np.corrcoef(data['Weight'], data['Body mass index'])  #0.86293245
np.corrcoef(data['Age'], data['Service time'])  #0.66102651

#As we can see Weight/bmi and Age/Service time are highly correlated, so we can drop one of the values
data = data.drop('Service time', axis=1)
data = data.drop('Weight', axis=1)

##Model creation
#Creating test and train data
X_train = data.drop('Absenteeism time in hours', axis = 1)
Y_train = data['Absenteeism time in hours']
X_test = X_train


from datetime import datetime
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error, make_scorer, accuracy_score
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from mlxtend.regressor import StackingCVRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


#Defining folds and score functions

kfolds = KFold(n_splits=10, shuffle=True, random_state=42)

# model scoring and validation function
def cv_rmse(model, X_train=X_train):
    rmse = np.sqrt(-cross_val_score(model, X_train, Y_train,scoring="neg_mean_squared_error",cv=kfolds))
    return (rmse)

# rmsle scoring function
def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))

#Defining models
lightgbm = LGBMRegressor(objective='regression', 
                                       num_leaves=4, #was 3
                                       learning_rate=0.01, 
                                       n_estimators=10000,
                                       max_bin=200, 
                                       bagging_fraction=0.75,
                                       bagging_freq=5, 
                                       bagging_seed=7,
                                       feature_fraction=0.2, # 'was 0.2'
                                       feature_fraction_seed=7,
                                       verbose=-1,
                                       )

xgboost = XGBRegressor(learning_rate=0.01,n_estimators=3460,
                                      max_depth=3, min_child_weight=0,
                                      gamma=0, subsample=0.7,
                                      colsample_bytree=0.7,
                                      objective='reg:linear', nthread=-1,
                                      scale_pos_weight=1, seed=27,
                                      reg_alpha=0.00006)



# setup models hyperparameters using a pipline
# The purpose of the pipeline is to assemble several steps that can be cross-validated together, while setting different parameters.
# This is a range of values that the model considers each time in runs a CV
e_alphas = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007]
e_l1ratio = [0.8, 0.85, 0.9, 0.95, 0.99, 1]
alphas_alt = [14.5, 14.6, 14.7, 14.8, 14.9, 15, 15.1, 15.2, 15.3, 15.4, 15.5]
alphas2 = [5e-05, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008]




# Kernel Ridge Regression : made robust to outliers
ridge = make_pipeline(RobustScaler(), RidgeCV(alphas=alphas_alt, cv=kfolds))

# LASSO Regression : made robust to outliers
lasso = make_pipeline(RobustScaler(), LassoCV(max_iter=1e7, 
                    alphas=alphas2,random_state=42, cv=kfolds))

# Elastic Net Regression : made robust to outliers
elasticnet = make_pipeline(RobustScaler(), ElasticNetCV(max_iter=1e7, 
                         alphas=e_alphas, cv=kfolds, l1_ratio=e_l1ratio))


stack_gen = StackingCVRegressor(regressors=(ridge, elasticnet, lightgbm),
                                meta_regressor=elasticnet,
                                use_features_in_secondary=True)

# store models, scores and prediction values 
models = {'Ridge': ridge,
          'Lasso': lasso, 
          'ElasticNet': elasticnet,
          'lightgbm': lightgbm,
          'xgboost': xgboost}
predictions = {}
scores = {}

#Training the models
for name, model in models.items():
    
    model.fit(X_train, Y_train)
    predictions[name] = np.expm1(model.predict(X_test))
    score = cv_rmse(model, X_train=X_train)
    scores[name] = (score.mean(), score.std())
    
#Validating and training each model
# get the performance of each model on training data(validation set)
print('---- Score with CV_RMSLE-----')
score = cv_rmse(ridge)
print("Ridge score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = cv_rmse(lasso)
print("Lasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = cv_rmse(elasticnet)
print("ElasticNet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = cv_rmse(lightgbm)
print("lightgbm score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = cv_rmse(xgboost)
print("xgboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


#Fit the training data X, y
print('----START Fit----',datetime.now())
print('Elasticnet')
elastic_model = elasticnet.fit(X_train, Y_train)
print('Lasso')
lasso_model = lasso.fit(X_train, Y_train)
print('Ridge')
ridge_model = ridge.fit(X_train, Y_train)
print('lightgbm')
lgb_model_full_data = lightgbm.fit(X_train, Y_train)

print('xgboost')
xgb_model_full_data = xgboost.fit(X_train, Y_train)


print('stack_gen')
stack_gen_model = stack_gen.fit(np.array(X_train), np.array(Y_train))

#Blend model prediction
def blend_models_predict(X_test):
    return ((0.2  * elastic_model.predict(X_test)) + \
            (0.25 * lasso_model.predict(X_test)) + \
            (0.2 * ridge_model.predict(X_test)) + \
            (0.15 * lgb_model_full_data.predict(X_test)) + \
             (0.1 * xgb_model_full_data.predict(X_test)) + \
            (0.2 * stack_gen_model.predict(np.array(X_test))))
pred =  blend_models_predict(X_test)

print('RMSLE score on train data:')
print(rmsle(Y_train,pred))


error = mean_squared_error(Y_train, pred)
print('Test MSE: %.3f' % error)



from sklearn.metrics import r2_score
r2score = r2_score(Y_train, pred,sample_weight=None, multioutput='uniform_average')
print('R Squared value: %.3f' % r2score)

#MAPE
num = abs(pred-Y_train).sum()
den = abs(Y_train).sum()

mape = (num/den)*100
print('MAPE: %3f' %mape)


# plotting test vs prediction 
Y_train = Y_train.to_frame().reset_index()
plt.plot(Y_train) 
plt.plot(pred, color='red') 
plt.show()

 







