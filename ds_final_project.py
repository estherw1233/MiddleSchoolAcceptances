#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 02:23:36 2020

@author: esther
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from scipy import stats
from sklearn import linear_model
import statsmodels.api as sm 

data = pd.read_csv("middleSchoolData.csv", delimiter = ",")

# created two datasets - one with all systematically missing rows are removed
# columns about "Per student spending, in $" and "Average class size" 
# are missing for charter schools
# columns are dropped
df_noEF = data.drop(["per_pupil_spending", "avg_class_size"], axis = 1)
# new_df = df_noEF.fillna(df_noEF.mean())
'''
"""
# dropping rows that have nan data

# this drops the charter schools, which is fine because I will just analyze
# the relationships of the variables of public high schools
# charter schools were missing info about student spending and average class size
# which I thought was important to study when determing HSPHS enrollment 

df = data.dropna(axis = 0)
print(df) # <-- df has no Nan and only public schools
"""
#1  correlation between the number of applications(2) and admissions to HSPHS(3)
x = df_noEF.iloc[:, 2] # applications
y = df_noEF.iloc[:, 3] # number of admissions

plt.plot(x, y, "o")
plt.xlabel("number of applications")
plt.ylabel("number of admissions")
plt.title("Effect of Number of Applications on Admissions to HSPHS")

r, p = pearsonr(x, y)

#2 Raw number of applications vs application *rate*
df_noEF["application rate"] = df_noEF["applications"] / df_noEF["school_size"]
x_rate = df_noEF.iloc[:, 22]
# fill the na with mean of column
new_x = x_rate.fillna(x_rate.mean())

plt.plot(new_x, y, "o", color = "red")
plt.xlabel("application rate")
plt.ylabel("number of admissions")
plt.title("Effect of Application Rate and Number of Apps on Admissions to HSPHS")

r_rate, p_rate = pearsonr(new_x, y)
'''
#%%3 best *per student* odds of sending someone to HSPHS
df_noEF["student odd"] = df_noEF["acceptances"] / (df_noEF["applications"] - df_noEF["acceptances"])
max_rate = df_noEF[df_noEF["student odd"] == df_noEF["student odd"].max()]

# %%4 relationship between how students perceive their school and how the school performs on objective measures of achievement
# PCA to see which factors of each variable matter most
# PCA of how students perceive their school

# since there is systematically missing data, removing rows that have Nan values
# removed together because we need to have the same size data in order to find correlation
ratings_perform = df_noEF.iloc[:, 9:22]
new_rp = ratings_perform.fillna(ratings_perform.mean())

# PCA on ratings dataset
ratings = new_rp.iloc[:, 0:6]
# show correlation between each variable, not that many highly correlated variables
r = np.corrcoef(ratings, rowvar = False)
#plt.imshow(r)
#plt.colorbar()

# create normally distributed data
z_ratings = stats.zscore(ratings)
pca = PCA()

pca.fit(z_ratings)
eig_vals = pca.explained_variance_
loadings = pca.components_

rotated_data_1 = pca.fit_transform(z_ratings)
covar_explained = eig_vals / sum(eig_vals) * 100

num_ratings = len(eig_vals)
"""
plt.bar(np.linspace(1, num_ratings, num_ratings), eig_vals)
plt.title("scree plot for ratings")
plt.xlabel('Principal component')
plt.ylabel('Eigenvalue')
plt.plot([0,num_ratings],[1,1],color='red',linewidth=1) # Kaiser criterion line

plt.bar(np.linspace(1, num_ratings, num_ratings), loadings[:,0]) #2 (collaborative teachers)
plt.xlabel('Question')
plt.ylabel('Loading')


"""
# draw PCA plot
perform = new_rp.iloc[:, 10:13] 
z_perform = stats.zscore(perform)
pca = PCA()
pca.fit(z_perform)

eig_vals = pca.explained_variance_
loadings = pca.components_

rotated_data_2 = pca.fit_transform(z_perform)
covar_explained = eig_vals / sum(eig_vals) * 100

num_perform = len(eig_vals)


plt.bar(np.linspace(1, num_perform, num_perform), eig_vals)
plt.title("scree plot for performance")
plt.xlabel('Principal component')
plt.ylabel('Eigenvalue')
plt.plot([0,num_perform],[1,1],color='red',linewidth=1) # Kaiser criterion line

plt.bar(np.linspace(1, num_perform, num_perform), loadings[:,0]) #1 (student achievement)
plt.xlabel('Question')
plt.ylabel('Loading')

rotated_rating = rotated_data_1[:, 0]
rotated_achieve = rotated_data_2[:, 0]
new_corr = np.corrcoef(rotated_rating, rotated_achieve)

#new_ratings = ratings["collaborative_teachers"]
#new_perform = perform["student_achievement"]
#relation = np.corrcoef(new_ratings, new_perform)
#corr, _ = pearsonr(new_ratings, new_perform)

# %% 5 hypothesis small vs large school admission to HSPHS
# null: The size of the school doesn't have an effect on the number of admissions to HSPHS
# alternate: The size of the school does effect the number of admissions 

# ks_test
new_df = df_noEF[["acceptances", "school_size"]].dropna()
size_mean = new_df["school_size"].mean()
small_school = new_df[new_df["school_size"] <= size_mean]
large_school = new_df[new_df["school_size"] > size_mean]

hist = plt.hist(new_df["acceptances"])

x = small_school["acceptances"]
y = large_school["acceptances"]

KS, p = stats.ks_2samp(x, y)


#%% 6 availability of material resources (e.g. per student spending or class size) impacts objective measures of achievement or admission to HSPHS
import pandas as pd

data = pd.read_csv("middleSchoolData.csv", delimiter = ",")
df = data.dropna(axis = 0)

x = df["per_pupil_spending"]
y = df["acceptances"]
m, b = np.polyfit(x, y, 1)

plt.plot(x, y, "o")
plt.plot(x, m * x + b, color = "red")
plt.xlabel("per student spending")
plt.ylabel("num of acceptances")
plt.title("student spending vs. acceptances")

spending_corr = np.corrcoef(x, y)

# %% 7 proportion of schools accounts for 90% of all students accepted to HSPHS
total_acceptances = np.array(df_noEF["acceptances"])
acceptance_sum = np.sum(total_acceptances)

sorted_acceptance = np.array(df_noEF.sort_values("acceptances", ascending = False))
percent = acceptance_sum * 0.9

counter = 0
position = 0
schools = []
while counter < percent:
    counter += sorted_acceptance[position][3]
    schools.append(sorted_acceptance[position][1])
    position += 1
    
print(percent)
print(counter)
print(len(schools))
print(position)
# %%8 multiple regression
import statsmodels.api as sm 
# a - sending students to HSPHS
# PCA to see most important factors
df_acceptances = df_noEF.dropna(axis = 0)
#applications = df_acceptances.iloc[:, 2]
factors = df_acceptances.iloc[:, 2:22]
#complete_factors = pd.concat([applications, factors], axis = 1)

z_accept_factors = stats.zscore(factors)

pca = PCA()
pca.fit(factors)

e_factors = pca.explained_variance_
l_factors = pca.components_

rotated_data_factor = pca.fit_transform(factors)
covar_explained = eig_vals / sum(eig_vals) * 100

num_factors = len(e_factors)
"""
plt.bar(np.linspace(1, num_factors, num_factors), e_factors)
plt.title("scree plot for factors")
plt.xlabel('Principal component')
plt.ylabel('Eigenvalue')
plt.plot([0,num_factors],[1,1],color='red',linewidth=1) # Kaiser criterion line

plt.bar(np.linspace(1, num_factors, num_factors), l_factors[:,2]) #1,8 (application, collaborative teacher)
plt.xlabel('Question')
plt.ylabel('Loading')
"""

factor_arr = np.array(df_acceptances)
A = np.transpose([factor_arr[:,2],factor_arr[:,4],factor_arr[:,5],
                  factor_arr[:,6],factor_arr[:,7],factor_arr[:,8],
                  factor_arr[:,9], factor_arr[:,10],factor_arr[:,11],
                  factor_arr[:,12],factor_arr[:,13],factor_arr[:,14],
                  factor_arr[:,15],factor_arr[:,16],factor_arr[:,17],
                  factor_arr[:,18],factor_arr[:,19],factor_arr[:,20],
                  factor_arr[:,21]]) # 21
B = np.transpose([factor_arr[:,2],factor_arr[:,3],factor_arr[:,4],
                  factor_arr[:,5],factor_arr[:,6],factor_arr[:,7],
                  factor_arr[:,8],factor_arr[:,9], factor_arr[:,10],
                  factor_arr[:,11],factor_arr[:,12],factor_arr[:,13],
                  factor_arr[:,14],factor_arr[:,15],factor_arr[:,16],
                  factor_arr[:,17],factor_arr[:,18],factor_arr[:,20],
                  factor_arr[:,21]]) # 21

Y = df_acceptances["acceptances"] # acceptances 11(student achievement), 19 (supportive_environment)
Z = df_acceptances["student_achievement"]
regr = linear_model.LinearRegression() # linearRegression function from linear_model
regr.fit(B,Z) # fit model
print(regr.score(B,Z)) # r^2
print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_) #beta, larger = more important






































