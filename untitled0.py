# -*- coding: utf-8 -*-
"""
Created on Tue May 22 14:58:34 2018

@author: sanch
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Reading the dataset in a dataframe using Pandas
df = pd.read_csv("train.csv") 
df.head(10)
df.describe()
df['Property_Area'].value_counts()

# Visualize Applicant Income
df['ApplicantIncome'].hist(bins=50)
df.boxplot(column='ApplicantIncome', by = 'Education')

# Visualize Loan Amount 
df['LoanAmount'].hist(bins=50)



# Data Manipulation
temp1 = df['Credit_History'].value_counts(ascending=True)
temp2 = df.pivot_table(values='Loan_Status',index=['Credit_History'],aggfunc=lambda x: x.map({'Y':1,'N':0}).mean())
print('Frequency Table for Credit History:')
print(temp1)

print('\nProbility of getting loan for each Credit History class:')
print(temp2)

temp3 = df['Married'].value_counts(ascending=True)
temp4 = df.pivot_table(values='Loan_Status',index=['Married'],aggfunc=lambda x: x.map({'Y':1,'N':0}).mean())
print('Frequency Table for Married History:')
print(temp3)

print('\nProbility of getting loan for each Married class:')
print(temp4)

# Visualize chances of getting loan 
fig = plt.figure(figsize=(8,4))
ax1 = fig.add_subplot(121)
ax1.set_xlabel('Credit_History')
ax1.set_ylabel('Count of Applicants')
ax1.set_title("Applicants by Credit_History")
temp1.plot(kind='bar')

ax2 = fig.add_subplot(122)
temp2.plot(kind = 'bar')
ax2.set_xlabel('Credit_History')
ax2.set_ylabel('Probability of getting loan')
ax2.set_title("Probability of getting loan by credit history")

# Visualize chances of getting loan -- Married
fig = plt.figure(figsize=(8,4))
ax1 = fig.add_subplot(121)
ax1.set_xlabel('Married status')
ax1.set_ylabel('Loan status')
ax1.set_title("Applicants by Married status")
temp3.plot(kind='bar')

ax2 = fig.add_subplot(122)
temp4.plot(kind = 'bar')
ax2.set_xlabel('Married status')
ax2.set_ylabel('Probability of getting loan')
ax2.set_title("Probability of getting loan by married status")


# Check missing values
df.apply(lambda x: sum(x.isnull()), axis=0)


# Fill missing values in LoanAmount
df['LoanAmount'].fillna(df['LoanAmount'].mean(), inplace = True)

df.boxplot(column='LoanAmount', by = ('Education','Self_Employed'))

# Frequency table of Self_Employed
df['Self_Employed'].value_counts() #since 86% of values are "no", it's safe to impute as "NO" as there is high probability of success
# Fill missing values in Self_Employed
df['Self_Employed'].fillna('No', inplace=True)


# Fill missing values in Credit_History
df['Credit_History'].value_counts()
df['Credit_History'].fillna(df['Credit_History'].mean(), inplace = True)


# Fill missing values in Gender
df['Gender'].value_counts()
df['Gender'].fillna(method='bfill', inplace=True)


# Fill missing values in Married 
df['Married'].value_counts()
df['Married'].fillna(method='bfill', inplace = True)


# Fill missing values in Dependents
df['Dependents'].value_counts()
df['Dependents'].fillna(method='bfill', inplace = True)


# Fill missing values in Self_Employed  
df['Self_Employed'].value_counts()
df['Self_Employed'].fillna('No', inplace = True)


# Fill missing values in Loan_Amount_Term
df['Loan_Amount_Term'].value_counts()
df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mean(), inplace = True)


# Merge to Sample_submission file
df.to_csv('Sample_Submission.csv')



