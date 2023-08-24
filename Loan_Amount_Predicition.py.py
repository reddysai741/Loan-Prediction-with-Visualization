#!/usr/bin/env python
# coding: utf-8

# # Exploratory Data Analysis(EDA)

# In[2]:


# import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# **Test Dataset**

# In[22]:


#load the test dataset
a=pd.read_csv("test.csv")
a


# In[ ]:


a.describe


# In[ ]:


# show the shape of the dataset i.e. no of rows, no of columns
a.shape


# In[ ]:


a_length=len(a)
a_length


# In[ ]:


# take a look at the features (i.e. independent variables) in the dataset
a.columns


# In[ ]:


# show the data types for each column of the test set
a.dtypes


# In[ ]:


# concise summary of the dataset, info about index dtype, column dtypes, non-null values and memory usage
a.info()


# In[35]:


a.mean()


# In[36]:


a.std()


# **Independent Variable(Categorical)**

# # *frequency table of a variable will give us the count of each category in that variable*

# In[ ]:


a['Married'].value_counts()


# In[ ]:


a['Gender'].value_counts()


# In[ ]:


a['Dependents'].value_counts()


# In[ ]:


a['Self_Employed'].value_counts()


# In[ ]:


a['Loan_Amount_Term'].value_counts()


# In[ ]:


a['Credit_History'].value_counts()


# **Data Pre-processing**

# ***Missing value and outlier treatment***

# In[12]:


# check for missing values
a.apply(lambda x:sum(x.isnull()),axis=0)


# There are missing values in Gender, Married, Dependents, Self_Employed, LoanAmount, Loan_Amount_Term and Credit_History features. We will treat the missing values in all the features one by one.
# 
# * For numerical variables: imputation using mean or median
# 
# * For categorical variables: imputation using mode
# 
# There are very less missing values in Gender, Married, Dependents, Credit_History and Self_Employed features so we can fill them using the mode of the features. If an independent variable in our dataset has huge amount of missing data e.g. 80% missing values in it, then we would drop the variable from the dataset.

# In[ ]:


# replace missing values with the mean ,max and others
a['LoanAmount'].fillna(a['LoanAmount'].mean(), inplace=True)
a['Loan_Amount_Term'].fillna(a['Loan_Amount_Term'].mean(), inplace=True)
a['Credit_History'].fillna(a['Credit_History'].max(), inplace=True)
a['Gender'].fillna('Female',inplace=True)
a['Married'].fillna('Yes',inplace=True)
a['Dependents'].fillna('0',inplace=True)
a['Self_Employed'].fillna('No',inplace=True)


# In[ ]:


a.head()


# In[ ]:


# check whether all the missing values are filled in the test dataset
a.apply(lambda x:sum(x.isnull()),axis=0)


# **Note:-**We need to replace the missing values in Test set using the mode/median/mean of the Training set, not from the Test set. Likewise, if you remove values above some threshold in the test case, make sure that the threshold is derived from the training and not test set. Make sure to calculate the mean (or any other metrics) only on the train data to avoid data leakage to your test set

# In[ ]:


# Visualizing categorical features
# plt.figure(1)
a['Gender'].value_counts().plot.bar(figsize=(16,8),title="Gender")
plt.show()
a['Married'].value_counts().plot.bar(title="Married")
plt.show()
a['Self_Employed'].value_counts().plot.bar(title="Self_employed")
plt.show()
a['Credit_History'].value_counts().plot.bar(title="Credit_History")
plt.show()


# In[23]:


sns.countplot(x=a['Gender'])
plt.show()
sns.countplot(x=a['Dependents'])
plt.show()
sns.countplot(x=a['Education'])
plt.show()
sns.countplot(x=a['Credit_History'])
plt.show()
sns.countplot(x=a['Property_Area'])


# It can be inferred from the above bar plots that:
# 
# 
# *   80% applicants in the dataset are male.
# *   Around 65% of the applicants in the dataset are married.
# 
# *   Around 15% applicants in the dataset are self employed.
# *   Around 85% applicants have credit history (repaid their debts).
# 
# *   Around 80% of the applicants are Graduate.
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 

# In[31]:


plt.pie(b.Property_Area.value_counts(),[0,0,0],labels=['Semi urban','Urban','Rural'])
plt.show()
plt.pie(b.Education.value_counts(),[0,0],labels=['Graduate','Not Graduate'])
plt.show()
plt.pie(b.Gender.value_counts(),[0,0],labels=['Male','Female'])
plt.show()
plt.pie(b.Self_Employed.value_counts(),[0,0],labels=['No','Yes'])


# **Independent Variable(ordinal)**

# In[ ]:


plt.subplot(121)
a['Dependents'].value_counts(normalize=True).plot.bar(figsize=(12,4), title= 'Dependents')

plt.subplot(122)
a['Property_Area'].value_counts(normalize=True).plot.bar(title= 'Property_Area')

plt.show()


# 
# 
# *   More than half of the applicants don’t have any dependents.
#  
# * Most of the applicants are from     Semiurban area.   
# 
# 

# **Independent Variable (Numerical)**
# 
# There are 4 features that are Numerical: These features have numerical values (ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term)

# In[ ]:


sns.displot(a['ApplicantIncome']);
plt.show()
a['ApplicantIncome'].plot.box(figsize=(14,8))
plt.show()


# It can be inferred that most of the data in the distribution of applicant income is towards left which means it is not normally distributed. The distribution is right-skewed (positive skewness). We will try to make it normal in later sections as algorithms works better if the data is normally distributed.
# 
# 

# In[ ]:


a.boxplot(column='ApplicantIncome',by='Self_Employed')
plt.suptitle("")


# In[ ]:


a.boxplot(column='ApplicantIncome',by='Married')
plt.suptitle("")


# The boxplot confirms the presence of a lot of outliers/extreme values. This can be attributed to the income disparity in the society. Part of this can be driven by the fact that we are looking at people with different education levels. Let us segregate them by Education and Married:

# In[ ]:


plt.subplot(121)
sns.distplot(a['CoapplicantIncome']);

plt.subplot(122)
a['CoapplicantIncome'].plot.box(figsize=(16,5))

plt.show()


# In[ ]:


a['Loan_Amount_Term'].value_counts(normalize=True).plot.bar(title= 'Loan_Amount_Term')


# In[ ]:


a.groupby('Married')['ApplicantIncome'].mean().plot(kind='bar')


# In[ ]:


a.groupby('Gender')['ApplicantIncome'].mean().plot(kind='bar')


# In[ ]:


a.plot()


# In[ ]:


a.boxplot()


# In[ ]:


a.corr()


# In[ ]:


sns.heatmap(a.corr())


# In[ ]:


Marr=a.groupby(by='Married').mean()
Marr.plot(kind='bar')
Edu1=a.groupby(by='Self_Employed').mean()
Edu1.plot(kind='bar')
Edu=a.groupby(by='Education').mean()
Edu.plot(kind='bar')


# In[ ]:


print(len(a[a["CoapplicantIncome"] == 0]))
"Percentage of CoapplicantIncome = 0 is:",  len(a[a["CoapplicantIncome"] == 0])/len(a["CoapplicantIncome"])


# It shows that if coapplicant’s income is less the chances of loan approval are high. But this does not look right. The possible reason behind this may be that most of the applicants don’t have any coapplicant, so the coapplicant income for such applicants is 0 and hence the loan approval is not dependent on it. So we can make a new variable in which we will combine the applicant’s and coapplicant’s income to visualize the combined effect of income on loan approval.

# In[ ]:


sns.heatmap(a.corr())


# Now lets look at the correlation between all the numerical variables. We can use the corr() to compute pairwise correlation of columns, excluding NA/null values using pearson correlation coefficient. Then we will use the heat map to visualize the correlation. Heatmaps visualize data through variations in coloring. The variables with darker color means their correlation is more.

# In[ ]:


matrix=a.corr()
f, ax=plt.subplots(figsize=(17,8))
sns.heatmap(matrix, vmax=1, square=True, cmap="BuPu", annot=True)

matrix


# **Note**: We see that the most correlated variables are
# 
# * (ApplicantIncome - LoanAmount) with correlation coefficient of 0.49
# 
# * (Credit_History - CoapplicantIncome) with correlation coefficient of -0.058
# 
# * LoanAmount is also correlated with CoapplicantIncome with correlation coefficient of 0.15.

# **Outlier Treatment**
# 
# As we saw earlier in univariate analysis, LoanAmount contains outliers so we have to treat them as the presence of outliers affects the distribution of the data. Having outliers in the dataset often has a significant effect on the mean and standard deviation and hence affecting the distribution. We must take steps to remove outliers from our data sets.
# 
# Due to these outliers bulk of the data in the loan amount is at the left and the right tail is longer. This is called right skewness (or positive skewness). One way to remove the skewness is by doing the log transformation. As we take the log transformation, it does not affect the smaller values much, but reduces the larger values. So, we get a distribution similar to normal distribution.

# In[34]:


a.hist()


# In[ ]:


# before log transformation
ax2 = plt.subplot(122)
a['LoanAmount'].hist(bins=20)
ax2.set_title("Test")


# In[ ]:


sns.scatterplot(x='ApplicantIncome',y='CoapplicantIncome',data=b)


# In[ ]:


sns.jointplot(x='ApplicantIncome',y='Credit_History',data=a)


# In[ ]:


sns.violinplot(x='CoapplicantIncome',data=a)


# In[ ]:


sns.violinplot(x='Credit_History',data=a)


# In[ ]:


sns.regplot(x='ApplicantIncome',y='Credit_History',data=a)


# In[ ]:


sns.regplot(x='CoapplicantIncome',y='Credit_History',data=a)


# In[ ]:


sns.boxplot(x='Married',y='ApplicantIncome',data=a)
plt.show()
sns.boxplot(x='Education',y='ApplicantIncome',data=a)
plt.show()
sns.boxplot(x='Gender',y='ApplicantIncome',data=a);
plt.show()


# In[ ]:


sns.pairplot(a)


# **Train dataset**

# In[3]:


#load the test dataset
b=pd.read_csv("train.csv")
b


# In[ ]:


b.describe


# In[ ]:


# show the shape of the dataset i.e. no of rows, no of columns
b.shape


# In[ ]:


b_length=len(b)


# In[ ]:


#take a look at the features (i.e. independent variables) in the test dataset
b.columns


# In[ ]:


b.dtypes


# In[39]:


b.mean()


# In[40]:


b.std()


# **Data Pre-Processing**

# **Missing value imputation**

# In[ ]:


# check for missing values
b.apply(lambda x:sum(x.isnull()),axis=0)


# There are missing values in Gender, Married, Dependents, Self_Employed, LoanAmount, Loan_Amount_Term and Credit_History features. We will treat the missing values in all the features one by one.
# 
# * For numerical variables: imputation using mean or median
# 
# * For categorical variables: imputation using mode
# 
# There are very less missing values in Gender, Married, Dependents, Credit_History and Self_Employed features so we can fill them using the mode of the features. If an independent variable in our dataset has huge amount of missing data e.g. 80% missing values in it, then we would drop the variable from the dataset.

# In[ ]:


# replace missing values with the mode
b['Gender'].fillna(b['Gender'].mode()[0], inplace=True)
b['Married'].fillna(b['Married'].mode()[0], inplace=True)
b['Dependents'].fillna(b['Dependents'].mode()[0], inplace=True)
b['Self_Employed'].fillna(b['Self_Employed'].mode()[0], inplace=True)
b['Credit_History'].fillna(b['Credit_History'].mode()[0], inplace=True)


# In[ ]:


# check whether all the missing values are filled in the Train dataset
b.apply(lambda x:sum(x.isnull()),axis=0)


# In[ ]:


b.describe()


# **Target Variable (Categorical)**
# 
# We will first look at the target variable, i.e., Loan_Status. As it is a categorical variable, let us look at its frequency table and bar plot.

# In[ ]:


b['Credit_History'].value_counts()


# In[ ]:


b['Loan_Amount_Term'].value_counts()


# In[ ]:


b['Self_Employed'].value_counts()


# In[ ]:


b['Dependents'].value_counts()


# In[ ]:


b['Gender'].value_counts()


# In[ ]:


b['Married'].value_counts()


# In[ ]:


# replacing 3+ in Dependents variable with 3 for both train and test set
tra=b['Dependents'].replace('3+', 3, inplace=True)


# In[ ]:


b['Loan_Status'].value_counts().plot.bar(title="Loan_Status")
plt.show()
b['Gender'].value_counts().plot.bar(title="Gender")
plt.show()
b['Married'].value_counts().plot.bar(title="Married")
plt.show()
b['Self_Employed'].value_counts().plot.bar(title="Self_employed")
plt.show()
b['Credit_History'].value_counts().plot.bar(title="Credit_History")
plt.show()


# **Independent Variable (Numerical)**
# 
# There are 4 features that are Numerical: These features have numerical values (ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term)
# 
# It can be inferred that most of the data in the distribution of applicant income is towards left which means it is not normally distributed. The distribution is right-skewed (positive skewness). We will try to make it normal in later sections as algorithms works better if the data is normally distributed.

# In[14]:


# Visualizing CoapplicantIncome
sns.displot(b['CoapplicantIncome'])
plt.show()
b['CoapplicantIncome'].plot.box(figsize=(14,8))
plt.show()


# In[ ]:


b.boxplot(column='ApplicantIncome',by='Self_Employed')
plt.suptitle("")


# The boxplot confirms the presence of a lot of outliers/extreme values. This can be attributed to the income disparity in the society. Part of this can be driven by the fact that we are looking at people with different education levels. Let us segregate them by Education:

# In[ ]:


b.boxplot(column='ApplicantIncome',by='Married')
plt.suptitle("")


# The boxplot confirms the presence of a lot of outliers/extreme values. This can be attributed to the income disparity in the society. Let us segregate them by Married:

# In[ ]:


b.boxplot(column='ApplicantIncome',by='Education')
plt.suptitle("")


# **Categorical Independent Variable vs Target Variable**
# 
# First of all we will find the relation between target variable and categorical independent variables. Let us look at the stacked bar plot now which will give us the proportion of approved and unapproved loans. For example, we want to see whether an applicant's gender will have any effect on approval chances.

# In[24]:


Gender=pd.crosstab(b['Gender'],b['Loan_Status'])
Married=pd.crosstab(b['Married'],b['Loan_Status'])
Dependents=pd.crosstab(b['Dependents'],b['Loan_Status'])
Education=pd.crosstab(b['Education'],b['Loan_Status'])
Self_Employed=pd.crosstab(b['Self_Employed'],b['Loan_Status'])
Married.div(Married.sum(1).astype(float), axis=0).plot(kind="bar",stacked=True,figsize=(4,4))
plt.show()
Dependents.div(Dependents.sum(1).astype(float), axis=0).plot(kind="bar",stacked=True,figsize=(4,4))
plt.show()
Education.div(Education.sum(1).astype(float), axis=0).plot(kind="bar",stacked=True,figsize=(4,4))
plt.show()
Self_Employed.div(Self_Employed.sum(1).astype(float), axis=0).plot(kind="bar",stacked=True,figsize=(4,4))
plt.show()


# From the bar charts above, it can be inferred that:
# 
# * proportion of male and female applicants is more or less same for both approved and unapproved loans
# 
# * proportion of married applicants is higher for the approved loans
# 
# * distribution of applicants with 1 or 3+ dependents is similar across both the categories of Loan_Status
# 
# * there is nothing significant we can infer from Self_Employed vs Loan_Status plot

# In[44]:


sns.countplot(x=b['Gender'])
plt.show()


# In[45]:


sns.countplot(x=b['Dependents'])
plt.show()


# In[46]:


sns.countplot(x=b['Education'])
plt.show()


# In[47]:


sns.countplot(x=b['Property_Area'])
plt.show()


# **Numerical Independent Variable vs Target Variable**
# 
# We will try to find the mean income of people for which the loan has been approved vs the mean income of people for which the loan has not been approved.

# In[ ]:


b.groupby('Married')['ApplicantIncome'].mean().plot(kind='bar')


# In[ ]:


b.groupby('Gender')['ApplicantIncome'].mean().plot(kind='bar')


# In[ ]:


b.plot()


# In[ ]:


b.boxplot()


# In[30]:


plt.pie(b.Property_Area.value_counts(),[0,0,0],labels=['Semi urban','Urban','Rural'])
plt.show()
plt.pie(b.Education.value_counts(),[0,0],labels=['Graduate','Not Graduate'])
plt.show()
plt.pie(b.Gender.value_counts(),[0,0],labels=['Male','Female'])
plt.show()
plt.pie(b.Self_Employed.value_counts(),[0,0],labels=['No','Yes'])


# In[ ]:


b.corr()


# In[ ]:


sns.heatmap(b.corr())


# In[ ]:


print(len(b[b["ApplicantIncome"] == 0]))
"Percentage of ApplicantIncome = 0 is:",  len(b[b["ApplicantIncome"] == 0])/len(b["ApplicantIncome"])


# In[ ]:


print(len(b[b["CoapplicantIncome"] == 0]))
"Percentage of CoapplicantIncome = 0 is:",  len(b[b["CoapplicantIncome"] == 0])/len(b["CoapplicantIncome"])


# In[ ]:


# calculate and visualize correlation matrix
matrix=b.corr()
f, ax=plt.subplots(figsize=(17,8))
sns.heatmap(matrix, vmax=1, square=True, cmap="BuPu", annot=True)

matrix


# **Note :**We see that the most correlated variables are
# 
# * (ApplicantIncome - LoanAmount) with correlation coefficient of 0.57
# 
# * (Credit_History - Loan_Status) with correlation coefficient of 0.56
# 
# * LoanAmount is also correlated with CoapplicantIncome with correlation coefficient of 0.19.

# In[33]:


b.hist()


# In[ ]:


ax1 = plt.subplot(121)
b['LoanAmount'].hist(bins=20, figsize=(12,4))
ax1.set_title("Train")


# In[ ]:


sns.scatterplot(x='ApplicantIncome', y='CoapplicantIncome', data=b)


# In[ ]:


sns.jointplot(x='ApplicantIncome', y='Credit_History', data=b)


# In[ ]:


sns.violinplot(x='ApplicantIncome', data=b)


# In[ ]:


sns.regplot(x='ApplicantIncome', y='Credit_History', data=b)


# In[ ]:


sns.boxplot(x='Married',y='ApplicantIncome',data=b,width=0.3)
plt.show()
sns.boxplot(x='Education',y='ApplicantIncome',data=b,width=0.3)
plt.show()
sns.boxplot(x='Gender',y='ApplicantIncome',data=b,width=0.3);
plt.show()


# In[ ]:


sns.swarmplot(x='Education', y='CoapplicantIncome', data=b)


# In[ ]:


sns.lmplot(x='ApplicantIncome',y='Credit_History',data=b)


# **Model Buliding : Part 1**
# 
# Let us make our first model to predict the target variable. We will start with Logistic Regression which is used for predicting binary outcome.
# 
# * Logistic Regression is a classification algorithm. It is used to predict a binary outcome (1 / 0, Yes / No, True / False) given a set of independent variables.
# 
# * Logistic regression is an estimation of Logit function. Logit function is simply a log of odds in favor of the event.
# 
# * This function creates a s-shaped curve with the probability estimate, which is very similar to the required step wise function

# **Predicting output Y column from Loan Prediction of Train Dataset**

# Lets drop the Loan_ID variable as it do not have any effect on the loan status. We will do the same changes to the test dataset which we did for the training dataset.

# In[5]:


# replacing Y and N in Loan_Status variable with 1 and 0 respectively
b['Loan_Status'].replace('N', 0, inplace=True)
b['Loan_Status'].replace('Y', 1, inplace=True)


# In[6]:


# drop Loan_ID 
train = b.drop('Loan_ID', axis=1)


# In[7]:


X = b.drop('Loan_Status', 1)
y = b.Loan_Status


# In[8]:


train.shape


# Let us understand the process of dummies first:
# 
# * Consider the “Gender” variable. It has two classes, Male and Female.
# 
# * As logistic regression takes only the numerical values as input, we have to change male and female into numerical value.
# 
# * Once we apply dummies to this variable, it will convert the “Gender” variable into two variables(Gender_Male and Gender_Female), one for each class, i.e. Male and Female.
# 
# * Gender_Male will have a value of 0 if the gender is Female and a value of 1 if the gender is Male.
# 
# We can use pandas **get_dummies** function to convert categorical variable into dummy/indicator variables, it will only convert "object" type and will not affect numerical type.

# In[9]:


# adding dummies to the dataset
X = pd.get_dummies(X)
train = pd.get_dummies(b)


# In[10]:


X.head()


# In[11]:


y

