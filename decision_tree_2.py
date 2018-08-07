
# coding: utf-8

# In[3]:


import pandas as pd


# In[4]:


import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.stats import spearmanr
from pylab import  rcParams
from pandas import Series, DataFrame


# In[5]:


import seaborn as sb
import matplotlib.pyplot as plt
import sklearn
from sklearn.preprocessing import scale
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import preprocessing
#from matlplotlib import rcParams
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('pylab', 'inline')


# In[6]:


rcParams['figure.figsize'] = 5,4
sb.set_style('whitegrid')
df =pd.read_csv('copy.csv')
df.head()


# In[13]:


from preamble import *
import mglearn


# In[14]:


from sklearn.tree import DecisionTreeClassifier


# In[26]:





# In[30]:


y = df['churn']
z=df.state
h=df.phone_number
y = pd.get_dummies(y)[' True.']
cat_vars=['international_plan','voice_mail','day_minute','day_charge','eve_minute','eve_charge','intl_call','customer_service']
noncat=[i for i in df.columns if i not in cat_vars]


X =df.drop(noncat,axis=1)


# In[31]:


X['international_plan'] = pd.get_dummies(X.international_plan)[' yes']
X['voice_mail'] = pd.get_dummies(X.voice_mail)[' yes']
#pd.get_dummies(X.voice_mail)


# In[32]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.5,random_state=42)


# In[33]:


tree = DecisionTreeClassifier(random_state=0)
tree.fit(X_train,y_train)
print('Accuracy on the training subset: {:.3f}'.format(tree.score(X_train,y_train)))


# In[34]:


tree = DecisionTreeClassifier(max_depth=4,random_state=0)
tree.fit(X_train,y_train)
print('Accuracy on the training subset: {:.3f}'.format(tree.score(X_train,y_train)))
print('Accuracy on the test subset: {:.3f}'.format(tree.score(X_test,y_test)))


# In[35]:


import graphviz
from sklearn.tree import export_graphviz


export_graphviz(tree,out_file='churn2.dot',class_names=['0','1'],feature_names=cat_vars,impurity=False,filled=True)


# In[36]:


import sys


# In[37]:


from sklearn import model_selection
from sklearn.model_selection import cross_val_score
kfold = model_selection.KFold(n_splits=10, random_state=7)
modelCV = DecisionTreeClassifier()
scoring = 'accuracy'
results = model_selection.cross_val_score(modelCV, X_train, y_train, cv=kfold, scoring=scoring)
print("10-fold cross validation average accuracy: %.3f" % (results.mean()))


# In[38]:


print('feature importance:{}'.format(tree.feature_importances_))
type(tree.feature_importances_)


# In[39]:


n_features=X.shape[1]
plt.barh(range(n_features),tree.feature_importances_,align='center')
plt.yticks(np.arange(n_features),cat_vars)
plt.show()


# In[40]:


y_pred_prob = tree.predict_proba(X_test)[:,1]


# In[41]:


# histogram of predicted probabilities
plt.hist(y_pred_prob, bins=8)
plt.xlim(0, 1)
plt.title('Histogram of predicted probabilities')
plt.xlabel('Predicted probability of diabetes')
plt.ylabel('Frequency')


# In[42]:


fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_prob)
plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title('ROC curve for diabetes classifier')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)


# In[43]:


print(metrics.roc_auc_score(y_test, y_pred_prob))

