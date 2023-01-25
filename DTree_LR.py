from sklearn.model_selection    import GridSearchCV
from sklearn.tree               import DecisionTreeClassifier
from sklearn.linear_model       import LogisticRegression
from sklearn.metrics            import accuracy_score
from sklearn.metrics            import confusion_matrix
from sklearn.metrics            import f1_score
from sklearn.metrics            import classification_report
from sklearn.metrics            import classification_report,roc_auc_score
import pandas as pd
import argparse
import joblib 
import pickle
import time

# Parser for command-line options, arguments and sub-commands
temp =  argparse.ArgumentParser()

# read data
X_train= pd.read_csv('data/X_train.csv')
y_train= pd.read_csv('data/y_train.csv')
X_test = pd.read_csv('data/X_test.csv')
y_test = pd.read_csv('data/y_test.csv')

# Decision Tree Classifier
dtree= DecisionTreeClassifier(random_state=7)
# grid search for optimum parameters
params = {'criterion': ['gini', 'entropy', 'log_loss'],
          'splitter': ['best', 'random'],
          'max_features': ['auto', 'sqrt', 'log2'],
          'min_samples_split': [2,3,4,5,6,7,8,9,10,11,12,13,14,15], 
          'min_samples_leaf':[1,2,3,4,5,6,7,8,9,10,11]}
dtree_clf = GridSearchCV(dtree, param_grid=params, n_jobs=-1)

# Logistic Regression
# search for optimun parameters using gridsearch
params = {'solver': ['newton-cg', 'lbfgs', 'liblinear'],
          'penalty':['l1','l2'],
          'C':[0.01,0.1,1,10,100],
          'class_weight':['balanced',None]}
logistic_clf = GridSearchCV(LogisticRegression(),param_grid=params,cv=10,n_jobs=-1)


# record start time
start_time = time.time()

# train the DTree and LR
dtree_clf.fit(X_train,y_train)
dtree_clf.best_params_ 
logistic_clf.fit(X_train,y_train)
logistic_clf.best_params_

# record end time 
end_time = time.time()
time_cost = start_time - end_time

# save the model
with open('pkl/dtree2_clf.pkl','wb') as f:
    pickle.dump(dtree_clf, f)
with open("pkl/lr2_clf.pkl", "wb") as f:
    pickle.dump(logistic_clf, f)



# predictions

#### dtree
dtree_predict = dtree_clf.predict(X_test)
#accuracy
dtree_accuracy = accuracy_score(y_test,dtree_predict)
print(f"Using Decision Trees we get an accuracy of {round(dtree_accuracy*100,2)}%")
cm=confusion_matrix(y_test,dtree_predict)
print(classification_report(y_test,dtree_predict))
dtree_f1 = f1_score(y_test, dtree_predict)
print(f'The f1 score Descision trees is {round(dtree_f1*100,2)}%')
# ROC curve and AUC 
probs = dtree_clf.predict_proba(X_test)
# keep probabilities for the positive outcome only
probs = probs[:, 1]
# calculate AUC
dtree_auc = roc_auc_score(y_test, probs)


#### LR
#make predictions
logistic_predict = logistic_clf.predict(X_test)
log_accuracy = accuracy_score(y_test,logistic_predict)
print(f"Using logistic regression we get an accuracy of {round(log_accuracy*100,2)}%")
cm=confusion_matrix(y_test,logistic_predict)
print(classification_report(y_test,logistic_predict))
logistic_f1 = f1_score(y_test, logistic_predict)
print(f'The f1 score for logistic regression is {round(logistic_f1*100,2)}%')
logistic_auc = roc_auc_score(y_test, logistic_predict)

# save the result
with open('pkl/dtree_lr_result.pkl','wb') as f:
    pickle.dump((dtree_accuracy, dtree_f1, dtree_auc, log_accuracy, logistic_f1, logistic_auc, time_cost), f)