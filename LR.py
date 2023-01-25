from sklearn.model_selection    import GridSearchCV
from sklearn.linear_model       import LogisticRegression
from sklearn.metrics            import accuracy_score
from sklearn.metrics            import confusion_matrix
from sklearn.metrics            import f1_score
from sklearn.metrics            import classification_report,roc_auc_score
import pandas as pd
import argparse
import pickle
import joblib 
import time

temp =  argparse.ArgumentParser()

# read data
X_train= pd.read_csv('data/X_train.csv')
y_train= pd.read_csv('data/y_train.csv')
X_test = pd.read_csv('data/X_test.csv')
y_test = pd.read_csv('data/y_test.csv')
y_train = y_train['HeartDisease']
y_test = y_test['HeartDisease']

# Logistic Regression
# search for optimun parameters using gridsearch
params = {'solver': ['newton-cg', 'lbfgs', 'liblinear'],
          'penalty':['l2','l2'],
          'C':[0.01,0.1,1,10,100],
          'class_weight':['balanced',None]}
logistic_clf = GridSearchCV(LogisticRegression(),param_grid=params,cv=10,n_jobs=-1)

# record start time
start_time = time.time()

#train the classifier
logistic_clf.fit(X_train,y_train)
logistic_clf.best_params_

# record end time 
end_time = time.time()
time_cost = start_time - end_time

# save the model
with open('pkl/lr_clf.pkl','wb') as f:
    pickle.dump(logistic_clf , f)

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
with open('pkl/lr_result.pkl','wb') as f:
    pickle.dump((log_accuracy, logistic_f1, logistic_auc, time_cost), f)