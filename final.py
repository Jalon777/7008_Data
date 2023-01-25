import pandas as pd
import pickle
import time

# load the model
with open("pkl/dtree_clf.pkl", "rb") as f:
    dtree = pickle.load(f)
    
with open("pkl/lr_clf.pkl", "rb") as f:
    lr = pickle.load(f)

with open("pkl/dtree2_clf.pkl", "rb") as f:
    dtree2 = pickle.load(f)

with open("pkl/lr2_clf.pkl", "rb") as f:
    lr2 = pickle.load(f)


# load the prediction result of model1, model2 and model3
with open("pkl/dtree_result.pkl", "rb") as f:
    dtree_accuracy, dtree_f1, dtree_auc, dtree_time_cost = pickle.load(f)
    
with open("pkl/lr_result.pkl", "rb") as f:
    log_accuracy, logistic_f1, logistic_auc, lr_time_cost = pickle.load(f)

with open("pkl/dtree_lr_result.pkl", "rb") as f:
    dtree2_accuracy, dtree2_f1, dtree2_auc, log2_accuracy, logistic2_f1, logistic2_auc, time2_cost = pickle.load(f)
# dtree2_accuracy, dtree2_f1, dtree2_auc, log2_accuracy, logistic2_f1, logistic2_auc, time2_cost = dtree_accuracy, dtree_f1, dtree_auc, log_accuracy, logistic_f1, logistic_auc, lr_time_cost


# compare the model
print("Time cost of dtree:   ", dtree_time_cost )
print("Time cost of lr:      ", lr_time_cost )
print("Time cost of dtree+lr:", time2_cost )

print("dtree accuracy:             ", dtree_accuracy, "\n dtree accuracy in dtree+lr:", dtree2_accuracy)
print("dtree f1:                  ", dtree_f1, "\ndtree f1 in dtree+lr:      ", dtree2_f1)
print("dtree auc:                 ", dtree_auc, "\ndtree auc in dtree+lr:     ", dtree2_auc)
print("lr accuracy:               ", log_accuracy, "\nlr accuracy in dtree+lr:   ", log2_accuracy)
print("lr f1:                     ", logistic_f1, "\nlr f1 in dtree+lr:         ", logistic2_f1)
print("lr auc:                    ", logistic_auc, "\nlr auc in dtree+lr:        ", logistic2_auc)

# write in final_result.txt
with open("final_result.txt", "w") as f:
    f.write("Time cost of dtree: " + str(dtree_time_cost) + "\n")
    f.write("Time cost of lr: " + str(lr_time_cost) + "\n")
    f.write("Time cost of dtree+lr: " + str(time2_cost) + "\n" + "\n")

    f.write("dtree accuracy: " + str(dtree_accuracy) + "\n dtree accuracy in dtree+lr:" + str(dtree2_accuracy) + "\n")
    f.write("dtree f1: " + str(dtree_f1) + "\n dtree f1 in dtree+lr: " + str(dtree2_f1) + "\n")
    f.write("dtree auc: " + str(dtree_auc) + "\n dtree auc in dtree+lr: " + str(dtree2_auc) + "\n")
    f.write("lr accuracy: " + str(log_accuracy) + "\n lr accuracy in dtree+lr: " + str(log2_accuracy) + "\n")
    f.write("lr f1: " + str(logistic_f1) + "\n lr f1 in dtree+lr: " + str(logistic2_f1) + "\n")
    f.write("lr auc: " + str(logistic_auc) + "\n lr auc in dtree+lr: " + str(logistic2_auc) + "\n")

    best_model = max(dtree_accuracy, log_accuracy, dtree2_accuracy, log2_accuracy)
    if best_model == dtree_accuracy or best_model == dtree2_accuracy:
        f.write("The best model is Decision Tree, with accuracy: " + str(best_model) + "\n")
    elif best_model == log_accuracy or best_model == log2_accuracy:
        f.write("The best model is Logistic Regression, with accuracy: " + str(best_model) + "\n")