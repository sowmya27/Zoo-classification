__author__ = 'annapurnaannadatha'


#############################################################################
##                               README                                    ##
##                                                                         ##
##     This program classifies animals into 7 classes using SVM (Support   ##
##     Vector machine) technique  and compares with Naive Bayes.           ##
##     This problem is considered to be a multiclass classification.       ##
##     The classification is measured using accuracy, and AUC value.       ##
##     The program also displays roc curve and graphs for all features     ##
##                                                                         ##
##                                                                         ##
#############################################################################

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split,KFold,cross_val_score
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")
import warnings  # to ignore all the deprecated warnings
warnings.filterwarnings("ignore")


# Reading data from text file with headers as feature names
zoo_df = pd.read_csv("zoo_data.txt", sep=',',names = ["animalname","hair","feathers","eggs","milk","airborne","aquatic",\
                                                   "predator","toothed","backbone","breathes","venomous","fins","legs",\
                                                   "tail","domestic","catsize","type"])

#check data size
print("Data size -- Rows:",zoo_df.shape[0]," Columns:",zoo_df.shape[1])


#seperating features and class type
zoo_data = zoo_df[["hair","feathers","eggs","milk","airborne","aquatic",\
                   "predator","toothed","backbone","breathes","venomous","fins","legs",\
                   "tail","domestic","catsize"]].values
zoo_target = zoo_df[["type"]].values


#initial feature visualization
zoo_df.plot(kind='density', subplots=True, layout=(4,5), figsize=(13,20), sharex=False, sharey=False)
plt.show()

#split into train and test data
validation_size = 0.20
X_train, X_validation, Y_train, Y_validation = train_test_split(zoo_data, zoo_target,test_size=validation_size, random_state=7)
Y_train = np.array(Y_train.ravel()).astype(int)

# just to make sure...preprocessing and scaling of the data
X_train = preprocessing.scale(X_train)
X_validation = preprocessing.scale(X_validation)
hash_format = '#'*20

#classification
def classify_result(name,model):
    model.fit(X_train,Y_train)
    predictscore= model.predict(X_validation)
    conf_mat = confusion_matrix(Y_validation,predictscore)
    report = classification_report(Y_validation,predictscore)
    score = accuracy_score(Y_validation,predictscore)
    print(hash_format+name+hash_format)
    print("Confusion Matrix")
    print(conf_mat)
    print(report)
    print("Accuracy Score:",score)
    print()

#comparision of classification results
def compare_classification(result1,name1,result2,name2):
    fig = plt.figure()
    plt.title('Comparison')
    ax = fig.add_subplot(111)
    plot_result =[result1,result2]
    plt.boxplot(plot_result)
    ax.set_xticklabels([name1,name2])
    plt.show()

#cross validatdation
def cross_valididate(model):
    num_folds = 10
    num_instances = len(X_train)
    k_fold = KFold(n=num_instances, n_folds= num_folds, random_state=7)
    cv_score = cross_val_score(model, X_train, Y_train, cv=k_fold, scoring='accuracy')
    print("Cross Validation: %0.3f ,%0.3f" %(cv_score.mean(), cv_score.std()))
    return cv_score

#compute and plot ROC
def compute_roc_auc():
    global zoo_target
    zoo_target = label_binarize(zoo_target, classes=[1,2,3,5,5,6,7])
    n_classes = zoo_target.shape[1]
    validation_size = 0.20
    seed = 7
    X_train, X_validation, Y_train, Y_validation = train_test_split(zoo_data, zoo_target,test_size=validation_size, random_state=seed)
    model=OneVsRestClassifier(SVC(kernel="rbf", C=3.00))
    # To compute and plot ROC curve
    y_score = model.fit(X_train,Y_train).decision_function(X_validation)
    # Compute ROC curve and AUC for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(Y_validation[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    # Plot of a ROC curve for a specific class
    plt.figure()
    plt.plot(fpr[6], tpr[6], color='darkorange',\
             lw=4, label='ROC curve (area =%0.2f )' % roc_auc[6])
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('SVM-Receiver operating characteristic ')
    plt.legend(loc="lower right")
    plt.show()
    print(hash_format+" AUC Value "+hash_format)
    print("AUC value:",roc_auc[6])


def main():
    # try different models
    model1 = GaussianNB()
    classify_result(" Naive Bayes ",model1)
    result1 = cross_valididate(model1)
    model2 = SVC(kernel="rbf", C=3.00)
    result2 = cross_valididate(model2)
    classify_result(" SVM RBF Kernel ",model2)
    compare_classification(result1,"Naive Bayes",result2,"SVM RBF Kernel")
    compute_roc_auc()


if __name__=="__main__":
    main()




