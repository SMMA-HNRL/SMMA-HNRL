import csv
import lightgbm as lgb
from sklearn import metrics
import matplotlib.pylab as plt
from sklearn.metrics import auc
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import GradientBoostingClassifier,AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from numpy import interp
from sklearn.linear_model import LogisticRegression
from IPython.core.display import display
import warnings
from mlxtend.classifier import StackingClassifier
warnings.filterwarnings("ignore")
csvFile = open("32+32.csv", "r", encoding='UTF-8-sig')
csvFile1= open("target.csv","r",encoding='UTF-8-sig')
reader = csv.reader(csvFile)
reader1=csv.reader(csvFile1)
x=[]
y=[]
for item in reader:
    x.append(item)
x=np.array(x)
x = x.astype(float)
for item in reader1:
    y.extend(item)
y = [ int(x) for x in y ]
y=np.array(y)
#StratifiedKFold分层采样，用于交叉验证；StratifiedKFold方法是根据标签中不同类别占比来进行拆分数据的。
#将训练/测试数据集划分n_splits个互斥子集，每次只用其中一个子集当做测试集；shuffle为True代表每次划分的结果不一样，表示经过洗牌随机取样的
kf=StratifiedKFold(n_splits=10,shuffle=True,random_state=1)
tprs_nbm=[]
aucs_nbm=[]
acc_nbm = []
F1score_nbm = []
precision_nbm = []
recall_nbm = []
AUPR_nbm=[]
mean_fpr_nbm=np.linspace(0,1,100)

tprs_knn=[]
aucs_knn=[]
acc_knn = []
F1score_knn = []
precision_knn = []
recall_knn = []
AUPR_knn=[]
mean_fpr_knn=np.linspace(0,1,100)


tprs_ada=[]
aucs_ada=[]
acc_ada = []
F1score_ada = []
precision_ada = []
recall_ada = []
AUPR_ada=[]
mean_fpr_ada=np.linspace(0, 1, 100)

tprs_lr=[]
aucs_lr=[]
acc_lr = []
F1score_lr = []
precision_lr = []
recall_lr = []
AUPR_lr=[]
mean_fpr_lr=np.linspace(0,1,100)

tprs_lightgbm=[]
aucs_lightgbm=[]
acc_lightgbm = []
F1score_lightgbm = []
precision_lightgbm = []
recall_lightgbm = []
AUPR_lightgbm=[]
mean_fpr_lightgbm=np.linspace(0,1,100)

auc_all = []
acc_all = []
F1score_all = []
precision_all = []
recall_all = []
AUPR_all=[]
data = x
label = y
i=0
for train_index,test_index in kf.split(data,label):#循环进行计算
#划分训练集和测试集
    X_train,X_test = data[train_index], data[test_index]
    Y_train, Y_test = label[train_index], label[test_index]
###################——————————Naive Bayes——————————####################

    model_nbm = BernoulliNB()
    model_nbm.fit(X_train,Y_train)
    y_pred_nbm = model_nbm.predict(X_test)
    y_pred_proba_nbm = model_nbm.predict_proba(X_test)[:, 1]
    fpr_nbm,tpr_nbm,thresholds_nbm= metrics.roc_curve(Y_test, y_pred_proba_nbm)
    tprs_nbm.append(interp(mean_fpr_nbm,fpr_nbm,tpr_nbm))
    tprs_nbm[-1][0]=0.0
    roc_auc_nbm = metrics.roc_auc_score(Y_test, y_pred_proba_nbm)
    aucs_nbm.append(roc_auc_nbm)
    acc_nbm.append(metrics.accuracy_score(Y_test,y_pred_nbm))
    recall_nbm.append(metrics.recall_score(Y_test, y_pred_nbm))
    F1score_nbm.append(metrics.f1_score(Y_test, y_pred_nbm))
    precision_nbm.append(metrics.precision_score(Y_test, y_pred_nbm))
    precision, recall, _thresholds = metrics.precision_recall_curve(Y_test,y_pred_proba_nbm)
    auPR = metrics.auc(recall,precision)
    AUPR_nbm.append(auPR)

 ###################——————————KNN——————————####################

    model_knn = KNeighborsClassifier()
    model_knn.fit(X_train,Y_train)
    y_pred_knn = model_knn.predict(X_test)
    y_pred_proba_knn = model_knn.predict_proba(X_test)[:, 1]
    fpr_knn,tpr_knn,thresholds_knn= metrics.roc_curve(Y_test, y_pred_proba_knn)
    tprs_knn.append(interp(mean_fpr_knn,fpr_knn,tpr_knn))
    tprs_knn[-1][0]=0.0
    roc_auc_knn = metrics.roc_auc_score(Y_test, y_pred_proba_knn)
    aucs_knn.append(roc_auc_knn)
    acc_knn.append(metrics.accuracy_score(Y_test,y_pred_knn))
    recall_knn.append(metrics.recall_score(Y_test, y_pred_knn))
    F1score_knn.append(metrics.f1_score(Y_test, y_pred_knn))
    precision_knn.append(metrics.precision_score(Y_test, y_pred_knn))
    precision, recall, _thresholds = metrics.precision_recall_curve(Y_test,y_pred_proba_knn)
    auPR = metrics.auc(recall,precision)
    AUPR_knn.append(auPR)


###################——————————Logistic Regression——————————####################
    model_lr = LogisticRegression(penalty="l1",solver="liblinear",C=0.01)
    model_lr.fit(X_train,Y_train)
    y_pred_lr = model_lr.predict(X_test)
    y_pred_proba_lr = model_lr.predict_proba(X_test)[:, 1]
    fpr_lr,tpr_lr,thresholds_lr= metrics.roc_curve(Y_test, y_pred_proba_lr)
    tprs_lr.append(interp(mean_fpr_lr,fpr_lr,tpr_lr))
    tprs_lr[-1][0]=0.0
    roc_auc_lr = metrics.roc_auc_score(Y_test, y_pred_proba_lr)
    aucs_lr.append(roc_auc_lr)
    acc_lr.append(metrics.accuracy_score(Y_test,y_pred_lr))
    recall_lr.append(metrics.recall_score(Y_test, y_pred_lr))
    F1score_lr.append(metrics.f1_score(Y_test, y_pred_lr))
    precision_lr.append(metrics.precision_score(Y_test, y_pred_lr))
    precision, recall, _thresholds = metrics.precision_recall_curve(Y_test, y_pred_lr)
    auPR = metrics.auc(recall, precision)
    AUPR_ada.append(auPR)

##############——————————AdaBOOST——————————####################
    model_ada = AdaBoostClassifier(learning_rate=0.01, n_estimators=350)
    model_ada.fit(X_train, Y_train)
    y_pred_ada = model_ada.predict(X_test)
    y_pred_proba_ada = model_ada.predict_proba(X_test)[:, 1]
    fpr_ada, tpr_ada, thresholds_ada = metrics.roc_curve(Y_test, y_pred_proba_ada)
    tprs_ada.append(interp(mean_fpr_ada, fpr_ada, tpr_ada))
    tprs_ada[-1][0] = 0.0
    roc_auc_ada = metrics.roc_auc_score(Y_test, y_pred_proba_ada)
    aucs_ada.append(roc_auc_ada)
    acc_ada.append(metrics.accuracy_score(Y_test, y_pred_ada))
    recall_ada.append(metrics.recall_score(Y_test, y_pred_ada))
    F1score_ada.append(metrics.f1_score(Y_test, y_pred_ada))
    precision_ada.append(metrics.precision_score(Y_test, y_pred_ada))
    precision, recall, _thresholds = metrics.precision_recall_curve(Y_test, y_pred_proba_ada)
    auPR = metrics.auc(recall, precision)
    AUPR_lr.append(auPR)

##################------------lightgbm-------#########################
    model_lightgbm = lgb.LGBMClassifier(learning_rate=0.01, n_estimators=350,num_leaves=8,max_depth=4)
    model_lightgbm.fit(X_train, Y_train)
    y_pred_lightgbm = model_lightgbm.predict(X_test)
    y_pred_proba_lightgbm = model_lightgbm.predict_proba(X_test)[:, 1]
    fpr_lightgbm, tpr_lightgbm, thresholds_lightgbm = metrics.roc_curve(Y_test, y_pred_proba_lightgbm)
    tprs_lightgbm.append(interp(mean_fpr_lightgbm, fpr_lightgbm, tpr_lightgbm))
    tprs_lightgbm[-1][0] = 0.0
    roc_auc_lightgbm = metrics.roc_auc_score(Y_test, y_pred_proba_lightgbm)
    aucs_lightgbm.append(roc_auc_lightgbm)
    acc_lightgbm.append(metrics.accuracy_score(Y_test, y_pred_lightgbm))
    recall_lightgbm.append(metrics.recall_score(Y_test, y_pred_lightgbm))
    F1score_lightgbm.append(metrics.f1_score(Y_test, y_pred_lightgbm))
    precision_lightgbm.append(metrics.precision_score(Y_test, y_pred_lightgbm))
    precision, recall, _thresholds = metrics.precision_recall_curve(Y_test, y_pred_proba_lightgbm)
    auPR = metrics.auc(recall, precision)
    AUPR_lightgbm.append(auPR)

mean_tpr_nbm=np.mean(tprs_nbm,axis=0)
mean_tpr_nbm[-1]=1.0
mean_auc_nbm=auc(mean_fpr_nbm,mean_tpr_nbm)#计算平均AUC值
std_auc_nbm=np.std(tprs_nbm,axis=0)

mean_tpr_knn=np.mean(tprs_knn,axis=0)
mean_tpr_knn[-1]=1.0
mean_auc_knn=auc(mean_fpr_knn,mean_tpr_knn)#计算平均AUC值
std_auc_knn=np.std(tprs_knn,axis=0)

mean_tpr_lr=np.mean(tprs_lr,axis=0)
mean_tpr_lr[-1]=1.0
mean_auc_lr=auc(mean_fpr_lr,mean_tpr_lr)#计算平均AUC值
std_auc_lr=np.std(tprs_lr,axis=0)

mean_tpr_ada=np.mean(tprs_ada, axis=0)
mean_tpr_ada[-1]=1.0
mean_auc_ada=auc(mean_fpr_ada, mean_tpr_ada)#计算平均AUC值
std_auc_ada=np.std(tprs_ada, axis=0)

mean_tpr_lightgbm=np.mean(tprs_lightgbm,axis=0)
mean_tpr_lightgbm[-1]=1.0
mean_auc_lightgbm=auc(mean_fpr_lightgbm,mean_tpr_lightgbm)#计算平均AUC值
std_auc_lightgbm=np.std(tprs_lightgbm,axis=0)

auc_all.append(mean_auc_knn)
auc_all.append(mean_auc_nbm)
auc_all.append(mean_auc_lr)
auc_all.append(mean_auc_ada)
auc_all.append(mean_auc_lightgbm)
acc_all.append(np.mean(acc_knn))
acc_all.append(np.mean(acc_nbm))
acc_all.append(np.mean(acc_lr))
acc_all.append(np.mean(acc_ada))
acc_all.append(np.mean(acc_lightgbm))



recall_all.append(np.mean(recall_knn))
recall_all.append(np.mean(recall_nbm))
recall_all.append(np.mean(recall_lr))
recall_all.append(np.mean(recall_ada))
recall_all.append(np.mean(recall_lightgbm))




F1score_all.append(np.mean(F1score_knn))
F1score_all.append(np.mean(F1score_nbm))
F1score_all.append(np.mean(F1score_lr))
F1score_all.append(np.mean(F1score_ada))
F1score_all.append(np.mean(F1score_lightgbm))



precision_all.append(np.mean(precision_knn))
precision_all.append(np.mean(precision_nbm))
precision_all.append(np.mean(precision_lr))
precision_all.append(np.mean(precision_ada))
precision_all.append(np.mean(precision_lightgbm))

AUPR_all.append(np.mean(AUPR_knn))
AUPR_all.append(np.mean(AUPR_nbm))
AUPR_all.append(np.mean(AUPR_ada))
AUPR_all.append(np.mean(AUPR_lr))
AUPR_all.append(np.mean(AUPR_lightgbm))

result_all = pd.DataFrame(index=['KNN','Naive Bayes','Logistic Regression','AdaBoost','lightgbm'])



result_all['Recall'] = recall_all
result_all['Precision'] = precision_all
result_all['Acc'] = acc_all
result_all['F1-Score'] = F1score_all
result_all['Auc'] = auc_all
result_all['AUPR'] = AUPR_all
result_all.to_csv('six_classifier_score.csv',sep=',',index=True,header=True)

plt.plot(mean_fpr_nbm,mean_tpr_nbm,color='darkorange',label=r'Naive Bayes (area=%0.4f)'%mean_auc_nbm)
plt.plot(mean_fpr_knn,mean_tpr_knn,color='red',label=r'KNN (area=%0.4f)'%mean_auc_knn)
plt.plot(mean_fpr_lr,mean_tpr_lr,color='green',label=r'Logistic Regression (area=%0.4f)'%mean_auc_lr)
plt.plot(mean_fpr_ada, mean_tpr_ada, color='turquoise', label=r'AdaBoost (area=%0.4f)' % mean_auc_ada)
plt.plot(mean_fpr_lightgbm,mean_tpr_lightgbm,color='navy',label=r'LightGBM (area=%0.4f)'%mean_auc_lightgbm)

plt.xlim([0,1])
plt.ylim([0,1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC',fontsize='x-large',fontweight='bold')
plt.legend(loc='lower right')
plt.savefig('train-ROC.tif')
plt.show()

