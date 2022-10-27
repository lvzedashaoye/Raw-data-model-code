# encoding='ISO-8859-
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from sklearn.decomposition import PCA
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

os.chdir(r'C:\Users\ruhua\Downloads')
plt.rcParams['font.sans-serif'] = ['Simhei']

df = pd.read_excel('444原始分数.xlsx')
feature = df[
    ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q7', 'Q10', 'Q12', 'Q13', 'Q14', 'Q15', 'Q16', 'Q17', 'Q18', 'Q19', 'Q20', 'Q21',
     'Q22', 'Q23', 'Q24', 'Q25']]
print(feature)
target = df['kind']
print(target)

'''模型一，随机森林'''
pca = PCA(n_components=21)
data1 = pca.fit_transform(feature)

X_train, X_test, Y_train, Y_test = train_test_split(data1, target, random_state=16)
biaozhun=StandardScaler()
X_train=biaozhun.fit_transform(X_train)
X_test=biaozhun.fit_transform(X_test)
k1=RandomForestClassifier(n_estimators=115,max_depth=2,random_state=33)
k1.fit(X_train, Y_train)
fenshu=k1.score(X_test, Y_test)
print(fenshu)
print('随机森林混淆矩阵报告')
YY_predict=k1.predict(X_test)
from sklearn.metrics import classification_report
report1=classification_report(Y_test,YY_predict,labels=[1,2],target_names=['正常','抑郁'])
print(report1)

'''模型二、支持向量机'''
pca=PCA(n_components=17)
data2=pca.fit_transform(feature)

X_train2,X_test2,Y_train2,Y_test2 = train_test_split(data2, target,random_state=16)
biaozhun=StandardScaler()
X_train2=biaozhun.fit_transform(X_train2)
X_test2=biaozhun.fit_transform(X_test2)
k2=SVC(C=7,gamma=0.006)
k2.fit(X_train2,Y_train2)
fenshu2=k2.score(X_test2,Y_test2)
print(fenshu2)
print('支持向量机混淆矩阵报告')
Y2_predict=k2.predict(X_test2)

report2=classification_report(Y_test2,Y2_predict,labels=[1,2],target_names=['正常','抑郁'])
print(report2)
'''模型三、逻辑回归'''
pca=PCA(n_components=19)
data3=pca.fit_transform(feature)

X_train3,X_test3,Y_train3,Y_test3 = train_test_split(data3, target,random_state=16)
biaozhun=StandardScaler()
X_train3=biaozhun.fit_transform(X_train3)
X_test3=biaozhun.fit_transform(X_test3)
k3=LogisticRegression(class_weight='balanced',multi_class='multinomial',penalty='l2',solver='lbfgs')
k3.fit(X_train3,Y_train3)
fenshu3=k3.score(X_test3,Y_test3)
print(fenshu3)
print('逻辑回归混淆矩阵报告')
Y3_predict=k3.predict(X_test3)

report3=classification_report(Y_test3,Y3_predict,labels=[1,2],target_names=['正常','抑郁'])
print(report3)
'''模型四，神经网络模型'''
pca=PCA(n_components=19)
data4=pca.fit_transform(feature)

X_train4,X_test4,Y_train4,Y_test4 = train_test_split(data4, target,random_state=16)
biaozhun=StandardScaler()
X_train4=biaozhun.fit_transform(X_train4)
X_test4=biaozhun.fit_transform(X_test4)
k4=MLPClassifier(activation='logistic',alpha=0.01,hidden_layer_sizes= [700, 700],solver= 'adam'  )
k4.fit(X_train4,Y_train4)
fenshu4=k4.score(X_test4,Y_test4)
print(fenshu4)
print('逻辑回归混淆矩阵报告')
Y4_predict=k4.predict(X_test4)

report4=classification_report(Y_test4,Y4_predict,labels=[1,2],target_names=['正常','抑郁'])
print(report3)

''''模型五，无降维的随机森林'''
#pca = PCA(n_components=21)
#feature = pca.fit_transform(feature)

X_train5, X_test5, Y_train5, Y_test5 = train_test_split(feature, target, random_state=16)
biaozhun=StandardScaler()
X_train5=biaozhun.fit_transform(X_train5)
X_test5=biaozhun.fit_transform(X_test5)
k6=RandomForestClassifier(n_estimators=17,max_depth=4,random_state=90)
k6.fit(X_train5, Y_train5)
fenshu5 = k6.score(X_test5, Y_test5)

print(fenshu5)
print('无降维随机森林混淆矩阵报告')
YY_predict=k6.predict(X_test5)

report5=classification_report(Y_test5,YY_predict,labels=[1,2],target_names=['正常','抑郁'])
print(report5)



#ROC曲线绘制和AUC计算
from sklearn.metrics import roc_curve,auc
import matplotlib.pyplot as plt
'''模型一，随机森林ROC曲线绘制和AUC计算'''

res_1 = Y_test.values#预测数据的真值
pre_1 = k1.predict(X_test)#模型的预测值
fpr_1, tpr_1, threshold_1 = roc_curve(res_1, pre_1,pos_label = 2)  ###计算真正率和假正率
roc_auc_1 = auc(fpr_1, tpr_1)  ###计算auc的值


'''模型二，支持向量机ROC曲线绘制和AUC计算'''
res_2 = Y_test2.values
pre_2 = k2.predict(X_test2)
fpr_2, tpr_2, threshold_2 = roc_curve(res_2, pre_2,pos_label = 2)
roc_auc_2 = auc(fpr_2, tpr_2)

'''模型三，逻辑回归ROC曲线绘制和AUC计算'''
res_3 = Y_test3.values
pre_3 = k3.predict(X_test3)
fpr_3, tpr_3, threshold_3 = roc_curve(res_3, pre_3,pos_label = 2)
roc_auc_3 = auc(fpr_3, tpr_3)

'''模型四，神经网络ROC曲线绘制和AUC计算'''

res_4 = Y_test4.values
pre_4 = k4.predict(X_test4)
fpr_4, tpr_4, threshold_4 = roc_curve(res_4, pre_4,pos_label = 2)
roc_auc_4 = auc(fpr_4, tpr_4)

'''模型五，ROC曲线绘制和AUC计算'''
#res_1 = Y_test.tolist()#预测数据的真值,.tolist可以将数据转换为一维数组
#pre_1 = k6.predict(Y_train).tolist()#模型的预测值
#print(X_test5)
#print(Y_test5)
res_5 = Y_test5.values#预测数据的真值
pre_5 = k6.predict(X_test5)##模型的预测值
#print(res_5)
#print(pre_5)
fpr_5, tpr_5, threshold_5 = roc_curve(res_5, pre_5,pos_label = 2)  ###计算真正率和假正率,pos_label=几说明那个值为目标值
roc_auc_5 = auc(fpr_5, tpr_5)  ###计算auc的值

#for i in range(n_classes):

    #fpr[i], tpr[i], _ = roc_curve(res_1[:, i], pre_1[:, i])
    #y_test样例真实标签，y_score学习器预测的样例的概率
    #roc_auc[i] = auc(fpr[i], tpr[i])
    #计算ROC曲线下方的面积，fpr假正例率数组(横坐标)，tpr真正例率数组(纵坐标）




plt.figure(figsize=(8, 5))
plt.plot(fpr_1, tpr_1, color='darkorange',  ###假正率为横坐标，真正率为纵坐标做曲线
         lw=2, label='随机森林 (area = %0.3f)' % roc_auc_1, linestyle='-')  # linestyle为线条的风格（共五种）,color为线条颜色
plt.plot(fpr_2, tpr_2, color='red',
         lw=2, label='支持向量机 (area = %0.3f)' % roc_auc_2, linestyle='--')
plt.plot(fpr_3, tpr_3, color='green',
         lw=2, label='逻辑回归 (area = %0.3f)' % roc_auc_3, linestyle='--')
plt.plot(fpr_5, tpr_5, color='#800080',
         lw=2, label='无降维随机森林 (area = %0.3f)' % roc_auc_5, linestyle=':')
plt.plot(fpr_4, tpr_4, color='#D2691E',
         lw=2, label='神经网络 (area = %0.3f)' % roc_auc_4, linestyle='-.')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([-0.02, 1.05])  # 横竖增加一点长度 以便更好观察图像
plt.ylim([-0.02, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.savefig("hyh.png", dpi=600)  # 保存图片，dpi设置分辨率
plt.show()
