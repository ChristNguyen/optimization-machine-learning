# -*- coding: utf-8 -*-
"""Line search optimization.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/12CKkK2Uq0m2Lt6l_2Y5QaROJwtK3gdmr

1. **preprocesssing data**
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from  sklearn.preprocessing import StandardScaler

sn= StandardScaler()
from sklearn.metrics import r2_score

dataset=pd.read_csv("https://raw.githubusercontent.com/basilhan/datasets/master/kc-house-data.csv")
data=dataset.copy()
data.info()
data.describe()
data.info()
dell=["id","date"]
data=data.drop(dell,axis=1)
 
data["yr_built"].describe()
data["age"]=2020-data["yr_built"]
data["age"].describe()
#data=data[(data["age"]>=10)]
data=data.drop("yr_built",axis=1)
data1=data.copy()
for i in range(0,len(data1)):
    if(data1["yr_renovated"][i]>0):
        data1["yr_renovated"][i]=1

colmn=list(data1.columns)

feature= list(set(colmn)-set(["price"]))

y1=data1["price"].values
x1=data1[feature].values

x1=sn.fit_transform(x1)
y1=np.log(y1)

print("data train: ")
x0 = np.full((x1.shape[0],1),1)
X = np.vstack([np.hstack([x0, x1])])
N = X.shape[0]
D = X.shape[1]
print(X.shape)

"""2. **Tính tham số theta dựa vào phương trình chuẩn**"""

#tinh theta_0 neu khong dung gradient decent
t1 = np.linalg.inv(np.dot(X.transpose(),X))
t2 = np.dot(X.transpose(),y1)
theta_0 = np.dot(t1,t2)
print("Number of parameter:",theta_0.shape)
print(" Parameter are:","\n", theta_0)

"""3. **Khởi tạo các hàm**"""

def loss(X,y,theta):
    u = np.dot(X, theta) - y
    return np.dot(u.transpose(),u)/N
def grad(X, y, theta):
    
    grad = 1/N * X.T.dot(X.dot(theta) - y)
    
    return grad

def bgd(X, y, alpha, T):
    theta = np.zeros((D, T))
    iters = 0
    for t in range(0,T-1):
        
        temp = np.dot(X, theta[:,t]) - y
        gradient = np.dot(X.transpose(), temp)/N
        theta[:,t+1] = theta[:,t] - alpha*gradient
        if np.linalg.norm(grad(X, y, theta[:, t])) / D < 1e-3:
            iters = t
            break
    return (theta[:,:t], iters)

def plot_gda(theta):
 
    J = np.zeros((theta.shape[1]))
    for i in range(theta.shape[1]):
        J[i] = loss(X, y1, theta[:, i])
    #
    # print(J)
    X_plot = np.arange(0, theta.shape[1])
    plt.title(" Loss function and iteration\n")
    plt.xlabel('iteration')
    plt.ylabel('Loss square')
    plt.plot(X_plot, J)
    plt.show()
def plot_y1_ypre(theta): ### vẽ y dự đoán của mô hình và y thực tế
    plt.scatter(np.dot(X, theta0[:,-1]),y1)
    plt.axis('square')
    plt.xlabel("Real price")
    plt.ylabel(" Predict price")
    plt.plot(y1, y1, color='green')  
    plt.title(" Predict price and real price") 
for alpha in [0.9, 0.5, 0.1, 0.01]:
    print( "Loss plot with alpha=",alpha)
    (theta0, t0)=bgd(X,y1,alpha,3000)
    
    print( "With leanring rate",alpha, ", theta: ","\n",theta0[:,-1])
    plot_gda(theta0)
    print(" Number of iteration:",t0+1 )
    y_pred=np.dot(X, theta0[:,-1])
    R_square=np.corrcoef(y_pred,y1)**2    # đo độ phù hợp của mô hình
    print("Accuracy of model:", "{:.0%}".format(R_square[0,1]),"\n"*3,"--------------------------")

"""4. **Biểu diễn tương quan dữa dữ liệu dự đoán(y_pred) và dữ liệu thực(y1)**:
nhận xét: tương đối đúng theo đường chéo
"""

for alpha in [0.3, 0.1, 0.01, 0.005]:
    (theta0, t0)=bgd(X,y1,alpha,3000)
    
    plot_y1_ypre(theta0)

"""5. **Thử thay đổi hệ số alpha theo 1 tỷ lệ beta ( dựa theo phương pháp  backtracking)** để tìm xem giá trị nào của alpha sẽ hội tụ nhanh nhất"""

beta=0.9
b=np.arange(8,18) # để alpha khoảng từ (0.4 đến 0.1)
alpha_list=(beta**b)
print(alpha_list)

for alpha in (alpha_list):
    print( "Loss plot with alpha =",alpha)
    (theta0, t0)=bgd(X,y1,alpha,3000)
    
    plot_gda(theta0)
    print(" Number of iteration:",t0+1,"\n"*3,"--------------" )

"""**Nhận xét: chọn hệ số alpha ~ 0.3 sẽ hội tụ nhanh nhất**( số lần lặp là 19)"""