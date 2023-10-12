# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 03:10:22 2023

@author: ashysky
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from generate_covariance_matrix import generate_covariance_matrix

seed = 10
np.random.seed(seed)
rng = np.random.default_rng(seed=seed)

num_variables = 50
num_groups = 40
# 生成1000个观测的样本
n = 2000
EV_ratio = 0.90
maxpcs = 20
q_ = 14

data = np.zeros((n,1))

for ii in range(0,2):        
    min_diag = 1
    max_diag = 100
    min_rho = 0.95
    max_rho = 0.99
    variance_factor = 0.01

    # 生成对角元素为随机数的矩阵
    sigma_square = round(np.random.uniform(min_diag, max_diag), 2)
    rho = round(np.random.uniform(min_rho, max_rho), 2)
    
    g = int(num_groups/2)
    p_data = int(num_variables*1)
    p_noise = num_variables - p_data
    sigma_data = sigma_square
    sigma_noise = sigma_square * variance_factor
    
    rho_data = rho
    rho_noise = rho * variance_factor
    Rho = 0.75
    
    covariance_matrix = generate_covariance_matrix(g, p_data, p_noise, sigma_data, sigma_noise, rho_data, rho_noise, Rho)
    # print("协方差矩阵：")
    # print(covariance_matrix)
           
    # 生成方差为 0.1 * sigma_square 的噪声协方差矩阵
    noise_cov_matrix = np.eye(covariance_matrix.shape[0]) * (variance_factor * sigma_square)
    
    # 生成服从 cov_matrix 的随机样本
    X = np.random.multivariate_normal(mean=np.zeros(covariance_matrix.shape[0]),
                                      cov=covariance_matrix, size=n)
    
    # 生成服从 noise_cov_matrix 的随机样本
    X_noise = np.random.multivariate_normal(mean=np.zeros(noise_cov_matrix.shape[0]),
                                            cov=noise_cov_matrix, size=n)
        
    X_temp = X + X_noise
    data = np.hstack((data,X_temp))
        
X = data[:,1:]
Y = np.zeros((n,1))

dd = int(num_variables * num_groups / 2)
cofflist = np.random.randint(1,10, dd*2)
#print(cofflist)
print(dd)
for i in range(0,int(num_groups*0.3*num_variables)):
    for j in range(0,n):
        Y[j,0] = X[j,i]*cofflist[i] + Y[j,0]
                
for i in range(dd,int(num_groups*0.3*num_variables) + dd):
    for j in range(0,n):
        Y[j,0] = X[j,i]*cofflist[i-dd] + Y[j,0]
        
noise_Y = np.random.normal(0, 0.1*np.std(Y), n)
observation_y_values = Y + noise_Y.reshape((n,1))

# 指定的缩放范围
new_min = 100
new_max = 200
# 计算缩放后的数值
observation_y_values = ((observation_y_values - np.min(observation_y_values)) / (np.max(observation_y_values) - np.min(observation_y_values))) * (new_max - new_min) + new_min
# 绘制y值的直方图
plt.figure(figsize=(10, 6))
plt.hist(observation_y_values, bins=20, edgecolor='black', alpha=0.7)
plt.xlabel('y Value')
plt.ylabel('Frequency')
plt.title('Distribution of y Values')
plt.grid(True)
plt.show()
# 绘制observation_y_values和observation的图形
plt.figure(figsize=(10, 6))
plt.plot(observation_y_values)
plt.xlabel('Observation')
plt.ylabel('observation_y_values')
plt.title('observation_y_values for Each Observation')
plt.grid(True)
plt.show()
# 将数组按从小到大排序
sorted_values = np.sort(observation_y_values, axis=0)
# 绘制sorted_y_values和相应的observation的图形
plt.figure(figsize=(10, 6))
plt.plot(sorted_values)
plt.xlabel('Observation')
plt.ylabel('Sorted observation_y_values')
plt.title('Sorted observation_y_values for Each Observation')
plt.grid(True)
plt.show()
    
data[:,0] = observation_y_values[:,0]

Rho = np.corrcoef(X.T)
S = np.cov(X.T)

df_syn_x = pd.DataFrame(data)
df_syn_Cov = pd.DataFrame(S)
df_syn_Corr = pd.DataFrame(Rho)
df_syn_x.to_csv("synX__old.csv",index=False,header=None)
df_syn_Cov.to_csv("synCov.csv",index=False,header=None)
# df_syn_Corr.to_csv("synCorr.csv",index=False,header=None)

y = observation_y_values

df_y = pd.DataFrame(observation_y_values)
outlier = df_y.boxplot(return_type = 'dict')
plt.figure()
plt.subplot(1,2,1)
plt.boxplot(x = df_y, # 指定绘制箱线图的数据
         whis = 1.5, # 指定1.5倍的四分位差
         widths = 0.7, # 指定箱线图的宽度为0.8
         patch_artist = True, # 指定需要填充箱体颜色
         showmeans = True, # 指定需要显示均值
         boxprops = {'facecolor':'steelblue'}, # 指定箱体的填充色为铁蓝色
        # 指定异常点的填充色、边框色和大小
         flierprops = {'markerfacecolor':'red', 'markeredgecolor':'red', 'markersize':4}, 
         # 指定均值点的标记符号（菱形）、填充色和大小
        meanprops = {'marker':'D','markerfacecolor':'black', 'markersize':4}, 
         medianprops = {'linestyle':'--','color':'orange'}, # 指定中位数的标记符号（虚线）和颜色
         labels = [''] # 去除箱线图的x轴刻度值
         )

Y_raw = observation_y_values
Y_raw.resize(Y_raw.shape[0])

XY_raw = pd.read_csv('synX__old.csv', header = None)

X_raw = XY_raw.iloc[: , 1:]  # Input X
X_raw = X_raw.to_numpy()
X_raw.resize(X_raw.shape[0],int(X_raw.shape[1]/num_variables),num_variables)

# Initialize a matrix for the transformed variables
Z = np.zeros((X_raw.shape[0]+2,1))
A = np.zeros((X_raw.shape[2]+2,1))
p = 1   # Variable index



for i in list(range(0,X_raw.shape[1])):    # For each sensor/data source
    # Extract the variables(around 650) for all obns(298) from sensor i    
    temp = X_raw[:,i,:]     # [298 * 648]
    
    # Standardlized
    scaler = StandardScaler()
    scaler.fit(temp)
    temp = scaler.transform(temp)
    
    q = q_
    # Extract the first q features/transformed variables
    pca = PCA(n_components = q)    
    pca.fit(temp)
    
    # While the explained variance of the first q features is less than 0.9
    while np.sum(pca.explained_variance_ratio_)<EV_ratio:
        q = q + 1   # Increase the q(# of extracted features) by 1
        pca = PCA(n_components = q)    
        pca.fit(temp)
        
        print(q)    
        if q > maxpcs:  # If the # of feature reach 40 break
            print(np.sum(pca.explained_variance_ratio_))
            break
    # Find the transformed variables z
    z = pca.fit_transform(temp)
    
    # Find the loading vectors for the pca
    a = pca.components_    
    # z = temp * a.T    
    a = a.T
    
    # Write the sensor name/data source for each z
    s = np.ones((1,z.shape[1])) * i   + 1
    # Write the index/name of the each features
    v = list(range(p , p + z.shape[1]))
    p = p + z.shape[1]    
    
    a = np.vstack((np.vstack((s,v)),a))
    # Update the matrix for the loadings
    A = np.append(A,a,axis=1) # (650 * 359)
    
    z = np.vstack((np.vstack((s,v)),z))
    # Update the matrix for the transformed variables
    Z = np.append(Z,z,axis=1)   # (300 * 359)

# Put the output y/ in the first column    
Z[2:Z.shape[0],0] = Y_raw

# Save the processed data in .csv
dataframe = pd.DataFrame(Z)
dataframe.to_csv('Feature%s,%s.csv'%(EV_ratio, maxpcs),index=False,header=False,sep=',')

# Save the processed data in .csv
dataframe = pd.DataFrame(A)
dataframe.to_csv('Loading%s,%s.csv'%(EV_ratio, maxpcs),index=False,header=False,sep=',')


# 将数据分成训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(Z[2:Z.shape[0],1:], y, test_size=0.2, random_state=42)
# 创建线性回归模型并在训练集上拟合
model = LinearRegression()
model.fit(X_train, y_train)
# 在测试集上进行预测
y_pred = model.predict(X_test)
# 计算MSE
mse = np.mean((y_test - y_pred) ** 2)
print("Mean Squared Error:", mse)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
print("Mean Absolute Percentage Error:", mape)
# 计算 R^2
mean_actual = np.mean(y_test)
r_squared = 1 - (np.sum((y_test - y_pred) ** 2) / np.sum((y_test - mean_actual) ** 2))
print("R-squared:", r_squared)
# 计算 MAE
mae = np.mean(np.abs(y_test - y_pred))
print("Mean Absolute Error:", mae)

lb_actual_test = int(min(y_test)**0.95)-1
ub_actual_test = int(max(y_test)*1.05)+1

# 绘制预测值和实际值之间的散点图
plt.figure(figsize = (7,7))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.xlim(lb_actual_test,ub_actual_test)
plt.ylim(lb_actual_test,ub_actual_test)
plt.title("Actual vs. Predicted Values")
plt.show()