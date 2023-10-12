# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 14:56:31 2023

@author: ashysky
"""

'''
Multi-source AdaBoost algorithm and benchmarks
'''

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import datetime
import csv
import time
#import random
#import math
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold

from sklearn import tree
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import AdaBoostRegressor

#from twilio.rest import Client

def print_line(char):
    print(char*60)

def mae(y_true, y_pred):
    np.set_printoptions(suppress=False)
    n = len(y_true)
    mae = sum(np.abs(y_true - y_pred))/n    
    return mae

def mape(y_true, y_pred):
    n = len(y_true)
    mape = (100/n) * np.sum(np.abs((y_true - y_pred) / y_true))
    return mape

def rsquare(y_true, y_pred):
    corr_matrix = np.corrcoef(y_true, y_pred)
    corr = corr_matrix[0,1]
    rsquare = corr**2
    return rsquare

# Data set            
# data = pd.read_csv('Feature0.9,20.csv', header = None)    # Data set
data = pd.read_csv('10,Feature with noise = 0.0, 1.csv', header = None)    # Data set


df_v_c = data.iloc[: , 1:]  # Input X
df_y_c = data.iloc[: , :1]  # Output Y

df_v = df_v_c.iloc[2:, :]   # X without label
df_y = df_y_c.iloc[2:, :]   # Y without label            
df_y = df_y.iloc[:,:]

# Draw the boxplot for the data to find outliers
plt.subplot(1,2,2)
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
#plt.show() # 显示图形

var_name = np.asarray(df_v_c.iloc[0])   # Variable label/name

purpose = "optimization"  # 可以根据实际情况赋值
# purpose = "test"  # 可以根据实际情况赋值

# 创建算法列表（只保留名称）
algorithms = [
    "Y-bar",
    "MLR",
    "CART",
    "SVR",
    "XGBoost",
    "AdaBoost.R2",    
    "AdaBoost.SiSB",
    "AdaBoost.AVE",
    "mAdaBoost.CW"
]

# algorithms = [
#     "AdaBoost.SiSB",
#     "AdaBoost.AVE",
#     "mAdaBoost.CW"
# ]
#--------------parameters of Proposed method
if purpose == "test":
    list_m1 = [2,3]
    list_m2 = [2]
    list_max_depth = [8,9,10] #[3,6,8,10]
    # gamma = 20
    # list_gamma = list(np.linspace(0, 1, gamma))
    gamma = 2
    list_gamma = list(np.linspace(0.8, 1, gamma))    
    seed = 0    #Random Seed
    K = 3 # Num Experiments
    test_size = 0.25    #1/4 of the observations are testing data
    n_folds = 2     #CV fold
    ii = 1
    T = 3
    
elif purpose == "optimization":
    list_m1 = [3] #[0,2,3]
    list_m2 = [2]
    list_max_depth = [8] #[3,6,8,10,7, 8, 9, 10, 11]
    gamma = 11    
    list_gamma = list(np.linspace(0.8, 1, gamma))    
    
    seed = 0    #Random Seed
    K = 10 # Num Experiments
    test_size = 0.25    #1/4 of the observations are testing data
    n_folds = 5     #CV fold
    ii = 1
    T = 8
    
# Split training and testing data
train_vA, test_v, train_yA, test_y = train_test_split(df_v, df_y, test_size = test_size, random_state = 0)


para_AdaBoostSiSB_best_n_estimators = np.zeros((int(df_v_c[df_v_c.idxmax(axis=1)[0]][0])))
# Noise Level
for nl in [0]:
#for nl in [0,1,2,5,10]:    
    # 创建结果存储的多维数组
    results = np.zeros((len(algorithms), 9, K))    
    actual_test = np.zeros((len(algorithms), len(test_y), K))
    predict_test = np.zeros((len(algorithms), len(test_y), K))
    # 循环迭代 k
    for k in range(0, K):
        print_line('-')
        print_line('-')
        print(k)
        print_line('-')
        print_line('-')
        # Split training and testing data
        train_vA, test_v, train_yA, test_y = train_test_split(df_v, df_y, test_size = test_size, random_state = k)
        # 获取训练集和测试集的特征名称
        train_feature_names = train_vA.columns.tolist()
        test_feature_names = test_v.columns.tolist()
        
        # 按照训练集的特征名称，将测试集的特征名称调整为一致
        test_v.columns = train_feature_names
        
        # Add noise to the test data
        test_std = np.std(test_v.to_numpy())
        test_v = test_v.to_numpy() + np.random.normal(0,test_std * nl,size=(test_v.shape[0],test_v.shape[1]))
        test_v = pd.DataFrame(data = test_v)
        
        filename = f"parameter_{k}.txt"
        # Open the file in write mode
        with open(filename, "w",encoding="utf-8") as file:
            
            for i, algorithm_name in enumerate(algorithms):
                print(f"Running algorithm: {algorithm_name}")
                start_time = time.time()  # 记录开始时间
                # 根据算法名称执行相应的代码块
                if algorithm_name == "MLR":
                    regr = LinearRegression()
                    regr.fit(train_vA, train_yA.values.ravel())                            
                    
                    results[i,0,k] = i
                    results[i,1,k] = mean_squared_error(train_yA.values.ravel(), regr.predict(train_vA))
                    results[i,2,k] = mean_squared_error(test_y.values.ravel(), regr.predict(test_v))
                    results[i,3,k] = mae(train_yA.values.ravel(), regr.predict(train_vA))
                    results[i,4,k] = mae(test_y.values.ravel(), regr.predict(test_v))
                    results[i,5,k] = regr.score(train_vA, train_yA.values.ravel())
                    results[i,6,k] = regr.score(test_v, test_y.values.ravel())
                    results[i,7,k] = mape(train_yA.values.ravel(), regr.predict(train_vA))
                    results[i,8,k] = mape(test_y.values.ravel(), regr.predict(test_v))
                    
                    actual_test[i,:,k] = test_y.values.ravel()
                    predict_test[i,:,k] = regr.predict(test_v)
                    
                elif algorithm_name == "CART":
                    if k == 0:                    
                        regr = DecisionTreeRegressor()
                        
                        if purpose == "test":
                            param_grid = {                                
                                'max_depth': [None,1]
                            }
                        elif purpose == "optimization":
                            param_grid = {
                                # 'max_depth': [None,1,2,3,4, 5, 10]
                                'max_depth': [None,3, 5, 7, 9, 11]
                            }                        
                        # 使用网格搜索
                        grid_search = GridSearchCV(estimator=regr, param_grid=param_grid, cv=n_folds)  # 设置 cv=5，表示5折交叉验证
                        grid_search.fit(train_vA, train_yA.values.ravel())
                        
                        # 获取最佳参数
                        best_params = grid_search.best_params_
                        
                        # 打印最佳参数和得分
                        output_string = f"算法名称: {algorithm_name} - 最佳参数: {best_params}\n"
                        print(output_string)
                        
                        # 将输出内容保存到文件
                        file.write(output_string)
                        
                        # print("最佳交叉验证分数：", grid_search.best_score_)
                        para_CART_best_max_depth = grid_search.best_params_['max_depth']
                    
                    regr = DecisionTreeRegressor(random_state=seed,max_depth=para_CART_best_max_depth)
                    regr.fit(train_vA, train_yA.values.ravel())                            
                    
                    results[i,0,k] = i
                    results[i,1,k] = mean_squared_error(train_yA.values.ravel(), regr.predict(train_vA))
                    results[i,2,k] = mean_squared_error(test_y.values.ravel(), regr.predict(test_v))
                    results[i,3,k] = mae(train_yA.values.ravel(), regr.predict(train_vA))
                    results[i,4,k] = mae(test_y.values.ravel(), regr.predict(test_v))
                    results[i,5,k] = regr.score(train_vA, train_yA.values.ravel())
                    results[i,6,k] = regr.score(test_v, test_y.values.ravel())
                    results[i,7,k] = mape(train_yA.values.ravel(), regr.predict(train_vA))
                    results[i,8,k] = mape(test_y.values.ravel(), regr.predict(test_v))
                    
                    actual_test[i,:,k] = test_y.values.ravel()
                    predict_test[i,:,k] = regr.predict(test_v)
                    
                elif algorithm_name == "SVR":
                    if k == 0:                    
                        regr = SVR()
                        if purpose == "test":
                            # 定义参数网格
                            param_grid = {
                                'kernel': ['linear'],
                                'C': [0.1, 1],
                                'epsilon': [0.005, 0.01]
                            }
                        elif purpose == "optimization":
                            # 定义参数网格
                            param_grid = {
                                'kernel': ['linear', 'rbf', 'poly'],
                                'C': [1, 10, 15, 20],
                                'epsilon': [0.002, 0.005, 0.01]
                                # 'C': [0.1, 1, 10, 15, 20],
                                # 'epsilon': [0.0005, 0.001, 0.005, 0.01]
                            }
                        
                        # 使用网格搜索
                        grid_search = GridSearchCV(estimator=regr, param_grid=param_grid, cv=n_folds)
                        grid_search.fit(train_vA, train_yA.values.ravel())
                        
                        # 获取最佳参数
                        best_params = grid_search.best_params_
                        
                        # 打印最佳参数和得分
                        output_string = f"算法名称: {algorithm_name} - 最佳参数: {best_params}\n"
                        print(output_string)
                        
                        # 将输出内容保存到文件
                        file.write(output_string)
                        
                        para_SVR_best_kernel = grid_search.best_params_['kernel']
                        para_SVR_best_C = grid_search.best_params_['C']
                        para_SVR_best_epsilon = grid_search.best_params_['epsilon']
                        
                    regr = SVR(kernel = para_SVR_best_kernel, C = para_SVR_best_C, epsilon = para_SVR_best_epsilon)
                    regr.fit(train_vA, train_yA.values.ravel()) 
                                                    
                    results[i,0,k] = i
                    results[i,1,k] = mean_squared_error(train_yA.values.ravel(), regr.predict(train_vA))
                    results[i,2,k] = mean_squared_error(test_y.values.ravel(), regr.predict(test_v))
                    results[i,3,k] = mae(train_yA.values.ravel(), regr.predict(train_vA))
                    results[i,4,k] = mae(test_y.values.ravel(), regr.predict(test_v))
                    results[i,5,k] = regr.score(train_vA, train_yA.values.ravel())
                    results[i,6,k] = regr.score(test_v, test_y.values.ravel())
                    results[i,7,k] = mape(train_yA.values.ravel(), regr.predict(train_vA))
                    results[i,8,k] = mape(test_y.values.ravel(), regr.predict(test_v))
                    
                    actual_test[i,:,k] = test_y.values.ravel()
                    predict_test[i,:,k] = regr.predict(test_v)
                                        
                elif algorithm_name == "XGBoost":
                    # 获取训练集和测试集的特征名称
                    train_feature_names = train_vA.columns.tolist()
                    test_feature_names = test_v.columns.tolist()
                    
                    # 按照训练集的特征名称，将测试集的特征名称调整为一致
                    test_v.columns = train_feature_names                
    
                    if k == 0:                    
                        # 创建 XGBoost 回归模型
                        model = xgb.XGBRegressor(objective='reg:squarederror', random_state=seed)
                        if purpose == "test":
                            # 定义参数网格
                            param_grid = {
                                'n_estimators': [1, 5],
                                'learning_rate': [0.1, 1.2],
                                'max_depth': [6, 7]
                            }
                        elif purpose == "optimization":
                            # 定义参数网格
                            param_grid = {
                                'n_estimators': [ 45, 50, 60],
                                'learning_rate': [0.08, 0.1, 1.2],
                                'max_depth': [7, 8, 9]
                                # 'n_estimators': [10, 20, 30, 35, 40,, 50],
                                # 'learning_rate': [0.05, 0.1, 1.0],
                                # 'max_depth': [5, 6, 7, 9]
                            }
                        
                        # 使用网格搜索来优化参数
                        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=n_folds)
                        grid_search.fit(train_vA, train_yA.values.ravel())
                        
                        # 获取最佳参数
                        best_params = grid_search.best_params_
                        
                        # 打印最佳参数和得分
                        output_string = f"算法名称: {algorithm_name} - 最佳参数: {best_params}\n"
                        print(output_string)
                        
                        # 将输出内容保存到文件
                        file.write(output_string)
                        
                        # 获取最佳参数
                        para_XGBoost_best_n_estimators = grid_search.best_params_['n_estimators']
                        para_XGBoost_best_learning_rate = grid_search.best_params_['learning_rate']
                        para_XGBoost_best_max_depth = grid_search.best_params_['max_depth']
                        
                    regr = xgb.XGBRegressor(objective='reg:squarederror', random_state=seed, 
                                    max_depth = para_XGBoost_best_max_depth, 
                                    n_estimators = para_XGBoost_best_n_estimators, 
                                    learning_rate = para_XGBoost_best_learning_rate)
                    regr.fit(train_vA, train_yA.values.ravel()) 
                                                    
                    results[i,0,k] = i
                    results[i,1,k] = mean_squared_error(train_yA.values.ravel(), regr.predict(train_vA))
                    results[i,2,k] = mean_squared_error(test_y.values.ravel(), regr.predict(test_v))
                    results[i,3,k] = mae(train_yA.values.ravel(), regr.predict(train_vA))
                    results[i,4,k] = mae(test_y.values.ravel(), regr.predict(test_v))
                    results[i,5,k] = regr.score(train_vA, train_yA.values.ravel())
                    results[i,6,k] = regr.score(test_v, test_y.values.ravel())
                    results[i,7,k] = mape(train_yA.values.ravel(), regr.predict(train_vA))
                    results[i,8,k] = mape(test_y.values.ravel(), regr.predict(test_v))
                    
                    actual_test[i,:,k] = test_y.values.ravel()
                    predict_test[i,:,k] = regr.predict(test_v)
                                       
                elif algorithm_name == "AdaBoost.R2":
                    if k == 0:                    
                        regr = AdaBoostRegressor()
                        if purpose == "test":
                            # 定义参数网格
                            param_grid = {
                                'n_estimators': [1, 5],
                                'learning_rate': [0.1, 0.5]
                            }
                        elif purpose == "optimization":
                            # 定义参数网格
                            param_grid = {
                                'n_estimators': [10, 20, 30, 40],
                                'learning_rate': [0.05, 0.1, 0.5]
                                # 'n_estimators': [5, 10, 15, 20, 25, 30, 35],
                            }
                        # 使用网格搜索
                        grid_search = GridSearchCV(estimator=regr, param_grid=param_grid, cv=n_folds)
                        grid_search.fit(train_vA, train_yA.values.ravel())
                        
                        # 获取最佳参数
                        best_params = grid_search.best_params_
                        
                        # 打印最佳参数和得分
                        output_string = f"算法名称: {algorithm_name} - 最佳参数: {best_params}\n"
                        print(output_string)
                        
                        # 将输出内容保存到文件
                        file.write(output_string)
                        
                        para_AdaBoostR2_best_n_estimators = grid_search.best_params_['n_estimators']
                        para_AdaBoostR2_best_learning_rate = grid_search.best_params_['learning_rate']
                        
                    regr = AdaBoostRegressor(n_estimators = para_AdaBoostR2_best_n_estimators, 
                                             learning_rate = para_AdaBoostR2_best_learning_rate)
                    regr.fit(train_vA, train_yA.values.ravel()) 
                                                    
                    results[i,0,k] = i
                    results[i,1,k] = mean_squared_error(train_yA.values.ravel(), regr.predict(train_vA))
                    results[i,2,k] = mean_squared_error(test_y.values.ravel(), regr.predict(test_v))
                    results[i,3,k] = mae(train_yA.values.ravel(), regr.predict(train_vA))
                    results[i,4,k] = mae(test_y.values.ravel(), regr.predict(test_v))
                    results[i,5,k] = regr.score(train_vA, train_yA.values.ravel())
                    results[i,6,k] = regr.score(test_v, test_y.values.ravel())
                    results[i,7,k] = mape(train_yA.values.ravel(), regr.predict(train_vA))
                    results[i,8,k] = mape(test_y.values.ravel(), regr.predict(test_v))
                    
                    actual_test[i,:,k] = test_y.values.ravel()
                    predict_test[i,:,k] = regr.predict(test_v)
                    
                elif algorithm_name == "AdaBoost.SiSB":
                    temp_results = np.zeros(((int(df_v_c[df_v_c.idxmax(axis=1)[0]][0])),9))
                    temp_predict = np.zeros((test_y.size,(int(df_v_c[df_v_c.idxmax(axis=1)[0]][0]))))
                    temp_train = np.zeros((train_yA.size,(int(df_v_c[df_v_c.idxmax(axis=1)[0]][0]))))
                    
                    for m in range(int(df_v_c[df_v_c.idxmin(axis=1)[0]][0])-1, (int(df_v_c[df_v_c.idxmax(axis=1)[0]][0]))):
                        idx_m = [i for i, x in enumerate(var_name) if x == m + 1]   # Find the Index of Data Source m
                        train_m = train_vA.iloc[:, idx_m]    # Training data for Source m
                        test_m = test_v.iloc[:, idx_m]     # Testing data for Source m            
                        
                        if k == 0:                                                
                            regr = AdaBoostRegressor()
                            if purpose == "test":
                                # 定义参数网格
                                param_grid = {
                                    # 'n_estimators': [3,5,10,15,20].
                                    'n_estimators': [3,4]
                                }   
                            elif purpose == "optimization":
                                # 定义参数网格
                                param_grid = {
                                    'n_estimators': [25, 50, 75]
                                    # 'n_estimators': [None,4, 3, 5, 7, 9]
                                } 
                            # 使用网格搜索
                            grid_search = GridSearchCV(estimator=regr, param_grid=param_grid, cv=n_folds)
                            grid_search.fit(train_vA, train_yA.values.ravel())                            
                            # 获取最佳参数
                            best_params = grid_search.best_params_                            
                            # 打印最佳参数和得分
                            output_string = f"算法名称: {algorithm_name} - 最佳参数: {best_params}\n"
                            print(output_string)                            
                            # 将输出内容保存到文件
                            file.write(output_string)
                            
                            para_AdaBoostSiSB_best_n_estimators[m] = grid_search.best_params_['n_estimators']
                            # para_AdaBoostSiSB_best_learning_rate = grid_search.best_params_['learning_rate']
                            
                        # regr = AdaBoostRegressor(n_estimators = para_AdaBoostSiSB_best_n_estimators, 
                                                 # learning_rate = para_AdaBoostSiSB_best_learning_rate)
                        regr = AdaBoostRegressor(n_estimators = int(para_AdaBoostSiSB_best_n_estimators[m]))
                        regr.fit(train_m, train_yA.values.ravel()) 
                                                        
                        temp_results[m,0] = m
                        temp_results[m,1] = mean_squared_error(train_yA.values.ravel(), regr.predict(train_m))
                        temp_results[m,2] = mean_squared_error(test_y.values.ravel(), regr.predict(test_m))
                        temp_results[m,3] = mae(train_yA.values.ravel(), regr.predict(train_m))
                        temp_results[m,4] = mae(test_y.values.ravel(), regr.predict(test_m))
                        temp_results[m,5] = regr.score(train_m, train_yA.values.ravel())
                        temp_results[m,6] = regr.score(test_m, test_y.values.ravel())
                        temp_results[m,7] = mape(train_yA.values.ravel(), regr.predict(train_m))
                        temp_results[m,8] = mape(test_y.values.ravel(), regr.predict(test_m))
                        
                        temp_train[:,m] = regr.predict(train_m)
                        temp_predict[:,m] = regr.predict(test_m)
                    # 找到第二列的最小元素的索引
                    min_index = np.argmin(temp_results[:, 1]) 
                    
                    results[i,0,k] = i
                    for temp in range(1,9):
                        results[i,temp,k] = temp_results[min_index,temp]
                        
                    actual_test[i,:,k] = test_y.values.ravel()
                    predict_test[i,:,k] = temp_predict[:,min_index]
                    
                    i = i + 1
                    predict_test[i,:,k] = np.mean(temp_predict, axis=1)
                    predict_train = np.mean(temp_train, axis=1)
                    
                    results[i,0,k] = i
                    results[i,1,k] = mean_squared_error(train_yA.values.ravel(), predict_train)
                    results[i,2,k] = mean_squared_error(test_y.values.ravel(), predict_test[i,:,k])
                    results[i,3,k] = mae(train_yA.values.ravel(), predict_train)
                    results[i,4,k] = mae(test_y.values.ravel(), predict_test[i,:,k])
                    results[i,5,k] = rsquare(train_yA.values.ravel(), predict_train)
                    results[i,6,k] = rsquare(test_y.values.ravel(), predict_test[i,:,k])
                    results[i,7,k] = mape(train_yA.values.ravel(), predict_train)
                    results[i,8,k] = mape(test_y.values.ravel(), predict_test[i,:,k])
                    
                    # ai = i + 1
                    # predict_test[ai,:,k] = np.mean(temp_predict, axis=1)
                    # predict_train = np.mean(temp_train, axis=1)                    
                    # results[ai,0,k] = ai
                    # results[ai,1,k] = mean_squared_error(train_yA.values.ravel(), predict_train)
                    # results[ai,2,k] = mean_squared_error(test_y.values.ravel(), predict_test[i,:,k])
                    # results[ai,3,k] = mae(train_yA.values.ravel(), predict_train)
                    # results[ai,4,k] = mae(test_y.values.ravel(), predict_test[i,:,k])
                    # results[ai,5,k] = rsquare(train_yA.values.ravel(), predict_train)
                    # results[ai,6,k] = rsquare(test_y.values.ravel(), predict_test[i,:,k])
                    # results[ai,7,k] = mape(train_yA.values.ravel(), predict_train)
                    # results[ai,8,k] = mape(test_y.values.ravel(), predict_test[i,:,k])
                    
                elif algorithm_name == "AdaBoost.AVE":
                    regr = AdaBoostRegressor()
                elif algorithm_name == "mAdaBoost.CW":
                    if k == 0:                                            
                        # parameters need to be optimized                                       
                        # list_m1 = [2,3]
                        # list_m2 = [2]
                        # list_max_depth = [3]
                        # gamma = 2
                                                
                        result_proposed = np.random.random((n_folds*ii, len(list_m1), len(list_m2), gamma, len(list_max_depth)))
                        # 创建 KFold 对象
                        kf = KFold(n_splits=n_folds*ii)                
                        # 遍历交叉验证的迭代
                        index1 = -1
                        for train_index, val_index in kf.split(train_vA):
                            index1 = index1 + 1
                            print(index1)
                            # 获取训练集和验证集
                            train_vp, vali_v = train_vA.iloc[train_index], train_vA.iloc[val_index]
                            train_yp, vali_y = train_yA.iloc[train_index], train_yA.iloc[val_index]
                            
                            S = int(df_v_c[df_v_c.idxmax(axis=1)[0]][0])     # Number of Data Source S
                            N = len(train_yp)
                            
                            
                            #method = "MLR"     #Regereesion Method
                            method = "CART"
                            index5 = -1
                            for max_depth in list_max_depth:                    
                                index5 = index5 + 1
                                index2 = -1
                                for m1 in list_m1: #[0:"Single iteration", 1:"Median t iteration", 2:"Mean t iteration", 3:"Regreesion t iteration"]
                                    # print(m1)
                                    index2 = index2 + 1
                                    index3 = -1
                                    for m2 in list_m2: #[0:"Simple Mean", 1:"Median M models", 2:"Mean M models",3: "Regreesion M models"]
                                        # print(m2)
                                        index3 = index3 + 1
                                        # Selet the fusion rule
                                        Fusion_1_list = ["Single iteration", "Median t iteration", "Mean t iteration", "Regreesion t iteration"]
                                        fusion_1 = Fusion_1_list[m1]
                                        Fusion_2_list = ["Simple Mean", "Median M models", "Mean M models", "Regreesion M models"]
                                        fusion_2 = Fusion_2_list[m2]
                                        index4 = -1
                                        for Gamma in list_gamma:
                                            # print(Gamma)
                                            index4 = index4 + 1
                                            # Weight of data sources and overall
                                            w = np.ones((T,N,S))/N
                                            wA = np.ones((T,N))/N
                                            # loss of data sources and overall
                                            l = np.zeros((T,N,S))
                                            lA = np.zeros((T,N))
                                            # Loss function of data sources and overall
                                            L = np.zeros((T,N,S))
                                            LA = np.zeros((T,N))
                                        
                                            Dent = np.zeros((T,S))
                                            DentA = np.zeros((T,1))
                                        
                                            Lbar = np.zeros((T,S))
                                            LbarA = np.zeros((T,1))
                                        
                                            Beta = np.zeros((T,S))
                                            Alaph = np.zeros((T,1))
                                        
                                            Beta_sum = np.zeros((T,S))
                                            Beta_log = np.zeros((T,S)) 
                                        
                                            Z = np.zeros((T,S))
                                            Z_A = np.zeros((T,1))
                                        
                                            y_hat_train_m = np.zeros((T,N,S))
                                            y_hat_test_m = np.zeros((T,len(vali_v),S))
                                            
                                            y_hat_train_m_f = np.zeros((T,N,S))
                                            y_hat_test_m_f = np.zeros((T,len(vali_v),S))
                                        
                                        
                                            y_hat_train = np.zeros((T,N))
                                            y_hat_test = np.zeros((T,len(vali_v)))
                                        
                                            s_model = [[] for j in range(0,T)]
                                            s_modelA = [[] for j in range(0,T)]
                                        
                                            train_re = np.zeros((T,N,S))
                                            test_re = np.zeros((T,len(vali_v),S))
                                        
                                            train_w_re = np.zeros((T,N,S))
                                            test_w_re = np.zeros((T,len(vali_v),S))
                                        
                                            train_w_re_sum = np.zeros((T,N,S))
                                            test_w_re_sum = np.zeros((T,len(vali_v),S))
                                        
                                            MSE_trre = np.zeros((T,S))
                                            MSE_tere = np.zeros((T,S))
                                            # R2
                                            R2_trre = np.zeros((T,S))
                                            R2_tere = np.zeros((T,S))
                        
                                            # Weighted Result
                                            train_w_reA = np.zeros((T,N))
                                            test_w_reA = np.zeros((T,len(vali_v)))
                                        
                                            # MSE
                                            MSE_trreA = np.zeros((T,1))
                                            MSE_tereA = np.zeros((T,1))
                                            # mae
                                            mae_trreA = np.zeros((T,1))
                                            mae_tereA = np.zeros((T,1))
                                            # R-square
                                            R2_trreA = np.zeros((T,1))
                                            R2_tereA = np.zeros((T,1))
                                            
                                            #----------------------------------------------------------------
                                            # Create linear regression object
                                            if method == "MLR":
                                                regr = linear_model.LinearRegression()
                                            else:
                                                regr = tree.DecisionTreeRegressor(max_depth=max_depth)
                                                
                                            for t in range(0,T-1):
                                                for m in range(int(df_v_c[df_v_c.idxmin(axis=1)[0]][0])-1, (int(df_v_c[df_v_c.idxmax(axis=1)[0]][0]))):
                                                    idx_m = [i for i, x in enumerate(var_name) if x == m + 1]   # Find the Index of Data Source m
                                                    if t == 0:                
                                                        train_m = train_vp.iloc[:, idx_m]    # Training data for Source m
                                                        test_m_origi = vali_v.iloc[:, idx_m]     # Testing data for Source m            
                                                        train_m_origi = train_m
                                                        train_ypm = train_yp                                
                                                        #----------------------------------------------
                                                        # Train the model using the training sets
                                                        regr.fit(train_m, train_yp.values.ravel())                
                                                    else:
                                                        train_m_origi = train_vp.iloc[:, idx_m] 
                                                        test_m_origi = vali_v.iloc[:, idx_m]     # Original Testing data for Source m
                                                                                        
                                                        # Sampling based on Weight
                                                        x = np.arange(len(train_yp))
                                                        rate = w[t,:,m]
                                                        # Check if rate contains NaN values
                                                        if np.any(np.isnan(rate)):
                                                            rate = np.ones(len(rate)) * (1 / len(rate))
                                                            w[t,:,m] = rate
                                                        # print("Modified rate:", rate)
                                                        temp = np.random.choice(a=x, size=len(train_yp), replace=True, p=rate)
                                                        temp = temp.tolist()       
                                                        # Original Training data for Source m
                                                        train_m_origi = train_m_origi.to_numpy()
                                                        train_yptemp = train_yp.to_numpy()
                                                        # Sampling based on Weight
                                                        train_m = train_m_origi[temp[:],:]
                                                        train_ypm = train_yptemp[temp[:]]
                                                        #-------------------------------------------------------------
                                                        # Train the model using the training sets
                                                        regr.fit(train_m, train_ypm)                            
                                                    # Save the model to list
                                                    s_model[t].append(pickle.dumps(regr))                            
                                                    
                                                    # Calculate loss for each training example
                                                    l[t,:,m]  = abs(regr.predict(train_m_origi) - train_yp.values.ravel())               
                                                    Dent[t,m] = max(l[t,:,m])       # Calculate Dent_t^m (the max error)
                                                    for ii in range(0,N):           # Calculate Loss Function L_t^m(i)
                                                        if Dent[t, m] != 0:
                                                            L[t, ii, m] = l[t,ii,m]/Dent[t,m]     # Linear Loss Function
                                                        else:
                                                            L[t, ii, m] = 1e-8
                                                        
                                                    # Calculate an average loss Lbar_t^m
                                                    for ii in range(0,N):
                                                        # print(Lbar[t,m])
                                                        # print(L[t,ii,m] * w[t,ii,m])   
                                                        Lbar[t,m]  = Lbar[t,m] + L[t,ii,m] * w[t,ii,m]
                                                        # print(Lbar[t,m])
                                                        # print(ii)                            
                                                    # Calculate Beta
                                                    Beta[t,m] = Lbar[t,m] / (1 - Lbar[t,m])                            
                                                    #---------------------------------------------------------------------------
                                                    # Fusion the y_hat_m from t different iterations for data source m
                                                    if fusion_1 == "Single iteration":
                                                        # Generate y_hat_m for All Source based on the models.
                                                        y_hat_train_m[t,:,m] = regr.predict(train_m_origi)
                                                        y_hat_test_m[t,:,m] = regr.predict(test_m_origi)
                                                        
                                                        y_hat_train_m_f[t,:,m] = regr.predict(train_m_origi)
                                                        y_hat_test_m_f[t,:,m] = regr.predict(test_m_origi)
                                                    elif fusion_1 == "Median t iteration":
                                                        y_hat_train_m[t,:,m] = regr.predict(train_m_origi)
                                                        y_hat_test_m[t,:,m] = regr.predict(test_m_origi)
                                                        
                                                        Beta_temp = Beta[0:t+1,m]
                                                        WeightedMedian = np.concatenate((np.asmatrix(np.arange(len(Beta_temp))).T,np.asmatrix(Beta_temp).T), axis=1)
                                                        WeightedMedian = WeightedMedian[np.lexsort(WeightedMedian.T)]
                                                        temp = 0
                                                        for ii in range(0,len(Beta_temp)):
                                                            temp = temp + WeightedMedian[0,ii,1]
                                                            if temp >= np.sum(WeightedMedian[0,:,1],1)/2:
                                                                index = ii
                                                                iter_index = WeightedMedian[0,index,0]
                                                                break
                                                        #print("iter_index =", iter_index)
                                                        iter_index = int(iter_index)
                                                        #Check if Model is Saved
                                                        regr_temp = pickle.loads(s_model[iter_index][m])
                                                        
                                                        y_hat_train_m_f[t,:,m] = y_hat_train_m[iter_index,:,m]
                                                        y_hat_test_m_f[t,:,m] = y_hat_test_m[iter_index,:,m]
         
                                                    elif fusion_1 == "Mean t iteration":
                                                        y_hat_train_m[t,:,m] = regr.predict(train_m_origi)
                                                        y_hat_test_m[t,:,m] = regr.predict(test_m_origi)
                                                        # Weighted Result
                                                        train_w_re[t,:,m] = y_hat_train_m[t,:,m] * np.log(1/Beta[t,m])
                                                        test_w_re[t,:,m] = y_hat_test_m[t,:,m] * np.log(1/Beta[t,m])
                                                        
                                                        if t == 0:
                                                            Beta_sum[t,m] = np.log(1/Beta[t,m])                
                                                            train_w_re_sum[t,:,m] = train_w_re[t,:,m]              
                                                            test_w_re_sum[t,:,m] = test_w_re[t,:,m]                
                                                        else:
                                                            Beta_sum[t,m] = Beta_sum[t-1,m] + np.log(1/Beta[t,m])                
                                                            train_w_re_sum[t,:,m] = train_w_re_sum[t-1,:,m] + train_w_re[t,:,m]
                                                            test_w_re_sum[t,:,m] = test_w_re_sum[t-1,:,m] + test_w_re[t,:,m]
                                                                                        
                                                        y_hat_train_m_f[t,:,m]= train_w_re_sum[t,:,m] / Beta_sum[t,m]
                                                        y_hat_test_m_f[t,:,m]= test_w_re_sum[t,:,m] / Beta_sum[t,m]
                                                        
                                                    elif fusion_1 == "Regreesion t iteration":
                                                        y_hat_train_m[t,:,m] = regr.predict(train_m_origi)
                                                        y_hat_test_m[t,:,m] = regr.predict(test_m_origi)
                                                        # Build regression model based on M models from each Data Source
                                                        regr.fit(y_hat_train_m[0:t+1,:,m].T, train_yp.values.ravel())
                                                        # Save the model to list
                                                        s_modelA[t].append(pickle.dumps(regr)) 
                                                        
                                                        # y_hat_train_m[t,:,m] = regr.predict(y_hat_train_m[0:t+1,:,m].T)
                                                        # y_hat_test_m[t,:,m] = regr.predict(y_hat_test_m[0:t+1,:,m].T)
                                                        
                                                        y_hat_train_m_f[t,:,m] = regr.predict(y_hat_train_m[0:t+1,:,m].T)
                                                        y_hat_test_m_f[t,:,m] = regr.predict(y_hat_test_m[0:t+1,:,m].T)
                                                    
                                                #---------------------------------------------------------------------------
                                                # Fusion the y_hat from different data source
                                        
                                                if fusion_2 == "Simple Mean":
                                                    y_hat_train[t,:] = np.sum(y_hat_train_m_f[t,:,:], axis = 1) / N
                                                    y_hat_test[t,:] = np.sum(y_hat_test_m_f[t,:,:], axis = 1) / len(vali_v)
                                                    
                                                elif fusion_2 == "Median M models":
                                                    #-------------------------------
                                                    # Different way to take median                                                
                                                    #Beta_temp = Beta[t,:] # Current iteration
                                                    
                                                    Beta_temp = Beta[0:t+1,:] # Overall iterations
                                                    Beta_temp = np.mean(Beta_temp, axis = 0)
                                                    #----------------------------------
                                                    WeightedMedian = np.concatenate((np.asmatrix(np.arange(len(Beta_temp))).T,np.asmatrix(Beta_temp).T), axis=1)
                                                    WeightedMedian = WeightedMedian[np.lexsort(WeightedMedian.T)]
                                                    temp = 0
                                                    for ii in range(0,len(Beta_temp)):
                                                        temp = temp + WeightedMedian[0,ii,1]
                                                        if temp >= np.sum(WeightedMedian[0,:,1],1)/2:
                                                            index = ii
                                                            iter_index = WeightedMedian[0,index,0]
                                                            break
                                                    #print("iter_index =", iter_index)
                                                    iter_index = int(iter_index)
                                        
                                                    #y_hat_train[t,:]= np.sum(y_hat_train_m_f[t,:,:], axis = 0)
                                                    y_hat_train[t,:]= y_hat_train_m_f[t,:,iter_index]
                                                    y_hat_test[t,:] = y_hat_test_m_f[t,:,iter_index]
                                        
                                                elif fusion_2 == "Mean M models":
                                                    Beta_temp = Beta[0:t+1,:] # Overall iterations
                                                    Beta_temp = np.mean(Beta_temp, axis = 0)
                                                    
                                                    for m in range(int(df_v_c[df_v_c.idxmin(axis=1)[0]][0])-1, (int(df_v_c[df_v_c.idxmax(axis=1)[0]][0]))):
                                                        train_w_re[t,:,m] = y_hat_train_m_f[t,:,m] * np.log(1/Beta_temp[m])
                                                        test_w_re[t,:,m] = y_hat_test_m_f[t,:,m] * np.log(1/Beta_temp[m])
                                                    
                                                    
                                                    Beta_log[t,:] = np.log(1/Beta_temp[:]) 
                                                    
                                                    y_hat_train[t,:]= np.sum(train_w_re[t,:,:], axis = 1) / np.sum(Beta_log[t,:], axis = 0)
                                                    y_hat_test[t,:] = np.sum(test_w_re[t,:,:], axis = 1) / np.sum(Beta_log[t,:], axis = 0)
                                                    
                                                elif fusion_2 == "Regreesion M models":
                                                    # Build regression model based on M models from each Data Source
                                                    regr.fit(y_hat_train_m_f[t,:,:], train_yp.values.ravel())
                                                    # Save the model to list
                                                    #s_modelA[t].append(pickle.dumps(regr)) 
                                                    
                                                    y_hat_train[t,:] = regr.predict(y_hat_train_m_f[t,:,:])
                                                    y_hat_test[t,:] = regr.predict(y_hat_test_m_f[t,:,:])
                                        
                                        
                                                # Calculate loss for each training example
                                                lA[t,:]  = abs(y_hat_train[t,:] - train_yp.values.ravel())
                                                DentA[t,0] = max(lA[t,:])       # Calculate Dent_t^m (the max error)
                                                for ii in range(0,N):           # Calculate Loss Function L_t^m(i)
                                                    LA[t,ii] = lA[t,ii]/DentA[t,0]     # Linear Loss Function
                                                # Calculate an average loss Lbar_t^m
                                                for ii in range(0,N):
                                                    LbarA[t,0]  = LbarA[t,0] + LA[t,ii] * wA[t,ii]    
                                                # Calculate Alaph
                                                Alaph[t,0] = LbarA[t,0] / (1 - LbarA[t,0])
                                                
                                                # MSE
                                                MSE_trreA[t,0] = mean_squared_error(train_yp.values.ravel(),y_hat_train[t,:]) 
                                                # print(MSE_trreA[t,0])
                                                # print(np.mean((y_hat_train[t,:] - train_yp.values.ravel())**2))
                                                
                                                MSE_tereA[t,0] = mean_squared_error(vali_y,y_hat_test[t,:]) 
                                                # print(MSE_tereA[t,0])
                                                # print(np.mean((y_hat_test[t,:] - vali_y.values.ravel())**2))
                                                                                        
                                                # mae
                                                mae_trreA[t,0] = mae(train_yp.values.ravel(),y_hat_train[t,:]) 
                                                # print(MSE_trreA[t,0])
                                                # print(np.mean((y_hat_train[t,:] - train_yp.values.ravel())**2))
                                                
                                                mae_tereA[t,0] = mae(vali_y.values.ravel(),y_hat_test[t,:]) 
                                                # print(MSE_tereA[t,0])
                                                # print(np.mean((y_hat_test[t,:] - vali_y.values.ravel())**2))
                                                
                                                # R2
                                                actual = train_yp.values.ravel()
                                                predict = y_hat_train[t,:]     
                                                # corr_matrix = np.corrcoef(actual, predict)
                                                # corr = corr_matrix[0,1]
                                                # R_sq = corr**2
                                        
                                                R2_trreA[t,0] = rsquare(actual, predict)
                                                
                                                actual = vali_y.values.ravel()
                                                predict = y_hat_test[t,:]     
                                                # corr_matrix = np.corrcoef(actual, predict)
                                                # corr = corr_matrix[0,1]
                                                # R_sq = corr**2
                                                
                                                R2_tereA[t,0] = rsquare(actual, predict)                
                                        
                                                # Update Weight for the sampled Observation
                                                wA[t+1,:] = wA[t,:]
                                                w[t+1,:,:] = w[t,:,:]
                                                for ii in range(0,N):
                                                    wA[t+1,ii] = wA[t,ii] * (Alaph[t,0] ** (1-LA[t,ii]))
                                                    for mm in range(0,S):
                                                        w[t+1,ii,mm] = w[t,ii,mm] * ((1 - Gamma) * (Beta[t,mm] ** (1-L[t,ii,mm])) + Gamma * (Alaph[t,0] ** (1-LA[t,ii])))
                                                # Normalization factor
                                                Z_A[t,0] = sum(wA[t+1,:])
                                                for mm in range(0,S):
                                                    Z[t,mm] = sum(w[t+1,:,mm])
                                                for ii in range(0,N):
                                                    wA[t+1,ii] = wA[t+1,ii] / Z_A[t,0]
                                                    for mm in range(0,S):
                                                        w[t+1,ii,mm] = w[t+1,ii,mm] / Z[t,mm] 
                        
                                            result_proposed[index1, index2, index3, index4, index5] = MSE_tereA[t,0]
                        # # 计算第一维的平均值生成四维数组
                        # result_proposed_ave = np.mean(result_proposed, axis=0)
                        
                        # # 获取数组的形状
                        # shape_result = result_proposed_ave.shape
                        
                        # # 创建一个 Pandas Excel writer 使用 xlsxwriter 作为引擎
                        # excel_filename = "Proposed_CV_result.xlsx"
                        # with pd.ExcelWriter(excel_filename, engine='xlsxwriter') as excel_writer:
                        #     for i in range(len(list_max_depth)):  # 遍历第四维
                        #         sheet_name = f'max_depth_{list_max_depth[i]}'
                        #         temp = result_proposed_ave[:, :, :, i]
                        #         # 获取数组的形状
                        #         shape = temp.shape                                
                        #         # 初始化一个空数组，用于存储重新排列后的结果
                        #         reshaped_array = np.zeros((shape[0] * shape[2], shape[1]))
                                
                        #         # 使用循环将切片重新排列
                        #         for i in range(shape[2]):
                        #             reshaped_array[i*shape[0]:(i+1)*shape[0], :] = temp[:,:,i]
                        #         df_sheet = pd.DataFrame(reshaped_array)
                        #         df_sheet.to_excel(excel_writer, sheet_name=sheet_name, index=False, header=False)

                        # 计算第一维的平均值生成四维数组
                        result_proposed_ave = np.mean(result_proposed, axis=0)
                        # 获取数组的形状
                        shape_result = result_proposed_ave.shape
                        # 创建一个 Pandas Excel writer 使用 xlsxwriter 作为引擎
                        excel_filename = "Proposed_CV_result.xlsx"
                        with pd.ExcelWriter(excel_filename, engine='xlsxwriter') as excel_writer:
                            for iii in range(len(list_max_depth)):  # 遍历第四维
                                sheet_name = f'max_depth_{list_max_depth[iii]}'
                                temp = result_proposed_ave[:, :, :, iii]
                                # 获取数组的形状
                                shape = temp.shape                                
                                # 将前两维变成一个一维数组
                                reshaped_array = temp.reshape((shape[0] * shape[1], shape[2]))
                                    
                                df_sheet = pd.DataFrame(reshaped_array.T)
                                df_sheet.to_excel(excel_writer, sheet_name=sheet_name, index=False, header=False)

                        # 找到最小元素的四维位置
                    min_indices = np.unravel_index(np.argmin(result_proposed_ave), result_proposed_ave.shape)
                                        
                    fusion_1 = Fusion_1_list[list_m1[min_indices[0]]]
                    fusion_2 = Fusion_2_list[list_m2[min_indices[1]]]                
                    Gamma = list_gamma[min_indices[2]]
                    max_depth = list_max_depth[min_indices[3]]                     
                    with open("parameters_proposed.txt", "w",encoding="utf-8") as file_proposed:
                        # 创建保存print内容的字符串
                        output_string = f"最小元素的三维位置: {min_indices}\n"
                        output_string += f"fusion_1: {fusion_1}\n"
                        output_string += f"fusion_2: {fusion_2}\n"
                        output_string += f"Gamma: {Gamma}\n"
                        output_string += f"max_depth: {max_depth}\n"
                        
                        # 打印输出字符串
                        print(output_string)
                                                               
                        # 将输出字符串按行写入文件
                        file_proposed.writelines(output_string)
                                        
                    N = len(train_yA)
                    mse_trreA_values_list = []
                    mse_tereA_values_list = []
                    
                    # Weight of data sources and overall
                    w = np.ones((T,N,S))/N
                    wA = np.ones((T,N))/N
                    # loss of data sources and overall
                    l = np.zeros((T,N,S))
                    lA = np.zeros((T,N))
                    # Loss function of data sources and overall
                    L = np.zeros((T,N,S))
                    LA = np.zeros((T,N))
                
                    Dent = np.zeros((T,S))
                    DentA = np.zeros((T,1))
                
                    Lbar = np.zeros((T,S))
                    LbarA = np.zeros((T,1))
                
                    Beta = np.zeros((T,S))
                    Alaph = np.zeros((T,1))
                
                    Beta_sum = np.zeros((T,S))
                    Beta_log = np.zeros((T,S)) 
                
                    Z = np.zeros((T,S))
                    Z_A = np.zeros((T,1))
                
                    y_hat_train_m = np.zeros((T,N,S))
                    y_hat_test_m = np.zeros((T,len(test_v),S))
                    
                    y_hat_train_m_f = np.zeros((T,N,S))
                    y_hat_test_m_f = np.zeros((T,len(test_v),S))
                
                
                    y_hat_train = np.zeros((T,N))
                    y_hat_test = np.zeros((T,len(test_v)))
                
                    s_model = [[] for j in range(0,T)]
                    s_modelA = [[] for j in range(0,T)]
                
                    train_re = np.zeros((T,N,S))
                    test_re = np.zeros((T,len(test_v),S))
                
                    train_w_re = np.zeros((T,N,S))
                    test_w_re = np.zeros((T,len(test_v),S))
                
                    train_w_re_sum = np.zeros((T,N,S))
                    test_w_re_sum = np.zeros((T,len(test_v),S))
                
                    MSE_trre = np.zeros((T,S))
                    MSE_tere = np.zeros((T,S))
                    # R2
                    R2_trre = np.zeros((T,S))
                    R2_tere = np.zeros((T,S))
                
                    # Training result & Testing result
                    # train_reA = np.zeros((T,N))
                    # test_reA = np.zeros((T,len(test_v)))
                
                    # Weighted Result
                    train_w_reA = np.zeros((T,N))
                    test_w_reA = np.zeros((T,len(test_v)))
                
                    # MSE
                    MSE_trreA = np.zeros((T,1))
                    MSE_tereA = np.zeros((T,1))
                    # mae
                    mae_trreA = np.zeros((T,1))
                    mae_tereA = np.zeros((T,1))
                    # R-square
                    R2_trreA = np.zeros((T,1))
                    R2_tereA = np.zeros((T,1))
                    
                    # Initialize lists to accumulate data
                    t_values = list(range(T-1))
                    mse_trreA_values = []
                    mse_tereA_values = []
                    
                    #----------------------------------------------------------------
                    # Regression Model
                    regr = linear_model.LinearRegression()
                    # Create linear regression object
                    if method == "MLR":
                        regr = linear_model.LinearRegression()
                    else:
                        regr =  tree.DecisionTreeRegressor(max_depth=max_depth)                    
                    for t in range(0,T-1):
                        for m in range(int(df_v_c[df_v_c.idxmin(axis=1)[0]][0])-1, (int(df_v_c[df_v_c.idxmax(axis=1)[0]][0]))):
                            idx_m = [i for i, x in enumerate(var_name) if x == m + 1]   # Find the Index of Data Source m
                            if t == 0:            
                                train_m = train_vA.iloc[:, idx_m]    # Training data for Source m
                                test_m_origi = test_v.iloc[:, idx_m]     # Testing data for Source m            
                                train_m_origi = train_m
                                train_yAm = train_yA                            
                                #----------------------------------------------
                                # Train the model using the training sets
                                regr.fit(train_m, train_yA.values.ravel())            
                            else:
                                train_m_origi = train_vA.iloc[:, idx_m] 
                                test_m_origi = test_v.iloc[:, idx_m]     # Original Testing data for Source m
    
                                # Sampling based on Weight
                                x = np.arange(len(train_yA))
                                rate = w[t,:,m]
                                # Check if rate contains NaN values
                                if np.any(np.isnan(rate)):
                                    rate = np.ones(len(rate)) * (1 / len(rate))
                                    w[t,:,m] = rate
                                temp = np.random.choice(a=x, size=len(train_yA), replace=True, p=rate)
                                temp = temp.tolist()
                                #-------------------------------------------------------------
                                # Original Training data for Source m
                                train_m_origi = train_m_origi.to_numpy()
                                train_yAtemp = train_yA.to_numpy()
                                # Sampling based on Weight
                                train_m = train_m_origi[temp[:],:]
                                train_yAm = train_yAtemp[temp[:]]
                                
                                regr.fit(train_m, train_yAm)
                            # Save the model to list
                            s_model[t].append(pickle.dumps(regr))                      
                            # Calculate loss for each training example
                            l[t,:,m]  = abs(regr.predict(train_m_origi) - train_yA.values.ravel())
                            Dent[t,m] = max(l[t,:,m])       # Calculate Dent_t^m (the max error)
                            for ii in range(0,N):           # Calculate Loss Function L_t^m(i)
                                if Dent[t, m] != 0:
                                    L[t, ii, m] = l[t,ii,m]/Dent[t,m]     # Linear Loss Function
                                else:
                                    L[t, ii, m] = 1e-8# Calculate an average loss Lbar_t^m
                            for ii in range(0,N):
                                Lbar[t,m]  = Lbar[t,m] + L[t,ii,m] * w[t,ii,m]                         
                            # Calculate Beta
                            Beta[t,m] = Lbar[t,m] / (1 - Lbar[t,m])
                            
                            # Fusion the y_hat_m from t different iterations for data source m
                            if fusion_1 == "Single iteration":
                                # Generate y_hat_m for All Source based on the models.
                                y_hat_train_m[t,:,m] = regr.predict(train_m_origi)
                                y_hat_test_m[t,:,m] = regr.predict(test_m_origi)
                                
                                y_hat_train_m_f[t,:,m] = regr.predict(train_m_origi)
                                y_hat_test_m_f[t,:,m] = regr.predict(test_m_origi)
                            elif fusion_1 == "Median t iteration":
                                y_hat_train_m[t,:,m] = regr.predict(train_m_origi)
                                y_hat_test_m[t,:,m] = regr.predict(test_m_origi)
                                
                                Beta_temp = Beta[0:t+1,m]
                                WeightedMedian = np.concatenate((np.asmatrix(np.arange(len(Beta_temp))).T,np.asmatrix(Beta_temp).T), axis=1)
                                WeightedMedian = WeightedMedian[np.lexsort(WeightedMedian.T)]
                                temp = 0
                                for ii in range(0,len(Beta_temp)):
                                    temp = temp + WeightedMedian[0,ii,1]
                                    if temp >= np.sum(WeightedMedian[0,:,1],1)/2:
                                        index = ii
                                        iter_index = WeightedMedian[0,index,0]
                                        break
                                iter_index = int(iter_index)
                                #Check if Model is Saved
                                regr_temp = pickle.loads(s_model[iter_index][m])
                                y_hat_train_m_f[t,:,m] = y_hat_train_m[iter_index,:,m]
                                y_hat_test_m_f[t,:,m] = y_hat_test_m[iter_index,:,m]
                                
                            elif fusion_1 == "Mean t iteration":
                                y_hat_train_m[t,:,m] = regr.predict(train_m_origi)
                                y_hat_test_m[t,:,m] = regr.predict(test_m_origi)
                                # Weighted Result
                                train_w_re[t,:,m] = y_hat_train_m[t,:,m] * np.log(1/Beta[t,m])
                                test_w_re[t,:,m] = y_hat_test_m[t,:,m] * np.log(1/Beta[t,m])
                                
                                if t == 0:
                                    Beta_sum[t,m] = np.log(1/Beta[t,m])                
                                    train_w_re_sum[t,:,m] = train_w_re[t,:,m]              
                                    test_w_re_sum[t,:,m] = test_w_re[t,:,m]                
                                else:
                                    Beta_sum[t,m] = Beta_sum[t-1,m] + np.log(1/Beta[t,m])                
                                    train_w_re_sum[t,:,m] = train_w_re_sum[t-1,:,m] + train_w_re[t,:,m]
                                    test_w_re_sum[t,:,m] = test_w_re_sum[t-1,:,m] + test_w_re[t,:,m]
                                
                                y_hat_train_m_f[t,:,m]= train_w_re_sum[t,:,m] / Beta_sum[t,m]
                                y_hat_test_m_f[t,:,m]= test_w_re_sum[t,:,m] / Beta_sum[t,m]
                                
                            elif fusion_1 == "Regreesion t iteration":
                                y_hat_train_m[t,:,m] = regr.predict(train_m_origi)
                                y_hat_test_m[t,:,m] = regr.predict(test_m_origi)
                                # Build regression model based on M models from each Data Source
                                regr.fit(y_hat_train_m[0:t+1,:,m].T, train_yA.values.ravel())
                                # Save the model to list
                                s_modelA[t].append(pickle.dumps(regr)) 
                                
                                y_hat_train_m_f[t,:,m] = regr.predict(y_hat_train_m[0:t+1,:,m].T)
                                y_hat_test_m_f[t,:,m] = regr.predict(y_hat_test_m[0:t+1,:,m].T)
                        #---------------------------------------------------------------------------
                        # Fusion the y_hat from different data source
                
                        if fusion_2 == "Simple Mean":
                            y_hat_train[t,:] = np.sum(y_hat_train_m_f[t,:,:], axis = 1) / N
                            y_hat_test[t,:] = np.sum(y_hat_test_m_f[t,:,:], axis = 1) / len(test_v)
                            
                        elif fusion_2 == "Median M models":
                            Beta_temp = Beta[t,:]
                            WeightedMedian = np.concatenate((np.asmatrix(np.arange(len(Beta_temp))).T,np.asmatrix(Beta_temp).T), axis=1)
                            WeightedMedian = WeightedMedian[np.lexsort(WeightedMedian.T)]
                            temp = 0
                            for ii in range(0,len(Beta_temp)):
                                temp = temp + WeightedMedian[0,ii,1]
                                if temp >= np.sum(WeightedMedian[0,:,1],1)/2:
                                    index = ii
                                    iter_index = WeightedMedian[0,index,0]
                                    break
                            #print("iter_index =", iter_index)
                            iter_index = int(iter_index)
                
                            #y_hat_train[t,:]= np.sum(y_hat_train_m_f[t,:,:], axis = 0)
                            y_hat_train[t,:]= y_hat_train_m_f[t,:,iter_index]
                            y_hat_test[t,:] = y_hat_test_m_f[t,:,iter_index]
                
                        elif fusion_2 == "Mean M models":                    
                            Beta_temp = Beta[0:t+1,:] # Overall iterations
                            Beta_temp = np.mean(Beta_temp, axis = 0)
                            for m in range(int(df_v_c[df_v_c.idxmin(axis=1)[0]][0])-1, (int(df_v_c[df_v_c.idxmax(axis=1)[0]][0]))):
                                train_w_re[t,:,m] = y_hat_train_m_f[t,:,m] * np.log(1/Beta_temp[m])
                                test_w_re[t,:,m] = y_hat_test_m_f[t,:,m] * np.log(1/Beta_temp[m])
                            
                            Beta_log[t,:] = np.log(1/Beta_temp[:]) 
                            
                            y_hat_train[t,:]= np.sum(train_w_re[t,:,:], axis = 1) / np.sum(Beta_log[t,:], axis = 0)
                            y_hat_test[t,:] = np.sum(test_w_re[t,:,:], axis = 1) / np.sum(Beta_log[t,:], axis = 0)
                            
                        elif fusion_2 == "Regreesion M models":
                            # Build regression model based on M models from each Data Source
                            regr.fit(y_hat_train_m_f[t,:,:], train_yA.values.ravel())
                            # Save the model to list
                            #s_modelA[t].append(pickle.dumps(regr)) 
                            
                            y_hat_train[t,:] = regr.predict(y_hat_train_m_f[t,:,:])
                            y_hat_test[t,:] = regr.predict(y_hat_test_m_f[t,:,:])
                
                
                        # Calculate loss for each training example
                        lA[t,:]  = abs(y_hat_train[t,:] - train_yA.values.ravel())
                        DentA[t,0] = max(lA[t,:])       # Calculate Dent_t^m (the max error)
                        for ii in range(0,N):           # Calculate Loss Function L_t^m(i)
                            LA[t,ii] = lA[t,ii]/DentA[t,0]     # Linear Loss Function
                        # Calculate an average loss Lbar_t^m
                        for ii in range(0,N):
                            LbarA[t,0]  = LbarA[t,0] + LA[t,ii] * wA[t,ii]    
                        # Calculate Alaph
                        Alaph[t,0] = LbarA[t,0] / (1 - LbarA[t,0])
                        
                        # MSE
                        MSE_trreA[t,0] = mean_squared_error(train_yA.values.ravel(),y_hat_train[t,:]) 
                        # print(MSE_trreA[t,0])
                        # print(np.mean((y_hat_train[t,:] - train_yA.values.ravel())**2))
                        
                        MSE_tereA[t,0] = mean_squared_error(test_y,y_hat_test[t,:]) 
                        # print(MSE_tereA[t,0])
                        # print(np.mean((y_hat_test[t,:] - test_y.values.ravel())**2))
                        
                        # mae
                        mae_trreA[t,0] = mae(train_yA.values.ravel(),y_hat_train[t,:]) 
                        # print(MSE_trreA[t,0])
                        # print(np.mean((y_hat_train[t,:] - train_yA.values.ravel())**2))
                        
                        mae_tereA[t,0] = mae(test_y.values.ravel(),y_hat_test[t,:]) 
                        # print(MSE_tereA[t,0])
                        # print(np.mean((y_hat_test[t,:] - test_y.values.ravel())**2))
                        
                        # R2
                        actual_train = train_yA.values.ravel()
                        predict_train = y_hat_train[t,:]     
                        corr_matrix = np.corrcoef(actual_train, predict_train)
                        corr = corr_matrix[0,1]
                        R_sq = corr**2
                
                        R2_trreA[t,0] = R_sq
                        
                        actual_te = test_y.values.ravel()
                        predict_te = y_hat_test[t,:]     
                        corr_matrix = np.corrcoef(actual_te, predict_te)
                        corr = corr_matrix[0,1]
                        R_sq = corr**2
                        
                        R2_tereA[t,0] = R_sq
                
                        # Update Weight for the sampled Observation
                        wA[t+1,:] = wA[t,:]
                        w[t+1,:,:] = w[t,:,:]
                        for ii in range(0,N):
                            wA[t+1,ii] = wA[t,ii] * (Alaph[t,0] ** (1-LA[t,ii]))
                            for mm in range(0,S):
                                w[t+1,ii,mm] = w[t,ii,mm] * ((1 - Gamma) * (Beta[t,mm] ** (1-L[t,ii,mm])) + Gamma * (Alaph[t,0] ** (1-LA[t,ii])))
                        # Normalization factor
                        Z_A[t,0] = sum(wA[t+1,:])
                        for mm in range(0,S):
                            Z[t,mm] = sum(w[t+1,:,mm])
                        for ii in range(0,N):
                            wA[t+1,ii] = wA[t+1,ii] / Z_A[t,0]
                            for mm in range(0,S):
                                w[t+1,ii,mm] = w[t+1,ii,mm] / Z[t,mm]     
           
                        mse_trreA_values.append(MSE_trreA[t, :])
                        mse_tereA_values.append(MSE_tereA[t, :])
                    # Convert lists to arrays for plotting
                    mse_trreA_values = np.array(mse_trreA_values)
                    mse_tereA_values = np.array(mse_tereA_values)
                    
                    mse_trreA_values_list.append(mse_trreA_values)
                    mse_tereA_values_list.append(mse_tereA_values)
    
                    pred_ada_propose = predict
                                
                    results[i,0,k] = i
                    results[i,1,k] = MSE_trreA[t,0]
                    results[i,2,k] = MSE_tereA[t,0]
                    results[i,3,k] = mae_trreA[t,0]
                    results[i,4,k] = mae_tereA[t,0]
                    results[i,5,k] = R2_trreA[t,0]
                    results[i,6,k] = R2_tereA[t,0]
                    results[i,7,k] = mape(train_yA.values.ravel(), y_hat_train[t,:])
                    results[i,8,k] = mape(test_y.values.ravel(), y_hat_test[t,:])
                                   
                    actual_test[i,:,k] = test_y.values.ravel()
                    predict_test[i,:,k] = y_hat_test[t,:]
                else:
                    results[i,0,k] = i
                    results[i,1,k] = mean_squared_error(train_yA.values.ravel(), 
                                         np.ones(len(train_yA))*float(np.mean(train_yA.values.ravel())))
                    results[i,2,k] = mean_squared_error(test_y.values.ravel(), 
                                         np.ones(len(test_y))*float(np.mean(test_y.values.ravel())))
                    results[i,3,k] = mae(train_yA.values.ravel(), 
                                         np.ones(len(train_yA))*float(np.mean(train_yA.values.ravel())))
                    results[i,4,k] = mae(test_y.values.ravel(), 
                                         np.ones(len(test_y))*float(np.mean(test_y.values.ravel())))
                    results[i,5,k] = 0
                    
                    results[i,6,k] = 0
                    
                    results[i,7,k] = mape(train_yA.values.ravel(), 
                                         np.ones(len(train_yA))*float(np.mean(train_yA.values.ravel())))
                    results[i,8,k] = mape(test_y.values.ravel(), 
                                         np.ones(len(test_y))*float(np.mean(test_y.values.ravel())))
                    actual_test[i,:,k] = test_y.values.ravel()
                    predict_test[i,:,k] = np.ones(len(test_y))*float(np.mean(test_y.values.ravel()))
                    
                end_time = time.time()  # 记录结束时间
                elapsed_time = end_time - start_time  # 计算执行时间
                print(f"Time elapsed: {elapsed_time:.2f} seconds")  # 打印执行时间
                print('----------------------------------------------------------------')
            
        
        # Get the shape of the 3D array
        depth, rows, cols = results.shape
        
        # Create a Pandas Excel writer using xlsxwriter as the engine
        with pd.ExcelWriter('result_tensor_data.xlsx', engine='xlsxwriter') as excel_writer:
            # Loop through the 3D array and create Excel sheets for each 2D slice
            for c in range(cols):
                df_slice = pd.DataFrame(results[:, :, c])
                df_slice.to_excel(excel_writer, sheet_name=f'Slice_{c}', index=False)
        
        # Get the shape of the 3D array
        depth, rows, cols = predict_test.shape
        
        # Create a Pandas Excel writer using xlsxwriter as the engine
        with pd.ExcelWriter('predict_tensor_data.xlsx', engine='xlsxwriter') as excel_writer:
            # Loop through the 3D array and create Excel sheets for each 2D slice
            for c in range(cols):
                df_slice = pd.DataFrame(predict_test[:, :, c])
                df_slice.to_excel(excel_writer, sheet_name=f'Slice_{c}', index=False)
                
        # Create a Pandas Excel writer using xlsxwriter as the engine
        with pd.ExcelWriter('true_tensor_data.xlsx', engine='xlsxwriter') as excel_writer:
            # Loop through the 3D array and create Excel sheets for each 2D slice
            for c in range(cols):
                df_slice = pd.DataFrame(actual_test[:, :, c])
                df_slice.to_excel(excel_writer, sheet_name=f'Slice_{c}', index=False)

        
        # # Create a Pandas Excel writer using xlsxwriter as the engine
        # excel_writer = pd.ExcelWriter('result_tensor_data.xlsx', engine='xlsxwriter')
        
        # # Loop through the 3D array and create Excel sheets for each 2D slice
        # for c in range(cols):
        #     df_slice = pd.DataFrame(results[:, :, c])
        #     df_slice.to_excel(excel_writer, sheet_name=f'Slice_{c}', index=False)

        # # Save the Excel file
        # excel_writer.save()
        
        # # Get the shape of the 3D array
        # depth, rows, cols = predict_test.shape
        
        # # Create a Pandas Excel writer using xlsxwriter as the engine
        # excel_writer = pd.ExcelWriter('predict_tensor_data.xlsx', engine='xlsxwriter')
        
        # # Loop through the 3D array and create Excel sheets for each 2D slice
        # for c in range(cols):
        #     df_slice = pd.DataFrame(predict_test[:, :, c])
        #     df_slice.to_excel(excel_writer, sheet_name=f'Slice_{c}', index=False)

        # # Save the Excel file
        # excel_writer.save()

    # Calculate the average along the third dimension
    average_results = np.mean(results, axis=2)
    
    # Convert the average_results matrix to a DataFrame
    df = pd.DataFrame(average_results)
    
    # Define the filename for the Excel file
    filename = "average_results.xlsx"
    
    # Save the DataFrame to Excel
    df.to_excel(filename, index=False)  # Set index=False if you don't want to save row indices
    
