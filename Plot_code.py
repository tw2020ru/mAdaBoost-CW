import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

# 用于存储每张表的x[i,:]的值
x_matrix = None  # 初始化一个空矩阵

# 打开Excel文件
file_path = "true_tensor_data.xlsx"
xls = pd.ExcelFile(file_path)

# 遍历每个Sheet
for sheet_name in xls.sheet_names:
    # 如果Sheet名以'Slice_'开头
    if sheet_name.startswith("Slice_"):
        # 读取该Sheet的数据，假设第二行是x[i,:]的值
        df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
        x_i_values = df.iloc[1, :].values

        if x_matrix is None:
            # 如果x_matrix还没有初始化，使用x_i_values来初始化它
            x_matrix = x_i_values.reshape(1, -1)
        else:
            # 否则，将x_i_values添加为新的行
            x_matrix = np.vstack((x_matrix, x_i_values))

# 现在x_matrix是一个包含所有x[i,:]值的矩阵

# 用于存储每张表的数据
predict_list = []

# 打开Excel文件
file_path = "predict_tensor_data.xlsx"
xls = pd.ExcelFile(file_path)

# 遍历每个Sheet
for sheet_name in xls.sheet_names:
    # 如果Sheet名以'Slice_'开头
    if sheet_name.startswith("Slice_"):
        # 读取该Sheet的数据，去除第一行
        df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
        data = df.iloc[1:, :].values  # 去除第一行

        predict_list.append(data)

# 将数据列表转换为一个三维矩阵
# 假设所有表的数据具有相同的形状，可以使用np.stack来堆叠
if predict_list:
    predict_matrix = np.stack(predict_list, axis=0)
else:
    predict_matrix = None

# predict_matrix现在包含了一个三维矩阵，其中每张表的数据都是一个二维切片


# 设置全局字体大小
plt.rcParams.update({'font.size': 20})  # 将字体大小设置为14


# 创建一个与算法对应的标签列表
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

for j in range(0,10):        
    # 提取x和y数据
    x = x_matrix[j, :]
    y_values = predict_list[j]
    # 创建一个散点图
    plt.figure(figsize=(10, 10))
    
    # 绘制x和每个y值
    for i in range(1,9):
        plt.scatter(x, y_values[i,:], label=algorithms[i])
    
    # 设置图的标题和标签
    plt.title('Scatter Plot of true and predict values')
    plt.xlabel('true value')
    plt.ylabel('predict value')
    plt.legend()   

    # 创建文件名，将 j 包含在文件名中
    filename = f'All_plot_{j}.png'
    
    # 保存图像以超高分辨率
    plt.savefig(filename, dpi=1200, bbox_inches='tight')  # 600 DPI的分辨率
    # 显示图形
    plt.show()
    
    # 创建4行2列的子图排列，最后一个位置不显示图
    fig, axs = plt.subplots(2, 4, figsize=(24, 12))
    fig.suptitle('Comparison of Algorithms')
    
    # 需要绘制的y的索引
    y_indices = [1, 2, 3, 4, 5, 6, 7]
    
    # 绘制每对y值与y=8的比较
    for i, y_index in enumerate(y_indices):
        row = i // 4  # 计算子图的行索引
        col = i % 4   # 计算子图的列索引
        axs[row, col].scatter(x, y_values[y_index,:], label=algorithms[y_index])
        axs[row, col].scatter(x, y_values[8,:], label='mAdaBoost.CW', alpha=0.35)  # y=8的散点图，透明度降低以区分
        axs[row, col].set_title(f'{algorithms[y_index]} vs mAdaBoost.CW')
        axs[row, col].set_xlabel('true value')
        axs[row, col].set_ylabel('predict value')
        axs[row, col].legend(loc='lower right')  # 设置图例位置为右下角
        
        # 设置坐标轴范围为0到5
        axs[row, col].set_xlim(0, 5)
        axs[row, col].set_ylim(0, 5)

    
    # 删除最后一个子图（第四行第二列）
    fig.delaxes(axs[1, 3])
    
    # 调整子图之间的间距
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # 创建文件名，将 j 包含在文件名中
    filename = f'comparison_plot_{j}.png'
    
    # 保存图像以超高分辨率
    plt.savefig(filename, dpi=600, bbox_inches='tight')  # 600 DPI的分辨率

    # 显示图形
    plt.show()                                                  


