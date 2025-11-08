import pandas as pd

# 1. 读取两个 CSV 文件
df1 = pd.read_csv(r'D:\yolo改进合集\yolov11改进\ultralytics-yolo11-main1028\runs\train\exp\results.csv')
df2 = pd.read_csv(r'D:\yolo改进合集\实验数据\yolo12n-300-24\results.csv')

# 2. 合并两个 DataFrame（纵向堆叠）
merged = pd.concat([df1, df2], ignore_index=True)

# 3. 按第一列排序（假设第一列没有列名，或你知道列名）
# 方法 A：如果第一列是普通数据列（不是索引），用 iloc 或列名
merged_sorted = merged.sort_values(by=merged.columns[0], ascending=True)

# 4. 重置索引（可选）
merged_sorted = merged_sorted.reset_index(drop=True)

# 5. 保存结果
merged_sorted.to_csv(r'D:\yolo改进合集\实验数据\yolo12n-300-24\results1.csv', index=False)