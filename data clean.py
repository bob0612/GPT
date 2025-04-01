import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# 读取Excel数据并仅跳过最后一列
file_path = 'biomass all.xlsx'
data = pd.read_excel(file_path)
last_column = data.iloc[:, -1].copy()  # 保存最后一列
data = data.iloc[:, :-1]  # 去除最后一列，仅保留中间列进行处理

# 保留原始的Name列
Name_column = data['Name'].copy()

# 将分类变量转换为数值类型（独热编码）
encoded_data = pd.get_dummies(data, columns=['Name'], drop_first=True)

# 使用随机森林填补空值的函数
def fill_na_with_rf(df, target_column):
    # 将不包含缺失值的数据分为特征和目标
    df_notna = df[df[target_column].notna()]
    X = df_notna.drop(columns=[target_column])
    y = df_notna[target_column]

    # 分离含有缺失值的数据
    df_na = df[df[target_column].isna()]
    X_na = df_na.drop(columns=[target_column])

    # 如果X_na为空，则表示该列无缺失值，跳过
    if X_na.empty:
        return df[target_column]

    # 训练随机森林模型
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)

    # 预测缺失值并填充
    df.loc[df[target_column].isna(), target_column] = rf.predict(X_na)
    return df[target_column]

# 对每一列应用RF填充
for col in encoded_data.columns:
    if encoded_data[col].isna().any():  # 只处理存在缺失值的列
        encoded_data[col] = fill_na_with_rf(encoded_data, col)

# 将保留的原始Name列和最后的last_column添加回数据框
encoded_data['Name'] = Name_column
encoded_data['doi'] = last_column  # 将最后一列重新加入数据框

# 按原始顺序排列
cleaned_data = encoded_data.loc[:, list(data.columns) + ['doi']]

# 保存去掉最后一列的版本 (wash)
cleaned_data_no_doi = cleaned_data.drop(columns=['doi'])
output_file_path_no_doi = 'wash.xlsx'
cleaned_data_no_doi.to_excel(output_file_path_no_doi, index=False)

# 输出完整的清洗后数据到另一个Excel文件 (washdoi)
output_file_path_doi = 'washdoi.xlsx'
cleaned_data.to_excel(output_file_path_doi, index=False)

print(f"数据清洗完成，已生成 {output_file_path_no_doi} 和 {output_file_path_doi}")
# 确认清理后的数据量
print("清理后的数据样本数量:", data.shape[0])

