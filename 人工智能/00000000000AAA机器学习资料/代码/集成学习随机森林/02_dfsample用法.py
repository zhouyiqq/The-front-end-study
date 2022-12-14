import pandas as pd

df = pd.DataFrame([[1, 5.56], [2, 5.7], [3, 5.91], [4, 6.4], [5, 6.8],
                   [6, 7.05], [7, 8.9], [8, 8.7], [9, 9], [10, 9.05]], columns=['X', 'Y'])

dfsample = df.sample(frac=1.0, replace=False)
print(dfsample)
print(dfsample.shape)

df01 = dfsample.drop_duplicates()  ###去重
print(df01)
print(df01.shape)
