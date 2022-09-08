import random
import pandas as pd

a = [["A", 10], ["B", 20], ["C", 30], ["D", 40]]
# 第二个元素越大，采到的概率越大，请采样1w次，然后统计频率
#
# ra=random.choices(population=["A",'B','C','D'],weights=[10,20,30,40],k=10000)
# print(pd.Series(ra).value_counts())
ra = random.choices(population=[i[0] for i in a], weights=[i[1] for i in a], k=10000)
print(pd.Series(ra).value_counts())
