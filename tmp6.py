# 리뷰어가 1/10 데이터 셋으로 실험한 결과 포함시켜달라해서 만드는 1/10 데이터 셋

import pandas as pd
import os
import sklearn


PATH = 'D:/project/OA_paper/DATA/Part/'

train = pd.read_csv(PATH + 'Train.csv', encoding='cp949')
test = pd.read_csv(PATH + 'Test.csv', encoding='cp949')

df = train.append([test])

# 섞기
df_shuffled = sklearn.utils.shuffle(df)
df_shuffled.reset_index(drop=True, inplace=True)

# 1/10 랜덤추출
df_sample = df_shuffled.sample(frac=0.1)
df_sample.reset_index(drop=True, inplace=True)

# Partitioning
Train = df_sample.sample(frac=0.8)
Test = df_sample.drop(Train.index)

# 저장
Train.to_csv(PATH + "Train_10per.csv", index=False, encoding='cp949')
Test.to_csv(PATH + "Test_10per.csv", index=False, encoding='cp949')