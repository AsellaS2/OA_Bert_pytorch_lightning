import os
import pandas as pd

inputpath = 'B:/BIRCCLOUD/(4)논문/51. 상표거절의견 분류 모델/DATA/Combine/'
savepath = 'B:/BIRCCLOUD/(4)논문/51. 상표거절의견 분류 모델/DATA/Part/'

Label_0 = pd.read_csv(inputpath + 'labeling_merge.csv', encoding='cp949')

# Label_0 = pd.read_csv(inputpath + 'Label_0.csv', encoding='cp949')
# Label_1 = pd.read_csv(inputpath + 'Label_1.csv', encoding='cp949')
# Label_2 = pd.read_csv(inputpath + 'Label_2.csv', encoding='cp949')
output = pd.DataFrame()

# samp_0 = Label_0.sample(n=23200)
# samp_1 = Label_1.sample(n=23200)
# samp_2 = Label_2.sample(n=23200)

# Learning = samp_0.append([samp_1, samp_2])

# Partitioning
Train = Label_0.sample(frac=0.8)
Test = Label_0.drop(Train.index)

Train.to_csv(savepath + "Train.csv", index=False, encoding='cp949')
Test.to_csv(savepath + "Test.csv", index=False, encoding='cp949')

print(len(Train))
print(len(Test))