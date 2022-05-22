import pandas as pd
import os
import sklearn


PATH = 'D:/project/OA_paper/DATA/Part/'

train = pd.read_csv(PATH + 'Train.csv', encoding='cp949')
test = pd.read_csv(PATH + 'Test.csv', encoding='cp949')

# 섞기
train_shuffled = sklearn.utils.shuffle(train)
train_shuffled.reset_index(drop=True, inplace=True)

# set2
test2 = train_shuffled.loc[:148994, :]
train2 = test.append([train_shuffled.drop(test2.index)])

# set3
test3 = train_shuffled.loc[148995:297989, :]
train3 = test.append([train_shuffled.drop(test3.index)])

# set4
test4 = train_shuffled.loc[297990:446984, :]
train4 = test.append([train_shuffled.drop(test4.index)])

# set5
test5 = train_shuffled.loc[446985:, :]
train5 = test.append([train_shuffled.drop(test5.index)])


test2.to_csv(PATH + "test2.csv", index=False, encoding='cp949')
test3.to_csv(PATH + "test3.csv", index=False, encoding='cp949')
test4.to_csv(PATH + "test4.csv", index=False, encoding='cp949')
test5.to_csv(PATH + "test5.csv", index=False, encoding='cp949')

train2.to_csv(PATH + "train2.csv", index=False, encoding='cp949')
train3.to_csv(PATH + "train3.csv", index=False, encoding='cp949')
train4.to_csv(PATH + "train4.csv", index=False, encoding='cp949')
train5.to_csv(PATH + "train5.csv", index=False, encoding='cp949')