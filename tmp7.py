import pandas as pd
import os
import sklearn
from sklearn.preprocessing import MultiLabelBinarizer

PATH = 'D:/project/OA_paper/DATA/Part/k_fold/5/'

# train = pd.read_csv(PATH + 'Train4.csv', encoding='cp949')
test = pd.read_csv(PATH + 'Test5.csv', encoding='cp949')

test['label'] = [i.replace('발음유사', '칭호유사') for i in test['label']]

x = list(test['rejectionContentDetail'])
y = list(map(lambda x: x.split(', '), test['label']))

mlb = MultiLabelBinarizer()
yt = mlb.fit_transform(y)
print(mlb.classes_)
print(yt)

print(yt[0][0])

label0 = sum([0+int(i[0]) for i in yt])
label1 = sum([0+int(i[1]) for i in yt])
label2 = sum([0+int(i[2]) for i in yt])
label3 = sum([0+int(i[3]) for i in yt])
label4 = sum([0+int(i[4]) for i in yt])
label5 = sum([0+int(i[5]) for i in yt])

print(label0)
print(label1)
print(label2)
print(label3)
print(label4)
print(label5)