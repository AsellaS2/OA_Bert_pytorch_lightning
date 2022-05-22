import pandas as pd

pd.set_option('display.max_columns', 10)

# Train = pd.read_csv('D:/project/OA_paper/DATA/Part/Train.csv', encoding='cp949')
#
# label = Train['label']
#
# label1 = 0
# label2 = 0
# label3 = 0
# label4 = 0
# label5 = 0
# label6 = 0
#
# for line in label:
#     if '발음유사' in line:
#         label1 += 1
#     if '관념유사' in line:
#         label2 += 1
#     if '외관유사' in line:
#         label3 += 1
#     if '식별력' in line:
#         label4 += 1
#     if '상품 불명확' in line:
#         label5 += 1
#     if '기타' in line:
#         label6 += 1
#
# print('발음:', label1)
# print('관념:', label2)
# print('외관:', label3)
# print('식별력:', label4)
# print('불명확:', label5)
# print('기타:', label6)

inputpath = 'D:/project/OA_paper/DATA/Part/'
savepath = 'D:/project/OA_paper/output/'

df = pd.read_csv(inputpath + 'Train.csv', encoding='cp949')

df['label'] = [i.replace('발음유사', '칭호유사') for i in df['label']]

print(df.head(5))