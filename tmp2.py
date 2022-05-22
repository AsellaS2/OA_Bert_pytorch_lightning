import os
import pandas as pd


file = pd.read_csv('D:/project/OA_paper/DATA/Etc/test/test_combine.csv', encoding='cp949')

file['label'] = ''

output = pd.DataFrame(columns=['patent_id', 'rejectionContentDetail', 'label'])


for i in range(len(file)):
    label = []
    if file.loc[i, '발음유사'] == 1:
        label.append('발음유사')
    if file.loc[i, '관념유사'] == 1:
        label.append('관념유사')
    if file.loc[i, '외관유사'] == 1:
        label.append('외관유사')
    if file.loc[i, '식별력'] == 1:
        label.append('식별력')
    if file.loc[i, '상품 불명확'] == 1:
        label.append('상품 불명확')
    if file.loc[i, '기타'] == 1:
        label.append('기타')

    output = output.append(pd.DataFrame([[file.loc[i, 'patent_id'], file.loc[i, 'rejectionContentDetail'], label]],
                           columns=['patent_id', 'rejectionContentDetail', 'label']))

output['label'] = output['label'].astype(str).str.strip('[|]')
output['label'] = [i.replace('\'', '') for i in output['label']]

output.to_csv('D:/project/OA_paper/DATA/Etc/test/test_labeling.csv', index=False, encoding='cp949')
