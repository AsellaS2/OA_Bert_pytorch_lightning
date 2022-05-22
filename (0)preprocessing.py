import os
import pandas as pd
import re
from tqdm import tqdm

inputpath = 'D:/project/OA_paper/DATA/OArawdata/'
savepath = 'D:/project/OA_paper/DATA/Preprocessing/'
files = os.listdir(inputpath)

files = tqdm(files)

for file in files:
    print(file)
    output = pd.DataFrame()
    raw = pd.read_csv(inputpath + file, encoding='cp949')

    if 'page_num' in raw.columns:
        raw = raw.drop(['page_num'], axis=1)

    # 결측치 제거
    raw = raw.loc[raw['rejectionContentDetail'].isnull() == False]

    # 전처리1 : html문 제거 및 대치
    raw['rejectionContentDetail'] = [i.replace('&', '') for i in raw['rejectionContentDetail']]
    raw['rejectionContentDetail'] = [i.replace('amp;', '&') for i in raw['rejectionContentDetail']]
    raw['rejectionContentDetail'] = [i.replace('quot;', '"') for i in raw['rejectionContentDetail']]
    raw['rejectionContentDetail'] = [i.replace('lt;', '<') for i in raw['rejectionContentDetail']]
    raw['rejectionContentDetail'] = [i.replace('gt;', '>') for i in raw['rejectionContentDetail']]
    raw['rejectionContentDetail'] = [i.replace('middot;', '·') for i in raw['rejectionContentDetail']]
    raw['rejectionContentDetail'] = [i.replace('<p>', '') for i in raw['rejectionContentDetail']]
    raw['rejectionContentDetail'] = [i.replace('</p>', '') for i in raw['rejectionContentDetail']]
    raw['rejectionContentDetail'] = [i.replace('<BR>', '') for i in raw['rejectionContentDetail']]
    raw['rejectionContentDetail'] = [i.replace('</BR>', '') for i in raw['rejectionContentDetail']]
    raw['rejectionContentDetail'] = [i.replace('<br>', '') for i in raw['rejectionContentDetail']]
    raw['rejectionContentDetail'] = [i.replace('</br>', '') for i in raw['rejectionContentDetail']]
    raw['rejectionContentDetail'] = [i.replace('nbsp;', '') for i in raw['rejectionContentDetail']]
    raw['rejectionContentDetail'] = [i.replace('<tbody>', '') for i in raw['rejectionContentDetail']]
    raw['rejectionContentDetail'] = [i.replace('</tbody>', '') for i in raw['rejectionContentDetail']]
    raw['rejectionContentDetail'] = [i.replace('<tr>', '') for i in raw['rejectionContentDetail']]
    raw['rejectionContentDetail'] = [i.replace('</tr>', '') for i in raw['rejectionContentDetail']]
    raw['rejectionContentDetail'] = [i.replace('<table style="\'border-collapse:collapse;\'">', '') for i in raw['rejectionContentDetail']]
    raw['rejectionContentDetail'] = [i.replace('</table>', '') for i in raw['rejectionContentDetail']]
    raw['rejectionContentDetail'] = [i.replace('</td>', '') for i in raw['rejectionContentDetail']]

    # 전처리2 : 표장 메타 데이터 제거
    for i, row in enumerate(raw['rejectionContentDetail']):
        # print('3: {}/{}'.format(i, len(raw)))
        # print(row)
        while True:
            if row.find('<emi file=') != -1:
                if row.find('/>') != -1:
                    if row.find('<emi file=') - row.find('/>') < 0:
                        idx_from = row.find('<emi file=')
                        idx_to = row.find('/>') + 2
                        target = row[idx_from:idx_to]
                        row = row.replace(target, '')
                        raw.iloc[i, 3] = row
                    elif row.find('<emi file=') - row.find('/>') > 0:
                        idx_from = row.find('/>')
                        idx_to = row.find('/>') + 2
                        target = row[idx_from:idx_to]
                        front = row[:idx_from]
                        back = row[idx_to:]
                        row = front + back
                        raw.iloc[i, 3] = row
                elif row.find('</emi>') != -1:
                    idx_from = row.find('<emi file=')
                    idx_to = row.find('</emi>') + 6
                    target = row[idx_from:idx_to]
                    row = row.replace(target, '')
                    raw.iloc[i, 3] = row
            elif row.find('<emi file=') == -1:
                break

    # 전처리3 : html 확장 table 제거
    for j, row in enumerate(raw['rejectionContentDetail']):
        # print('4: {}/{}'.format(j, len(raw)))
        # print(row)
        while True:
            if row.find('<td colspan=') != -1 and row.find('pt">') != -1:
                idx_from = row.find('<td colspan=')
                idx_to = row.find('pt">') + 4
                target = row[idx_from:idx_to]
                row = row.replace(target, '')
                raw.iloc[j, 3] = row
            elif row.find('<td colspan=') == -1:
                break

    # 중복 체크 및 제거
    print('{}에서 총 {}개의 중복 row가 제거됨'.format(file ,len(raw[raw.duplicated()])))
    raw = raw.drop_duplicates()

    # 전처리 결과 저장
    raw.to_csv(savepath + file, index=False, encoding='cp949')

print("Finish!")
