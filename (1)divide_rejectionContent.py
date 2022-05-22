import os
import pandas as pd
import re
from tqdm import tqdm


inputpath = 'D:/project/KPRIS_OA_1999_2020/(1)_1_Preprocessingdata/'
savepath = 'D:/project/KPRIS_OA_1999_2020/(1)_2_Preprocessingdata/'
files = os.listdir(inputpath)
files = tqdm(files)

for file in files:
    output = pd.DataFrame()
    raw = pd.read_csv(inputpath + file, encoding='cp949')

    # 결측치 제거
    raw = raw.loc[raw['rejectionContentDetail'].isnull() == False]

    # 중복제거
    print('{}에서 총 {}개의 중복 row가 제거됨'.format(file ,len(raw[raw.duplicated()])))
    raw = raw.drop_duplicates()

    # 특이 케이스 따로 처리
    raw['rejectionContentDetail'] = [i.replace('[거절이유 1] 및 [거절이유 2]는', '거절이유 1 및 거절이유 2는') for i in raw['rejectionContentDetail']]
    raw['rejectionContentDetail'] = [i.replace('[거절이유 1] 및 [거절이유 2]', '거절이유 1 및 거절이유 2') for i in raw['rejectionContentDetail']]
    raw['rejectionContentDetail'] = [i.replace('[거절이유 1]과', '거절이유 1과') for i in raw['rejectionContentDetail']]
    raw['rejectionContentDetail'] = [i.replace('[거절이유1]에', '거절이유 1에') for i in raw['rejectionContentDetail']]
    raw['rejectionContentDetail'] = [i.replace('[거절이유 1]에', '거절이유 1에') for i in raw['rejectionContentDetail']]
    raw['rejectionContentDetail'] = [i.replace('[ 거절이유 1 ]과', '거절이유 1과') for i in raw['rejectionContentDetail']]
    raw['rejectionContentDetail'] = [i.replace('[ 거절이유 1 ]에', '거절이유 1에') for i in raw['rejectionContentDetail']]
    raw['rejectionContentDetail'] = [i.replace('[ 거절이유 1 ]의', '거절이유 1의') for i in raw['rejectionContentDetail']]
    raw['rejectionContentDetail'] = [i.replace('[거절이유 2]는', '거절이유 2는') for i in raw['rejectionContentDetail']]
    raw['rejectionContentDetail'] = [i.replace('[ 거절이유 2 ]는', '거절이유 2는') for i in raw['rejectionContentDetail']]
    raw['rejectionContentDetail'] = [i.replace('[거절이유 1], [거절이유 2]는', '거절이유 1, 거절이유 2는') for i in raw['rejectionContentDetail']]
    raw['rejectionContentDetail'] = [i.replace('[ 거절이유 2 ], [ 거절이유 3 ]', '거절이유 2, 거절이유 3') for i in raw['rejectionContentDetail']]
    raw['rejectionContentDetail'] = [i.replace('[ 거절이유 2 ]와 [ 거절이유 3 ]은', '거절이유 2와 거절이유 3은') for i in raw['rejectionContentDetail']]
    raw['rejectionContentDetail'] = [i.replace('[ 거절이유 2 ], ', '거절이유 2, ') for i in raw['rejectionContentDetail']]
    raw['rejectionContentDetail'] = [i.replace('[거절이유 1]로', '거절이유 1로') for i in raw['rejectionContentDetail']]
    raw['rejectionContentDetail'] = [i.replace('[ 거절이유 1 ]로', '거절이유 1로') for i in raw['rejectionContentDetail']]
    raw['rejectionContentDetail'] = [i.replace('[거절이유 2]를', '거절이유 2를') for i in raw['rejectionContentDetail']]
    raw['rejectionContentDetail'] = [i.replace('동일한 [거절이유3] 타인', '동일한 거절이유3 타인') for i in raw['rejectionContentDetail']]
    raw['rejectionContentDetail'] = [i.replace('([거절이유1])로', '거절이유1로') for i in raw['rejectionContentDetail']]
    raw['rejectionContentDetail'] = [i.replace('[거절이유3]의', '거절이유3의') for i in raw['rejectionContentDetail']]
    raw['rejectionContentDetail'] = [i.replace('[ 거절이유 3]을', '거절이유 3을') for i in raw['rejectionContentDetail']]
    raw['rejectionContentDetail'] = [i.replace('[ 거절이유 3 ]에', '거절이유 3에') for i in raw['rejectionContentDetail']]
    raw['rejectionContentDetail'] = [i.replace('[ 거절이유 3 ]과 [ 거절이유 4 ]는', '거절이유 3과 거절이유 4는') for i in raw['rejectionContentDetail']]
    raw['rejectionContentDetail'] = [i.replace('[ 거절이유 3 ]과 [ 거절이유 4 ] 는', '거절이유 3과 거절이유 4는') for i in raw['rejectionContentDetail']]


    # 거절이유 문구 통일
    raw['rejectionContentDetail'] = [i.replace('[거절이유1]', '[거절이유]') for i in raw['rejectionContentDetail']]
    raw['rejectionContentDetail'] = [i.replace('[거절이유2]', '[거절이유]') for i in raw['rejectionContentDetail']]
    raw['rejectionContentDetail'] = [i.replace('[거절이유3]', '[거절이유]') for i in raw['rejectionContentDetail']]
    raw['rejectionContentDetail'] = [i.replace('[거절이유4]', '[거절이유]') for i in raw['rejectionContentDetail']]
    raw['rejectionContentDetail'] = [i.replace('[거절이유 1]', '[거절이유]') for i in raw['rejectionContentDetail']]
    raw['rejectionContentDetail'] = [i.replace('[거절이유 2]', '[거절이유]') for i in raw['rejectionContentDetail']]
    raw['rejectionContentDetail'] = [i.replace('[거절이유 3]', '[거절이유]') for i in raw['rejectionContentDetail']]
    raw['rejectionContentDetail'] = [i.replace('[거절이유 4]', '[거절이유]') for i in raw['rejectionContentDetail']]
    raw['rejectionContentDetail'] = [i.replace('[거절이유 1 ]', '[거절이유]') for i in raw['rejectionContentDetail']]
    raw['rejectionContentDetail'] = [i.replace('[거절이유 2 ]', '[거절이유]') for i in raw['rejectionContentDetail']]
    raw['rejectionContentDetail'] = [i.replace('[거절이유 3 ]', '[거절이유]') for i in raw['rejectionContentDetail']]
    raw['rejectionContentDetail'] = [i.replace('[거절이유 4 ]', '[거절이유]') for i in raw['rejectionContentDetail']]
    raw['rejectionContentDetail'] = [i.replace('[ 거절이유 ]', '[거절이유]') for i in raw['rejectionContentDetail']]
    raw['rejectionContentDetail'] = [i.replace('[ 거절이유]', '[거절이유]') for i in raw['rejectionContentDetail']]
    raw['rejectionContentDetail'] = [i.replace('[ 거절이유1]', '[거절이유]') for i in raw['rejectionContentDetail']]
    raw['rejectionContentDetail'] = [i.replace('[ 거절이유2]', '[거절이유]') for i in raw['rejectionContentDetail']]
    raw['rejectionContentDetail'] = [i.replace('[ 거절이유3]', '[거절이유]') for i in raw['rejectionContentDetail']]
    raw['rejectionContentDetail'] = [i.replace('[ 거절이유4]', '[거절이유]') for i in raw['rejectionContentDetail']]
    raw['rejectionContentDetail'] = [i.replace('[ 거절이유 1 ]', '[거절이유]') for i in raw['rejectionContentDetail']]
    raw['rejectionContentDetail'] = [i.replace('[ 거절이유 2 ]', '[거절이유]') for i in raw['rejectionContentDetail']]
    raw['rejectionContentDetail'] = [i.replace('[ 거절이유 3 ]', '[거절이유]') for i in raw['rejectionContentDetail']]
    raw['rejectionContentDetail'] = [i.replace('[ 거절이유 4 ]', '[거절이유]') for i in raw['rejectionContentDetail']]
    raw['rejectionContentDetail'] = [i.replace('[ 거절이유 5 ]', '[거절이유]') for i in raw['rejectionContentDetail']]
    raw['rejectionContentDetail'] = [i.replace('[ 거절이유 6 ]', '[거절이유]') for i in raw['rejectionContentDetail']]
    raw['rejectionContentDetail'] = [i.replace('[ 거절이유 7 ]', '[거절이유]') for i in raw['rejectionContentDetail']]
    raw['rejectionContentDetail'] = [i.replace('[ 거절이유 8 ]', '[거절이유]') for i in raw['rejectionContentDetail']]
    raw['rejectionContentDetail'] = [i.replace('[ 거절이유 9 ]', '[거절이유]') for i in raw['rejectionContentDetail']]
    raw['rejectionContentDetail'] = [i.replace('[ 거절이유 10 ]', '[거절이유]') for i in raw['rejectionContentDetail']]
    raw['rejectionContentDetail'] = [i.replace('[ 거절이유 11 ]', '[거절이유]') for i in raw['rejectionContentDetail']]
    raw['rejectionContentDetail'] = [i.replace('[ 거절이유 12 ]', '[거절이유]') for i in raw['rejectionContentDetail']]
    raw['rejectionContentDetail'] = [i.replace('[ 거절이유1 ]', '[거절이유]') for i in raw['rejectionContentDetail']]
    raw['rejectionContentDetail'] = [i.replace('[ 거절이유2 ]', '[거절이유]') for i in raw['rejectionContentDetail']]
    raw['rejectionContentDetail'] = [i.replace('[ 거절이유3 ]', '[거절이유]') for i in raw['rejectionContentDetail']]
    raw['rejectionContentDetail'] = [i.replace('[ 거절이유4 ]', '[거절이유]') for i in raw['rejectionContentDetail']]
    raw['rejectionContentDetail'] = [i.replace('[ 거절이유 1]', '[거절이유]') for i in raw['rejectionContentDetail']]
    raw['rejectionContentDetail'] = [i.replace('[ 거절이유 2]', '[거절이유]') for i in raw['rejectionContentDetail']]
    raw['rejectionContentDetail'] = [i.replace('[ 거절이유 3]', '[거절이유]') for i in raw['rejectionContentDetail']]
    raw['rejectionContentDetail'] = [i.replace('[ 거절이유 4]', '[거절이유]') for i in raw['rejectionContentDetail']]
    raw['rejectionContentDetail'] = [i.replace('[ 거절이유1,2 ]', '[거절이유]') for i in raw['rejectionContentDetail']]
    raw['rejectionContentDetail'] = [i.replace('[ 거절이유1~2 ]', '[거절이유]') for i in raw['rejectionContentDetail']]
    raw['rejectionContentDetail'] = [i.replace('[ 거절이유2 1 ]', '[거절이유]') for i in raw['rejectionContentDetail']]
    raw['rejectionContentDetail'] = [i.replace('[ 거절이유 1,2 ]', '[거절이유]') for i in raw['rejectionContentDetail']]
    raw['rejectionContentDetail'] = [i.replace('[ 거절이유 1, 2]', '[거절이유]') for i in raw['rejectionContentDetail']]
    raw['rejectionContentDetail'] = [i.replace('[ 거절이유1, 2 ]', '[거절이유]') for i in raw['rejectionContentDetail']]
    raw['rejectionContentDetail'] = [i.replace('[ 거절이유 1, 2 ]', '[거절이유]') for i in raw['rejectionContentDetail']]
    raw['rejectionContentDetail'] = [i.replace('[ 거절이유:1 ]', '[거절이유]') for i in raw['rejectionContentDetail']]
    raw['rejectionContentDetail'] = [i.replace('[ 거절이유:2 ]', '[거절이유]') for i in raw['rejectionContentDetail']]
    raw['rejectionContentDetail'] = [i.replace('[ 거절이유:3 ]', '[거절이유]') for i in raw['rejectionContentDetail']]
    raw['rejectionContentDetail'] = [i.replace('[ 거절이유:4 ]', '[거절이유]') for i in raw['rejectionContentDetail']]
    raw['rejectionContentDetail'] = [i.replace('[거절이유: 1]', '[거절이유]') for i in raw['rejectionContentDetail']]
    raw['rejectionContentDetail'] = [i.replace('[거절이유: 2]', '[거절이유]') for i in raw['rejectionContentDetail']]
    raw['rejectionContentDetail'] = [i.replace('[거절이유: 3]', '[거절이유]') for i in raw['rejectionContentDetail']]
    raw['rejectionContentDetail'] = [i.replace('[거절이유: 4]', '[거절이유]') for i in raw['rejectionContentDetail']]
    raw['rejectionContentDetail'] = [i.replace('[ 거절이유: 1]', '[거절이유]') for i in raw['rejectionContentDetail']]
    raw['rejectionContentDetail'] = [i.replace('[ 거절이유: 2]', '[거절이유]') for i in raw['rejectionContentDetail']]
    raw['rejectionContentDetail'] = [i.replace('[ 거절이유: 3]', '[거절이유]') for i in raw['rejectionContentDetail']]
    raw['rejectionContentDetail'] = [i.replace('[ 거절이유: 4]', '[거절이유]') for i in raw['rejectionContentDetail']]
    raw['rejectionContentDetail'] = [i.replace('[ 거절이유: 1 ]', '[거절이유]') for i in raw['rejectionContentDetail']]
    raw['rejectionContentDetail'] = [i.replace('[ 거절이유: 2 ]', '[거절이유]') for i in raw['rejectionContentDetail']]
    raw['rejectionContentDetail'] = [i.replace('[ 거절이유: 3 ]', '[거절이유]') for i in raw['rejectionContentDetail']]
    raw['rejectionContentDetail'] = [i.replace('[ 거절이유: 4 ]', '[거절이유]') for i in raw['rejectionContentDetail']]
    raw['rejectionContentDetail'] = [i.replace('[ 거절이유 1. ]', '[거절이유]') for i in raw['rejectionContentDetail']]
    raw['rejectionContentDetail'] = [i.replace('[ 거절이유 2. ]', '[거절이유]') for i in raw['rejectionContentDetail']]
    raw['rejectionContentDetail'] = [i.replace('[ 거절이유 3. ]', '[거절이유]') for i in raw['rejectionContentDetail']]
    raw['rejectionContentDetail'] = [i.replace('[거절이유 1.]', '[거절이유]') for i in raw['rejectionContentDetail']]
    raw['rejectionContentDetail'] = [i.replace('[거절이유 2.]', '[거절이유]') for i in raw['rejectionContentDetail']]
    raw['rejectionContentDetail'] = [i.replace('[거절이유 3.]', '[거절이유]') for i in raw['rejectionContentDetail']]
    raw['rejectionContentDetail'] = [i.replace('[ 거절이유 1.]', '[거절이유]') for i in raw['rejectionContentDetail']]
    raw['rejectionContentDetail'] = [i.replace('[ 거절이유 2.]', '[거절이유]') for i in raw['rejectionContentDetail']]
    raw['rejectionContentDetail'] = [i.replace('[ 거절이유 3.]', '[거절이유]') for i in raw['rejectionContentDetail']]
    raw['rejectionContentDetail'] = [i.replace('[ 거절이유 (1) ]', '[거절이유]') for i in raw['rejectionContentDetail']]
    raw['rejectionContentDetail'] = [i.replace('[ 거절이유 (2) ]', '[거절이유]') for i in raw['rejectionContentDetail']]
    raw['rejectionContentDetail'] = [i.replace('[ 거절이유 (3) ]', '[거절이유]') for i in raw['rejectionContentDetail']]
    raw['rejectionContentDetail'] = [i.replace('[ 거절이유 (1)]', '[거절이유]') for i in raw['rejectionContentDetail']]
    raw['rejectionContentDetail'] = [i.replace('[ 거절이유 (2)]', '[거절이유]') for i in raw['rejectionContentDetail']]
    raw['rejectionContentDetail'] = [i.replace('[ 거절이유 (3)]', '[거절이유]') for i in raw['rejectionContentDetail']]
    raw['rejectionContentDetail'] = [i.replace('[ 거절이유 ①]', '[거절이유]') for i in raw['rejectionContentDetail']]
    raw['rejectionContentDetail'] = [i.replace('[ 거절이유 ②]', '[거절이유]') for i in raw['rejectionContentDetail']]
    raw['rejectionContentDetail'] = [i.replace('[거절이유 Ⅰ]', '[거절이유]') for i in raw['rejectionContentDetail']]
    raw['rejectionContentDetail'] = [i.replace('[거절이유 Ⅱ]', '[거절이유]') for i in raw['rejectionContentDetail']]
    raw['rejectionContentDetail'] = [i.replace('[거절이유 Ⅲ]', '[거절이유]') for i in raw['rejectionContentDetail']]
    raw['rejectionContentDetail'] = [i.replace('[거절이유 Ⅰ ]', '[거절이유]') for i in raw['rejectionContentDetail']]
    raw['rejectionContentDetail'] = [i.replace('[거절이유 Ⅱ ]', '[거절이유]') for i in raw['rejectionContentDetail']]
    raw['rejectionContentDetail'] = [i.replace('[거절이유 Ⅲ ]', '[거절이유]') for i in raw['rejectionContentDetail']]
    raw['rejectionContentDetail'] = [i.replace('[거절이유Ⅰ]', '[거절이유]') for i in raw['rejectionContentDetail']]
    raw['rejectionContentDetail'] = [i.replace('[거절이유Ⅱ]', '[거절이유]') for i in raw['rejectionContentDetail']]
    raw['rejectionContentDetail'] = [i.replace('[거절이유Ⅲ]', '[거절이유]') for i in raw['rejectionContentDetail']]
    raw['rejectionContentDetail'] = [i.replace('【 거절이유 】', '[거절이유]') for i in raw['rejectionContentDetail']]
    raw['rejectionContentDetail'] = [i.replace('[ 거절이유; 2 ]', '[거절이유]') for i in raw['rejectionContentDetail']]

    # 거절이유 split
    for i, string in enumerate(raw['rejectionContentDetail']):
        row = raw.iloc[i,:]
        iter_num = len(re.findall('\[거절이유]', string))

        idxs = []
        for match in re.finditer('\[거절이유]', string):
            idxs.append(match.start())

        if len(idxs) == 0:
            # 그대로 쌓음
            output = output.append([raw.iloc[i,:]])
        elif len(idxs) == 1:
            # 그대로 쌓음
            output = output.append([raw.iloc[i,:]])
        elif len(idxs) == 2:
            row1 = row.copy()
            row2 = row.copy()
            row1['rejectionContentDetail'] = string[idxs[0]:idxs[1]]
            row2['rejectionContentDetail'] = string[idxs[1]:]
            output = output.append([row1, row2])
        elif len(idxs) == 3:
            row1 = row.copy()
            row2 = row.copy()
            row3 = row.copy()
            row1['rejectionContentDetail'] = string[idxs[0]:idxs[1]]
            row2['rejectionContentDetail'] = string[idxs[1]:idxs[2]]
            row3['rejectionContentDetail'] = string[idxs[2]:]
            output = output.append([row1, row2, row3])
        elif len(idxs) == 4:
            row1 = row.copy()
            row2 = row.copy()
            row3 = row.copy()
            row4 = row.copy()
            row1['rejectionContentDetail'] = string[idxs[0]:idxs[1]]
            row2['rejectionContentDetail'] = string[idxs[1]:idxs[2]]
            row3['rejectionContentDetail'] = string[idxs[2]:idxs[3]]
            row4['rejectionContentDetail'] = string[idxs[3]:]
            output = output.append([row1, row2, row3, row4])
        elif len(idxs) == 5:
            row1 = row.copy()
            row2 = row.copy()
            row3 = row.copy()
            row4 = row.copy()
            row5 = row.copy()
            row1['rejectionContentDetail'] = string[idxs[0]:idxs[1]]
            row2['rejectionContentDetail'] = string[idxs[1]:idxs[2]]
            row3['rejectionContentDetail'] = string[idxs[2]:idxs[3]]
            row4['rejectionContentDetail'] = string[idxs[3]:idxs[4]]
            row5['rejectionContentDetail'] = string[idxs[4]:]
            output = output.append([row1, row2, row3, row4, row5])
        elif len(idxs) == 6:
            row1 = row.copy()
            row2 = row.copy()
            row3 = row.copy()
            row4 = row.copy()
            row5 = row.copy()
            row6 = row.copy()
            row1['rejectionContentDetail'] = string[idxs[0]:idxs[1]]
            row2['rejectionContentDetail'] = string[idxs[1]:idxs[2]]
            row3['rejectionContentDetail'] = string[idxs[2]:idxs[3]]
            row4['rejectionContentDetail'] = string[idxs[3]:idxs[4]]
            row5['rejectionContentDetail'] = string[idxs[4]:idxs[5]]
            row6['rejectionContentDetail'] = string[idxs[5]:]
            output = output.append([row1, row2, row3, row4, row5, row6])
        elif len(idxs) == 7:
            row1 = row.copy()
            row2 = row.copy()
            row3 = row.copy()
            row4 = row.copy()
            row5 = row.copy()
            row6 = row.copy()
            row7 = row.copy()
            row1['rejectionContentDetail'] = string[idxs[0]:idxs[1]]
            row2['rejectionContentDetail'] = string[idxs[1]:idxs[2]]
            row3['rejectionContentDetail'] = string[idxs[2]:idxs[3]]
            row4['rejectionContentDetail'] = string[idxs[3]:idxs[4]]
            row5['rejectionContentDetail'] = string[idxs[4]:idxs[5]]
            row6['rejectionContentDetail'] = string[idxs[5]:idxs[6]]
            row7['rejectionContentDetail'] = string[idxs[6]:]
            output = output.append([row1, row2, row3, row4, row5, row6, row7])
        elif len(idxs) == 8:
            row1 = row.copy()
            row2 = row.copy()
            row3 = row.copy()
            row4 = row.copy()
            row5 = row.copy()
            row6 = row.copy()
            row7 = row.copy()
            row8 = row.copy()
            row1['rejectionContentDetail'] = string[idxs[0]:idxs[1]]
            row2['rejectionContentDetail'] = string[idxs[1]:idxs[2]]
            row3['rejectionContentDetail'] = string[idxs[2]:idxs[3]]
            row4['rejectionContentDetail'] = string[idxs[3]:idxs[4]]
            row5['rejectionContentDetail'] = string[idxs[4]:idxs[5]]
            row6['rejectionContentDetail'] = string[idxs[5]:idxs[6]]
            row7['rejectionContentDetail'] = string[idxs[6]:idxs[7]]
            row8['rejectionContentDetail'] = string[idxs[7]:]
            output = output.append([row1, row2, row3, row4, row5, row6, row7, row8])
        elif len(idxs) == 9:
            row1 = row.copy()
            row2 = row.copy()
            row3 = row.copy()
            row4 = row.copy()
            row5 = row.copy()
            row6 = row.copy()
            row7 = row.copy()
            row8 = row.copy()
            row9 = row.copy()
            row1['rejectionContentDetail'] = string[idxs[0]:idxs[1]]
            row2['rejectionContentDetail'] = string[idxs[1]:idxs[2]]
            row3['rejectionContentDetail'] = string[idxs[2]:idxs[3]]
            row4['rejectionContentDetail'] = string[idxs[3]:idxs[4]]
            row5['rejectionContentDetail'] = string[idxs[4]:idxs[5]]
            row6['rejectionContentDetail'] = string[idxs[5]:idxs[6]]
            row7['rejectionContentDetail'] = string[idxs[6]:idxs[7]]
            row8['rejectionContentDetail'] = string[idxs[7]:idxs[8]]
            row9['rejectionContentDetail'] = string[idxs[8]:]
            output = output.append([row1, row2, row3, row4, row5, row6, row7, row8, row9])
        elif len(idxs) == 10:
            row1 = row.copy()
            row2 = row.copy()
            row3 = row.copy()
            row4 = row.copy()
            row5 = row.copy()
            row6 = row.copy()
            row7 = row.copy()
            row8 = row.copy()
            row9 = row.copy()
            row10 = row.copy()
            row1['rejectionContentDetail'] = string[idxs[0]:idxs[1]]
            row2['rejectionContentDetail'] = string[idxs[1]:idxs[2]]
            row3['rejectionContentDetail'] = string[idxs[2]:idxs[3]]
            row4['rejectionContentDetail'] = string[idxs[3]:idxs[4]]
            row5['rejectionContentDetail'] = string[idxs[4]:idxs[5]]
            row6['rejectionContentDetail'] = string[idxs[5]:idxs[6]]
            row7['rejectionContentDetail'] = string[idxs[6]:idxs[7]]
            row8['rejectionContentDetail'] = string[idxs[7]:idxs[8]]
            row9['rejectionContentDetail'] = string[idxs[8]:idxs[9]]
            row10['rejectionContentDetail'] = string[idxs[9]:]
            output = output.append([row1, row2, row3, row4, row5, row6, row7, row8, row9, row10])
        elif len(idxs) == 11:
            row1 = row.copy()
            row2 = row.copy()
            row3 = row.copy()
            row4 = row.copy()
            row5 = row.copy()
            row6 = row.copy()
            row7 = row.copy()
            row8 = row.copy()
            row9 = row.copy()
            row10 = row.copy()
            row11 = row.copy()
            row1['rejectionContentDetail'] = string[idxs[0]:idxs[1]]
            row2['rejectionContentDetail'] = string[idxs[1]:idxs[2]]
            row3['rejectionContentDetail'] = string[idxs[2]:idxs[3]]
            row4['rejectionContentDetail'] = string[idxs[3]:idxs[4]]
            row5['rejectionContentDetail'] = string[idxs[4]:idxs[5]]
            row6['rejectionContentDetail'] = string[idxs[5]:idxs[6]]
            row7['rejectionContentDetail'] = string[idxs[6]:idxs[7]]
            row8['rejectionContentDetail'] = string[idxs[7]:idxs[8]]
            row9['rejectionContentDetail'] = string[idxs[8]:idxs[9]]
            row10['rejectionContentDetail'] = string[idxs[9]:idxs[10]]
            row11['rejectionContentDetail'] = string[idxs[10]:]
            output = output.append([row1, row2, row3, row4, row5, row6, row7, row8, row9, row10, row11])
        elif len(idxs) == 12:
            row1 = row.copy()
            row2 = row.copy()
            row3 = row.copy()
            row4 = row.copy()
            row5 = row.copy()
            row6 = row.copy()
            row7 = row.copy()
            row8 = row.copy()
            row9 = row.copy()
            row10 = row.copy()
            row11 = row.copy()
            row12 = row.copy()
            row1['rejectionContentDetail'] = string[idxs[0]:idxs[1]]
            row2['rejectionContentDetail'] = string[idxs[1]:idxs[2]]
            row3['rejectionContentDetail'] = string[idxs[2]:idxs[3]]
            row4['rejectionContentDetail'] = string[idxs[3]:idxs[4]]
            row5['rejectionContentDetail'] = string[idxs[4]:idxs[5]]
            row6['rejectionContentDetail'] = string[idxs[5]:idxs[6]]
            row7['rejectionContentDetail'] = string[idxs[6]:idxs[7]]
            row8['rejectionContentDetail'] = string[idxs[7]:idxs[8]]
            row9['rejectionContentDetail'] = string[idxs[8]:idxs[9]]
            row10['rejectionContentDetail'] = string[idxs[9]:idxs[10]]
            row11['rejectionContentDetail'] = string[idxs[10]:idxs[11]]
            row12['rejectionContentDetail'] = string[idxs[11]:]
            output = output.append([row1, row2, row3, row4, row5, row6, row7, row8, row9, row10, row11, row12])

    # 저장
    output.to_csv(savepath + file, index=False, encoding='cp949')
    # print(output)