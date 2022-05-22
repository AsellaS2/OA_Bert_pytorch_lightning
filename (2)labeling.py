import os
import pandas as pd
import re


inputpath = 'D:/project/OA_paper/DATA/Preprocessing/'
savepath = 'D:/project/OA_paper/DATA/Labeling/'
files = os.listdir(inputpath)

for file in files[8:]:
    print(file)
    output = pd.DataFrame()
    raw = pd.read_csv(inputpath + file, encoding='cp949')

    # 결측치 제거
    raw = raw.loc[raw['rejectionContentDetail'].isnull() == False]


    # 분류를 위한 컬럼 생성
    raw['발음유사'] = ''
    raw['관념유사'] = ''
    raw['외관유사'] = ''
    raw['식별력'] = ''
    raw['상품 불명확'] = ''
    raw['label'] = ''

    contents = raw['rejectionContentDetail']

    # 거절사유 분류
    for i, content in enumerate(contents):
        # 거절사유 "발음유사"
        if content.find('제7조 제1항 제7호') > content.find('칭호') > 0:
            raw.iloc[i, 4] = 1
        if content.find('제8조') > content.find('칭호') > 0:
            raw.iloc[i, 4] = 1
        if content.find('제34조 제1항 제7호') > content.find('칭호') > 0:
            raw.iloc[i, 4] = 1
        if content.find('제35조') > content.find('칭호') > 0:
            raw.iloc[i, 4] = 1

        # 거절사유 "관념유사"
        if content.find('제7조 제1항 제7호') > content.find('관념') > 0:
            raw.iloc[i, 5] = 1
        if content.find('제8조') > content.find('관념') > 0:
            raw.iloc[i, 5] = 1
        if content.find('제34조 제1항 제7호') > content.find('관념') > 0:
            raw.iloc[i, 5] = 1
        if content.find('제35조') > content.find('관념') > 0:
            raw.iloc[i, 5] = 1

        # 거절사유 "외관유사":
        if content.find('외관') > 0:
            raw.iloc[i, 6] = 1

        # 거절사유 "식별력":
        if content.find('제6조') > 0:
            raw.iloc[i, 7] = 1
        if content.find('제33조') > 0:
            raw.iloc[i, 7] = 1

        # 거절사유 "상품 불명확":
        if content.find('제10조') > 0:
            raw.iloc[i, 8] = 1
        if content.find('제38조') > 0:
            raw.iloc[i, 8] = 1
        if content.find('상품 불명확') > 0:
            raw.iloc[i, 8] = 1
        if content.find('불명확한 지정상품') > 0:
            raw.iloc[i, 8] = 1

    print('start labeling...')
    # 라벨링
    for i in range(len(raw)):
        if raw.iloc[i, 4] == 1:
            raw.iloc[i, 9] = 1
        elif raw.iloc[i, 5] == 1:
            raw.iloc[i, 9] = 1
        elif raw.iloc[i, 6] == 1:
            raw.iloc[i, 9] = 1
        elif raw.iloc[i, 7] == 1:
            raw.iloc[i, 9] = 1
        elif raw.iloc[i, 8] == 1:
            raw.iloc[i, 9] = 1
        else:
            raw.iloc[i, 9] = 2

    output = pd.concat([output, raw])

    # 분류결과 저장
    output.to_csv(savepath + file, index=False, encoding='cp949')