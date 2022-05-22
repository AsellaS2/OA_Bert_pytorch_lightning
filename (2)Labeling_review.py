import os
import pandas as pd

PATH = 'D:/project/OA_paper/DATA/Etc/'


labeling_etc = pd.read_csv(PATH + 'labeling_etc.csv', encoding='cp949')
labeling_1 = pd.read_csv(PATH + 'labeling_1.csv', encoding='cp949')

# 컬럼 추가
labeling_1['추가된 컬럼명'] = ''

labeling_etc['추가된 컬럼명'] = ''
labeling_etc['label'] = ''

contents = labeling_etc['rejectionContentDetail']

# 거절사유 분류
for i, content in enumerate(contents):

    # 거절사유 "발음유사"
    if content.find('제7조 제1항 제7호') > content.find('칭호') > 0:
        labeling_etc.loc[i, '발음유사'] = 1
    if content.find('제8조') > content.find('칭호') > 0:
        labeling_etc.loc[i, '발음유사'] = 1
    if content.find('제34조 제1항 제7호') > content.find('칭호') > 0:
        labeling_etc.loc[i, '발음유사'] = 1
    if content.find('제35조') > content.find('칭호') > 0:
        labeling_etc.loc[i, '발음유사'] = 1

    # 거절사유 "관념유사"
    if content.find('제7조 제1항 제7호') > content.find('관념') > 0:
        labeling_etc.loc[i, '관념유사'] = 1
    if content.find('제8조') > content.find('관념') > 0:
        labeling_etc.loc[i, '관념유사'] = 1
    if content.find('제34조 제1항 제7호') > content.find('관념') > 0:
        labeling_etc.loc[i, '관념유사'] = 1
    if content.find('제35조') > content.find('관념') > 0:
        labeling_etc.loc[i, '관념유사'] = 1

    # 거절사유 "외관유사"
    if content.find('외관') > 0:
        labeling_etc.loc[i, '외관유사'] = 1

    # 거절사유 "식별력":
    if content.find('제6조') > 0:
        labeling_etc.loc[i, '식별력'] = 1
    if content.find('제33조') > 0:
        labeling_etc.loc[i, '식별력'] = 1
    if content.find('식별력') > 0:
        labeling_etc.loc[i, '식별력'] = 1

    # 거절사유 "상품 불명확":
    if content.find('제10조') > 0:
        labeling_etc.loc[i, '상품 불명확'] = 1
    if content.find('제38조') > 0:
        labeling_etc.loc[i, '상품 불명확'] = 1
    if content.find('상품불명확') > 0:
        labeling_etc.loc[i, '상품 불명확'] = 1
    if content.find('불명확한 지정상품') > 0:
        labeling_etc.loc[i, '상품 불명확'] = 1

    # 거절사유 "품질오인(기타)":
    if content.find('품질을 오인혼동') > 0:
        labeling_etc.loc[i, '품질오인(기타)'] = 1

    # 거절사유 "공공질서(기타)":
    if content.find('공공의 질서 또는 선량한 풍속') > 0:
        labeling_etc.loc[i, '공공질서(기타)'] = 1

# 재라벨링
for i in range(len(labeling_etc)):
    if labeling_etc.loc[i, '발음유사'] == 1:
        labeling_etc.loc[i, 'label'] = 1
    elif labeling_etc.loc[i, '관념유사'] == 1:
        labeling_etc.loc[i, 'label'] = 1
    elif labeling_etc.loc[i, '외관유사'] == 1:
        labeling_etc.loc[i, 'label'] = 1
    elif labeling_etc.loc[i, '식별력'] == 1:
        labeling_etc.loc[i, 'label'] = 1
    elif labeling_etc.loc[i, '상품 불명확'] == 1:
        labeling_etc.loc[i, 'label'] = 1
    elif labeling_etc.loc[i, '품질오인(기타)'] == 1:
        labeling_etc.loc[i, 'label'] = 1
    elif labeling_etc.loc[i, '공공질서(기타)'] == 1:
        labeling_etc.loc[i, 'label'] = 1
    else:
        labeling_etc.loc[i, 'label'] = 2

Mo_labeling_1 = labeling_etc[labeling_etc['label'].isin([1])]
new_labeling_etc = labeling_etc[labeling_etc['label'].isin([2])]

labeling_1 = pd.concat([labeling_1, Mo_labeling_1])
print('{}건 수정됨'.format(len(Mo_labeling_1)))

print(Mo_labeling_1)
print(labeling_1)

# 수정결과 저장
# labeling_1.to_csv(PATH + 'labeling_1.csv', index=False, encoding='cp949')
# new_labeling_etc.to_csv(PATH + 'labeling_etc.csv', index=False, encoding='cp949')
