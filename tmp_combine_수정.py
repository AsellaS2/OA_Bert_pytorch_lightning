import os
import pandas as pd
from tqdm import tqdm


inputpath = 'D:/project/OA_paper/DATA/Etc/1/col_del/'
savepath = 'D:/project/OA_paper/DATA/combine/'


output = pd.DataFrame()
raw = pd.read_csv(inputpath + 'labeling_F.csv', encoding='cp949')

patent_id = set(raw['patent_id'].tolist())

output = pd.DataFrame(columns=['patent_id', 'rejectionContentDetail',
                               '발음유사', '관념유사', '외관유사', '식별력', '상품 불명확', '기타'])

for i in enumerate(patent_id):

    target = raw.loc[raw.patent_id == i[1]]

    combine_contents = ''
    col1 = ''  # 발음유사
    col2 = ''  # 관념유사
    col3 = ''  # 외관유사
    col4 = ''  # 식별력
    col5 = ''  # 상품 불명확
    col6 = ''  # 기타

    for j, row in enumerate(target):

        contents = target.loc[j, 'rejectionContentDetail']
        combine_contents = combine_contents + ' ' + contents

        if target.loc[j, '발음유사'] == 1:
            col1 = 1
        elif target.loc[j, '관념유사'] == 1:
            col2 = 1
        elif target.loc[j, '외관유사'] == 1:
            col3 = 1
        elif target.loc[j, '식별력'] == 1:
            col4 = 1
        elif target.loc[j, '상품 불명확'] == 1:
            col5 = 1

        elif target.loc[j, '품질오인(기타)'] == 1:
            col6 = 1
        elif target.loc[j, '공공질서(기타)'] == 1:
            col6 = 1
        elif target.loc[j, '상품류(기타)'] == 1:
            col6 = 1
        elif target.loc[j, '개단(기타)'] == 1:
            col6 = 1
        elif target.loc[j, '주소상이(기타)'] == 1:
            col6 = 1
        elif target.loc[j, '사용의사(기타)'] == 1:
            col6 = 1
        elif target.loc[j, '단체표장(기타)'] == 1:
            col6 = 1
        elif target.loc[j, '업무표장(기타)'] == 1:
            col6 = 1
        elif target.loc[j, '정의위배(기타)'] == 1:
            col6 = 1
        elif target.loc[j, '저명한성명포함(기타)'] == 1:
            col6 = 1
        elif target.loc[j, '수요자기만(기타)'] == 1:
            col6 = 1
        elif target.loc[j, '부정한목적(기타)'] == 1:
            col6 = 1
        elif target.loc[j, '저명한국제기관(기타)'] == 1:
            col6 = 1
        elif target.loc[j, '입체적형상(기타)'] == 1:
            col6 = 1
        elif target.loc[j, '포도주산지(기타)'] == 1:
            col6 = 1
        elif target.loc[j, '저명한업무(기타)'] == 1:
            col6 = 1
        elif target.loc[j, '소멸후3년내출원(기타)'] == 1:
            col6 = 1
        elif target.loc[j, '취소심판청구인외의출원(기타)'] == 1:
            col6 = 1
        elif target.loc[j, '상표유형오기재(기타)'] == 1:
            col6 = 1


    combine = pd.DataFrame([[target.loc[0, 'patent_id'], combine_contents,
                             col1, col2, col3, col4, col5, col6]],
                           columns=['patent_id', 'rejectionContentDetail',
                                    '발음유사', '관념유사', '외관유사', '식별력', '상품 불명확', '기타'])

    output = output.append(combine, ignore_index=True)

output.to_csv(savepath + 'labeling_F2.csv', index=False, encoding='cp949')