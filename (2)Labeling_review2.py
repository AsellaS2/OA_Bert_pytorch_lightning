import os
import pandas as pd
import numpy as np

PATH = 'B:/BIRCCLOUD/(4)논문/51. 상표거절의견 분류 모델/DATA/Labeling/'

labeling_etc = pd.read_csv(PATH + 'labeling_etc.csv', encoding='cp949')
labeling_1 = pd.read_csv(PATH + 'labeling_1.csv', encoding='cp949', dtype={'rejectionContentDetail': np.str, 'label': np.int})

# 컬럼 추가
labeling_1['개단(기타)'] = ''
labeling_etc['개단(기타)'] = ''
labeling_1['주소상이(기타)'] = ''
labeling_etc['주소상이(기타)'] = ''
labeling_1['사용의사(기타)'] = ''
labeling_etc['사용의사(기타)'] = ''
labeling_1['단체표장(기타)'] = ''
labeling_etc['단체표장(기타)'] = ''
labeling_1['업무표장(기타)'] = ''
labeling_etc['업무표장(기타)'] = ''
labeling_1['정의위배(기타)'] = ''
labeling_etc['정의위배(기타)'] = ''
labeling_1['저명한성명포함(기타)'] = ''
labeling_etc['저명한성명포함(기타)'] = ''
labeling_1['수요자기만(기타)'] = ''
labeling_etc['수요자기만(기타)'] = ''
labeling_1['부정한목적(기타)'] = ''
labeling_etc['부정한목적(기타)'] = ''
labeling_1['저명한국제기관(기타)'] = ''
labeling_etc['저명한국제기관(기타)'] = ''
labeling_1['입체적형상(기타)'] = ''
labeling_etc['입체적형상(기타)'] = ''
labeling_1['포도주산지(기타)'] = ''
labeling_etc['포도주산지(기타)'] = ''
labeling_1['저명한업무(기타)'] = ''
labeling_etc['저명한업무(기타)'] = ''
labeling_1['소멸후3년내출원(기타)'] = ''
labeling_etc['소멸후3년내출원(기타)'] = ''
labeling_1['취소심판청구인외의출원(기타)'] = ''
labeling_etc['취소심판청구인외의출원(기타)'] = ''
labeling_1['상표유형오기재(기타)'] = ''
labeling_etc['상표유형오기재(기타)'] = ''
labeling_etc['label'] = ''

contents = labeling_etc['rejectionContentDetail']

# 거절사유 분류
for i, content in enumerate(contents):

    # 거절사유 "발음유사"
    if content.find('제7조 제1항 제7호') > content.find('칭호') > 0:
        labeling_etc.loc[i, '발음유사'] = 1
    if content.find('제7조 제1항 제8호') > content.find('칭호') > 0:
        labeling_etc.loc[i, '발음유사'] = 1
    if content.find('제8조') > content.find('칭호') > 0:
        labeling_etc.loc[i, '발음유사'] = 1
    if content.find('제34조 제1항 제7호') > content.find('칭호') > 0:
        labeling_etc.loc[i, '발음유사'] = 1
    if content.find('제35조') > content.find('칭호') > 0:
        labeling_etc.loc[i, '발음유사'] = 1
    if content.find('칭호') > 0:
        labeling_etc.loc[i, '발음유사'] = 1
    if content.find('호칭') > 0:
        labeling_etc.loc[i, '발음유사'] = 1
    if content.find('일요부가 유사') > 0:
        labeling_etc.loc[i, '발음유사'] = 1
    if content.find('일요부가 동일 유사') > 0:
        labeling_etc.loc[i, '발음유사'] = 1
    if content.find('일요부가동일') > 0:
        labeling_etc.loc[i, '발음유사'] = 1
    if content.find('일요부가유사') > 0:
        labeling_etc.loc[i, '발음유사'] = 1
    if content.find('1요부가 동일 유사') > 0:
        labeling_etc.loc[i, '발음유사'] = 1
    if content.find('1요부가 유사') > 0:
        labeling_etc.loc[i, '발음유사'] = 1
    if content.find('요부가 동일 유사') > 0:
        labeling_etc.loc[i, '발음유사'] = 1
    if content.find('요부가 유사') > 0:
        labeling_etc.loc[i, '발음유사'] = 1
    if content.find('요부가 동일') > 0:
        labeling_etc.loc[i, '발음유사'] = 1
    if content.find('동일 유사한 상표') > 0:
        labeling_etc.loc[i, '발음유사'] = 1
    if content.find('과 유사') > 0:
        labeling_etc.loc[i, '발음유사'] = 1
    if content.find('와 유사') > 0:
        labeling_etc.loc[i, '발음유사'] = 1
    if content.find('동일 유사') > 0:
        labeling_etc.loc[i, '발음유사'] = 1
    if content.find('동일유사') > 0:
        labeling_etc.loc[i, '발음유사'] = 1
    if content.find('전체적으로 유사') > 0:
        labeling_etc.loc[i, '발음유사'] = 1
    if content.find('동일,유사') > 0:
        labeling_etc.loc[i, '발음유사'] = 1
    if content.find('동일, 유사') > 0:
        labeling_etc.loc[i, '발음유사'] = 1
    if content.find('동일.유사') > 0:
        labeling_etc.loc[i, '발음유사'] = 1
    if content.find('동일. 유사') > 0:
        labeling_etc.loc[i, '발음유사'] = 1
    if content.find('동일 또는 유사') > 0:
        labeling_etc.loc[i, '발음유사'] = 1
    if content.find('동일하므로') > 0:
        labeling_etc.loc[i, '발음유사'] = 1
    if content.find('동일 하므로') > 0:
        labeling_etc.loc[i, '발음유사'] = 1
    if content.find('유사') > 0:
        labeling_etc.loc[i, '발음유사'] = 1
    if content.find('동일') > 0:
        labeling_etc.loc[i, '발음유사'] = 1
    if content.find('제7조 제1항 제7호') > 0:
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
    if content.find('관념') > 0:
        labeling_etc.loc[i, '관념유사'] = 1

    # 거절사유 "외관유사"
    if content.find('외관') > 0:
        labeling_etc.loc[i, '외관유사'] = 1
    if content.find('도형이 유사') > 0:
        labeling_etc.loc[i, '외관유사'] = 1
    if content.find('도형') > content.find('유사') > 0:
        labeling_etc.loc[i, '외관유사'] = 1
    if content.find('(도형)') > content.find('유사하므로') > 0:
        labeling_etc.loc[i, '외관유사'] = 1
    if content.find('(도형,') > content.find('유사') > 0:
        labeling_etc.loc[i, '외관유사'] = 1
    if content.find('도형') > 0:
        labeling_etc.loc[i, '외관유사'] = 1

    # 거절사유 "식별력":
    if content.find('제6조') > 0:
        labeling_etc.loc[i, '식별력'] = 1
    if content.find('제33조') > 0:
        labeling_etc.loc[i, '식별력'] = 1
    if content.find('식별력') > 0:
        labeling_etc.loc[i, '식별력'] = 1
    if content.find('식별') > 0:
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
    if content.find('품질을 오인') > 0:
        labeling_etc.loc[i, '품질오인(기타)'] = 1
    if content.find('품질 오인') > 0:
        labeling_etc.loc[i, '품질오인(기타)'] = 1
    if content.find('품질오인') > 0:
        labeling_etc.loc[i, '품질오인(기타)'] = 1
    if content.find('상품의 품질') > 0:
        labeling_etc.loc[i, '품질오인(기타)'] = 1
    if content.find('상품품질의 오인') > 0:
        labeling_etc.loc[i, '품질오인(기타)'] = 1
    if content.find('상품품질에 대하여 오인') > 0:
        labeling_etc.loc[i, '품질오인(기타)'] = 1
    if content.find('상품품질 및 출처의 오인') > 0:
        labeling_etc.loc[i, '품질오인(기타)'] = 1
    if content.find('상품 자체를 오인') > 0:
        labeling_etc.loc[i, '품질오인(기타)'] = 1
    if content.find('상품자체를 오인') > 0:
        labeling_etc.loc[i, '품질오인(기타)'] = 1
    if content.find('오인') > 0:
        labeling_etc.loc[i, '품질오인(기타)'] = 1
    if content.find('혼동') > 0:
        labeling_etc.loc[i, '품질오인(기타)'] = 1

    # 거절사유 "공공질서(기타)":
    if content.find('공공의 질서 또는 선량한 풍속') > 0:
        labeling_etc.loc[i, '공공질서(기타)'] = 1
    if content.find('공공질서') > 0:
        labeling_etc.loc[i, '공공질서(기타)'] = 1
    if content.find('공중도덕감정') > 0:
        labeling_etc.loc[i, '공공질서(기타)'] = 1
    if content.find('선량한 풍속') > 0:
        labeling_etc.loc[i, '공공질서(기타)'] = 1
    if content.find('신의 성실의 원칙') > 0:
        labeling_etc.loc[i, '공공질서(기타)'] = 1
    if content.find('신의성실의 원칙') > 0:
        labeling_etc.loc[i, '공공질서(기타)'] = 1
    if content.find('사회통념') > 0:
        labeling_etc.loc[i, '공공질서(기타)'] = 1
    if content.find('공공기관') > 0:
        labeling_etc.loc[i, '공공질서(기타)'] = 1
    if content.find('캐릭터로 상표법') > 0:
        labeling_etc.loc[i, '공공질서(기타)'] = 1
    if content.find('비방') > 0:
        labeling_etc.loc[i, '공공질서(기타)'] = 1
    if content.find('저명한 타인의') > 0:
        labeling_etc.loc[i, '공공질서(기타)'] = 1
    if content.find('전속계약서') > 0:
        labeling_etc.loc[i, '공공질서(기타)'] = 1
    if content.find('전속계약') > 0:
        labeling_etc.loc[i, '공공질서(기타)'] = 1
    if content.find('제7조 제1항 제2호') > 0:
        labeling_etc.loc[i, '공공질서(기타)'] = 1
    if content.find('제34조 제1항 제2호') > 0:
        labeling_etc.loc[i, '공공질서(기타)'] = 1
    if content.find('제7조 제1항 제4호') > 0:
        labeling_etc.loc[i, '공공질서(기타)'] = 1
    if content.find('제34조 제1항 제4호') > 0:
        labeling_etc.loc[i, '공공질서(기타)'] = 1
    if content.find('제34조 제1항 제20호') > 0:
        labeling_etc.loc[i, '공공질서(기타)'] = 1
    if content.find('제7조 제1항 제18호') > 0:
        labeling_etc.loc[i, '공공질서(기타)'] = 1

    # 거절사유 "상품류(기타)":
    if content.find('상품류구분과 일치하지') > 0:
        labeling_etc.loc[i, '상품류(기타)'] = 1
    if content.find('상품류구분을 분할출원하면') > 0:
        labeling_etc.loc[i, '상품류(기타)'] = 1
    if content.find('불명확') > 0:
        labeling_etc.loc[i, '상품류(기타)'] = 1
    if content.find('류지정상품') > 0:
        labeling_etc.loc[i, '상품류(기타)'] = 1
    if content.find('류 지정상품') > 0:
        labeling_etc.loc[i, '상품류(기타)'] = 1
    if content.find('상품분류구분에 일치하지') > 0:
        labeling_etc.loc[i, '상품류(기타)'] = 1
    if content.find('포괄명칭') > 0:
        labeling_etc.loc[i, '상품류(기타)'] = 1
    if content.find('서비스류구분과 일치하지') > 0:
        labeling_etc.loc[i, '상품류(기타)'] = 1
    if content.find('타류') > 0:
        labeling_etc.loc[i, '상품류(기타)'] = 1

    # 거절사유 "개단(기타)":
    if content.find('개인') > content.find('단체') > 0:
        labeling_etc.loc[i, '개단(기타)'] = 1
    if content.find('개인') > content.find('법인') > 0:
        labeling_etc.loc[i, '개단(기타)'] = 1
    if content.find('개인') > content.find('법인명의') > 0:
        labeling_etc.loc[i, '개단(기타)'] = 1
    if content.find('개인이 법인명의의') > 0:
        labeling_etc.loc[i, '개단(기타)'] = 1
    if content.find('자연인') > content.find('법인') > 0:
        labeling_etc.loc[i, '개단(기타)'] = 1
    if content.find('법인') > 0:
        labeling_etc.loc[i, '개단(기타)'] = 1

    # 거절사유 "주소상이(기타)":
    if content.find('본원 갱신등록출원인이 등록권자와') > 0:
        labeling_etc.loc[i, '주소상이(기타)'] = 1
    if content.find('등록권리자와 일치하지 아니') > 0:
        labeling_etc.loc[i, '주소상이(기타)'] = 1
    if content.find('갱신등록출원인이 등록권자와 일치하지') > 0:
        labeling_etc.loc[i, '주소상이(기타)'] = 1
    if content.find('주소가 일치하지') > 0:
        labeling_etc.loc[i, '주소상이(기타)'] = 1

    # 거절사유 "사용의사(기타)":
    if content.find('상표법 제3조') > 0:
        labeling_etc.loc[i, '사용의사(기타)'] = 1
    if content.find('제3조') > 0:
        labeling_etc.loc[i, '사용의사(기타)'] = 1

    # 거절사유 "단체표장(기타)":
    if content.find('제54조') > content.find('단체') > 0:
        labeling_etc.loc[i, '단체표장(기타)'] = 1
    if content.find('제23조 제1항 제6호') > 0:
        labeling_etc.loc[i, '단체표장(기타)'] = 1

    # 거절사유 "업무표장(기타)":
    if content.find('2조') > content.find('업무') > 0:
        labeling_etc.loc[i, '업무표장(기타)'] = 1

    # 거절사유 "정의위배(기타)":
    if content.find('2조') > content.find('상표') > 0:
        labeling_etc.loc[i, '정의위배(기타)'] = 1
    if content.find('제23조 제1항 제4호') > 0:
        labeling_etc.loc[i, '정의위배(기타)'] = 1

    # 거절사유 "저명한성명포함(기타)":
    if content.find('제34조 제1항 제6호') > 0:
        labeling_etc.loc[i, '저명한성명포함(기타)'] = 1
    if content.find('제7조 제1항 제6호') > 0:
        labeling_etc.loc[i, '저명한성명포함(기타)'] = 1

    # 거절사유 "수요자기만(기타)":
    if content.find('제34조 제1항 제12호') > 0:
        labeling_etc.loc[i, '수요자기만(기타)'] = 1
    if content.find('제7조 제1항 제11호') > 0:
        labeling_etc.loc[i, '수요자기만(기타)'] = 1

    # 거절사유 "부정한목적(기타)":
    if content.find('제34조 제1항 제13호') > 0:
        labeling_etc.loc[i, '부정한목적(기타)'] = 1
    if content.find('제7조 제1항 제12호') > 0:
        labeling_etc.loc[i, '부정한목적(기타)'] = 1

    # 거절사유 "저명한국제기관(기타)":
    if content.find('제34조 제1항 제1호') > 0:
        labeling_etc.loc[i, '저명한국제기관(기타)'] = 1
    if content.find('제7조 제1항 제1호') > 0:
        labeling_etc.loc[i, '저명한국제기관(기타)'] = 1
    if content.find('제7조 제1항 제1의 5호') > 0:
        labeling_etc.loc[i, '저명한국제기관(기타)'] = 1
    if content.find('IOC') > 0:
        labeling_etc.loc[i, '저명한국제기관(기타)'] = 1

    # 거절사유 "입체적형상(기타)":
    if content.find('제7조 제1항 제13호') > 0:
        labeling_etc.loc[i, '입체적형상(기타)'] = 1

    # 거절사유 "포도주산지(기타)":
    if content.find('제7조 제1항 제14호') > 0:
        labeling_etc.loc[i, '포도주산지(기타)'] = 1

    # 거절사유 "저명한업무(기타)":
    if content.find('제7조 제1항 제3호') > 0:
        labeling_etc.loc[i, '저명한업무(기타)'] = 1

    # 거절사유 "소멸후3년내출원(기타)":
    if content.find('제7조 제5항') > 0:
        labeling_etc.loc[i, '소멸후3년내출원(기타)'] = 1

    # 거절사유 "취소심판청구인외의출원(기타)":
    if content.find('제8조 제5항') > 0:
        labeling_etc.loc[i, '취소심판청구인외의출원(기타)'] = 1

    # 거절사유 "상표유형오기재(기타)":
    if content.find('제9조 제1항 제4호2') > 0:
        labeling_etc.loc[i, '상표유형오기재(기타)'] = 1
    if content.find('권리구분') > 0:
        labeling_etc.loc[i, '상표유형오기재(기타)'] = 1


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
    elif labeling_etc.loc[i, '상품류(기타)'] == 1:
        labeling_etc.loc[i, 'label'] = 1
    elif labeling_etc.loc[i, '개단(기타)'] == 1:
        labeling_etc.loc[i, 'label'] = 1
    elif labeling_etc.loc[i, '주소상이(기타)'] == 1:
        labeling_etc.loc[i, 'label'] = 1
    elif labeling_etc.loc[i, '사용의사(기타)'] == 1:
        labeling_etc.loc[i, 'label'] = 1
    elif labeling_etc.loc[i, '단체표장(기타)'] == 1:
        labeling_etc.loc[i, 'label'] = 1
    elif labeling_etc.loc[i, '업무표장(기타)'] == 1:
        labeling_etc.loc[i, 'label'] = 1
    elif labeling_etc.loc[i, '정의위배(기타)'] == 1:
        labeling_etc.loc[i, 'label'] = 1
    elif labeling_etc.loc[i, '저명한성명포함(기타)'] == 1:
        labeling_etc.loc[i, 'label'] = 1
    elif labeling_etc.loc[i, '수요자기만(기타)'] == 1:
        labeling_etc.loc[i, 'label'] = 1
    elif labeling_etc.loc[i, '부정한목적(기타)'] == 1:
        labeling_etc.loc[i, 'label'] = 1
    elif labeling_etc.loc[i, '저명한국제기관(기타)'] == 1:
        labeling_etc.loc[i, 'label'] = 1
    elif labeling_etc.loc[i, '입체적형상(기타)'] == 1:
        labeling_etc.loc[i, 'label'] = 1
    elif labeling_etc.loc[i, '포도주산지(기타)'] == 1:
        labeling_etc.loc[i, 'label'] = 1
    elif labeling_etc.loc[i, '저명한업무(기타)'] == 1:
        labeling_etc.loc[i, 'label'] = 1
    elif labeling_etc.loc[i, '소멸후3년내출원(기타)'] == 1:
        labeling_etc.loc[i, 'label'] = 1
    elif labeling_etc.loc[i, '취소심판청구인외의출원(기타)'] == 1:
        labeling_etc.loc[i, 'label'] = 1
    elif labeling_etc.loc[i, '상표유형오기재(기타)'] == 1:
        labeling_etc.loc[i, 'label'] = 1

    else:
        labeling_etc.loc[i, 'label'] = 2

Mo_labeling_1 = labeling_etc[labeling_etc['label'].isin([1])]
new_labeling_etc = labeling_etc[labeling_etc['label'].isin([2])]

labeling_1 = pd.concat([labeling_1, Mo_labeling_1])
print('{}건 수정됨'.format(len(Mo_labeling_1)))

new_labeling_etc = new_labeling_etc[['patent_id', 'korean', 'english', 'rejectionContentDetail', '발음유사', '관념유사', '외관유사', '식별력',
                                     '상품 불명확', '품질오인(기타)', '공공질서(기타)', '상품류(기타)', '개단(기타)', '주소상이(기타)',
                                     '사용의사(기타)', '단체표장(기타)', '업무표장(기타)', '정의위배(기타)', '저명한성명포함(기타)',
                                     '수요자기만(기타)', '부정한목적(기타)', '저명한국제기관(기타)', '입체적형상(기타)',
                                     '포도주산지(기타)', '저명한업무(기타)', '소멸후3년내출원(기타)', '취소심판청구인외의출원(기타)', '상표유형오기재(기타)', 'label']]
labeling_1 = labeling_1[['patent_id', 'korean', 'english', 'rejectionContentDetail', '발음유사', '관념유사', '외관유사', '식별력',
                         '상품 불명확', '품질오인(기타)', '공공질서(기타)', '상품류(기타)', '개단(기타)', '주소상이(기타)',
                         '사용의사(기타)', '단체표장(기타)', '업무표장(기타)', '정의위배(기타)', '저명한성명포함(기타)',
                         '수요자기만(기타)', '부정한목적(기타)', '저명한국제기관(기타)', '입체적형상(기타)',
                        '포도주산지(기타)', '저명한업무(기타)', '소멸후3년내출원(기타)', '취소심판청구인외의출원(기타)', '상표유형오기재(기타)', 'label']]

# 수정결과 저장
labeling_1.to_csv(PATH + 'labeling_1_1.csv', index=False, encoding='cp949')
new_labeling_etc.to_csv(PATH + 'labeling_etc.csv', index=False, encoding='cp949')