# csv를 pandas dataframe으로 불러와서 series를 txt로 변환하여 저장하는 코드

import pandas as pd
from tqdm import tqdm


df = pd.read_csv('D:/project/OA_paper/DATA/Part/Train.csv', encoding='cp949')
df_contents = df['rejectionContentDetail']

print(df)

mystr = ''

for i in tqdm(range(len(df_contents))):
    mystr += str(df_contents[i]) + '\n'

with open('OA_corpus.txt', 'w', encoding='utf-8') as f:
    f.write(mystr)

r = open('OA_corpus.txt', mode='r', encoding='utf-8') # 로드한 txt 를 읽기

r.readlines()
