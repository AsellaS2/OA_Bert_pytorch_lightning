import pandas as pd


df = pd.read_csv('D:/project/OA_paper/DATA/Etc/test/labeling_etc.csv', encoding='cp949')

def slice_id(patent_id):
    patent_id = str(patent_id)
    return patent_id[:1]

df['filter'] = list(map(slice_id, df['patent_id']))
df = df[df['filter'].isin(['4'])]
df.drop(['filter'], axis=1)

df.to_csv('test.csv', index=False, encoding='cp949')
