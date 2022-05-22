import os
import pandas as pd
from tqdm import tqdm


inputpath = 'D:/project/OA_paper/DATA/Labeling/'
savepath = 'D:/project/OA_paper/DATA/'

files = os.listdir(inputpath)
files = tqdm(files)

labeling_all = pd.DataFrame()
for file in files:
    raw = pd.read_csv(inputpath + file, encoding='cp949')
    output = pd.concat([output, raw])


labeling_1 = labeling_all[labeling_all['label'].isin([1])]
labeling_2 = labeling_all[labeling_all['label'].isin([2])]

labeling_all.to_csv(savepath + "labeling_all.csv", index=False, encoding='cp949')
labeling_1.to_csv(savepath + "labeling_1.csv", index=False, encoding='cp949')
labeling_2.to_csv(savepath + "labeling_etc.csv", index=False, encoding='cp949')
