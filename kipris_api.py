import urllib.request
import requests
import urllib3
import pandas as pd
from time import sleep
import urllib.request
import xmltodict, json
from tqdm import tqdm
import http
import os, sys
import xml
from pathlib import Path


PATH = './'

api_key = '8cUXmiGi5gv7WVjqg2jiF0nzsN=6MYk/kCzJRscV2F0='

labeling_etc = pd.read_csv('D:/project/OA_paper/DATA/labeling_etc.csv', encoding='cp949')

applicationNumbers = labeling_etc['patent_id']


df = []
df = pd.DataFrame(columns=['patent_id', 'additionRejectInfo'])

# 진행상황 저장
log_filename = 'Log_OA_patent_additionRejectInfo.csv'
my_file_log_df = Path(PATH + log_filename)

if my_file_log_df.is_file():
    log_df = pd.read_csv(PATH + log_filename, encoding='euc-kr')
    num = int(log_df.iloc[0, 0])
    print(num)
    print('수집한 것 이후로 다시 수집 시작')
else:
    num = 0
    print('새로운 api 수집 시작, OA 파일 없었음')

idx = labeling_etc[labeling_etc['patent_id'] == num].index
idx = idx.tolist()[0]
applicationNumbers = tqdm(applicationNumbers[idx+1:])


for applicationNumber in applicationNumbers:

    # 로그파일 - 전체
    sys.stdout = open('api_log.txt', 'a')

    # 진행상황 로그
    log = 'last number'
    log_df = []
    log_df = pd.DataFrame(columns=['num', 'log'])
    log_df = log_df.append(pd.DataFrame([[applicationNumber, log]],
                                        columns=['num', 'log']), ignore_index=True)
    log_df.to_csv(PATH + log_filename, header=True, index=False, encoding='euc-kr')


    applicationNumber = str(applicationNumber)[:-2]

    url = 'http://plus.kipris.or.kr/openapi/rest/IntermediateDocumentREService/additionRejectInfo?applicationNumber={}&accessKey={}'.format(applicationNumber, api_key)

    sleep(0.7)

    try:
        response = urllib.request.urlopen(url)
    except urllib.error.URLError as e:
        print(e)
        print('urllib.error.URLError. Plz wait for 5 minutes')
        sleep(300)
        try:
            response = urllib.request.urlopen(url)
        except TimeoutError as e:
            print(e)
            print('TimeoutError agiain. Plz wait for 10 minute')
            sleep(600)
            response = urllib.request.urlopen(url)
            pass
        pass
    except urllib.error.HTTPError as e:
        print(e)
        print('urllib.error.HTTPError. Plz wait for 5 minutes')
        sleep(300)
        response = urllib.request.urlopen(url)
        pass
    except http.client.HTTPException as e:
        print(e)
        print('http.client.HTTPException. Plz wait for 5 minutes')
        sleep(300)
        response = urllib.request.urlopen(url)
        pass
    except requests.exceptions.ConnectionError as e:
        print(e)
        print('requests.exceptions.ConnectionError. Plz wait for 5 minutes')
        sleep(300)
        response = urllib.request.urlopen(url)
        pass
    except TimeoutError as e:
        print(e)
        print('TimeoutError. Plz wait for 5 minutes')
        sleep(300)
        response = urllib.request.urlopen(url)
        pass
    except urllib3.exceptions.NewConnectionError as e:
        print(e)
        print('urllib3.exceptions.NewConnectionError. Plz wait for 5 minutes')
        sleep(300)
        response = urllib.request.urlopen(url)
        pass
    except urllib3.exceptions.MaxRetryError as e:
        print(e)
        print('urllib3.exceptions.MaxRetryError. Plz wait for 5 minutes')
        sleep(300)
        response = urllib.request.urlopen(url)
        pass
    except http.client.RemoteDisconnected as e:
        print(e)
        print('http.client.RemoteDisconnected. Plz wait for 5 minutes')
        sleep(300)
        response = urllib.request.urlopen(url)
        pass
    except ConnectionResetError as e:
        print(e)
        print('ConnectionResetError: [WinError 10054] 현재 연결은 원격 호스트에 의해 강제로 끊겼습니다')
        sleep(300)
        response = urllib.request.urlopen(url)
        pass
    except xml.parsers.expat.ExpatError as e:
        print(e)
        print('xml.parsers.expat.ExpatError')
        sleep(300)
        response = urllib.request.urlopen(url)
        pass
    except  ConnectionError as e:
        print(e)
        sleep(300)
        response = urllib.request.urlopen(url)


    responseData = response.read()
    responseData = xmltodict.parse(responseData)
    responseData = json.dumps(responseData)
    responseData = json.loads(responseData)

    print(applicationNumber, ',', responseData)

    if not responseData['response']['body']['items'] == None:
        df = pd.read_csv(PATH + 'additionRejectInfo.csv', encoding='cp949')
        additionRejectInfo = responseData['response']['body']['items']['additionRejectInfo']
        df = df.append(pd.DataFrame([[applicationNumber, additionRejectInfo]],
                                    columns=['patent_id', 'additionRejectInfo']), ignore_index=True)
        df.to_csv(PATH + 'additionRejectInfo.csv', header=True, index=False, encoding='euc-kr')
