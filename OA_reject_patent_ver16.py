import urllib.request
import pandas as pd
from time import sleep
import urllib.request
import xmltodict, json
from pathlib import Path
import http
import urllib3
import requests
import urllib.error
from xml.parsers.expat import ExpatError
import xml
import re


def firstNon0(_list, startindex):
    for index in range(startindex, len(_list)):
        if _list[index] != 0: return index
    return None


#### 특허, 실용신안 OA 수집
#### 아래의 4종류 파라미터 설정 후 코드 진행
#### 설정해야하는 파라미터 4종류
#### error로 코드 중지시 그대로 재시작(이어서 시작됨)

#1. api_key
api_key = 'JNDlhXxeBGqzOEcqzllBnTAFlQss1EdWy=jWFjqRI=4='

#2. 저장 위치 설정
save_path = 'D:/01_coding_result/OA/'
error_save_path = 'D:/01_coding_result/OA/error/'

#3. 특허, 실용신안 선택(특허 : 10, 실용신안 : 20)
f_number = 10

#4. 수집기간
start_year = 2013
end_year = 2020




for year in range(start_year, end_year+1):
    df = []
    df = pd.DataFrame(columns=['patent_id', 'korean', 'english', 'rejectionContentDetail'])
    error_df = []
    error_df = pd.DataFrame(columns=['patent_id', 'error'])

    filename = f'OA_patent_{f_number}_{year}.csv'
    error_filename = f'error_OA_patent_{f_number}_{year}.csv'
    log_filename = f'Log_OA_patent_{f_number}_{year}.csv'

    my_file_df = Path(save_path + filename)
    my_file_log_df = Path(save_path + log_filename)
    my_file_error_df = Path(error_save_path + error_filename)
    # log_df = pd.read_csv(save_path + log_filename, encoding='euc-kr')

    if my_file_log_df.is_file():
    # if my_file_df.is_file():

        # df = pd.read_csv(save_path + filename,encoding='euc-kr')
        #
        # num = str(df['patent_id'][len(df)-1])
        #
        # abc = num[6:]
        # num = list(str(num)[6:])
        # num = list(map(int, num))
        # num = abc[firstNon0(num,0):]
        # num = int(num)
        log_df = pd.read_csv(save_path + log_filename, encoding='euc-kr')
        num = int(log_df.iloc[0,0])

        print(num)
        print('수집한 것 이후로 다시 수집 시작')
    else :
        num = 0
        print('새로운 api 수집 시작, OA 파일 없었음')

    if my_file_df.is_file():

        df = pd.read_csv(save_path + filename,encoding='euc-kr')
    else:
        print('기존 OA 파일 없었음')








    if my_file_error_df.is_file():
        error_df = pd.read_csv(error_save_path + error_filename, encoding='euc-kr')
    else : print('새로운 api 수집 시작, error 파일 없었음')

    for j in range(num+1,200001):
        log = 'last number'
        log_df = []
        log_df = pd.DataFrame(columns=['num', 'log'])
        log_df = log_df.append(pd.DataFrame([[j, log]],
                                    columns=['num', 'log']), ignore_index=True)
        log_df.to_csv(save_path + log_filename, header=True, index=False, encoding='euc-kr')



        applicationNumber=str(f_number) + str(year) + str(j).zfill(7)


        # url = "http://plus.kipris.or.kr/openapi/rest/IntermediateDocumentOPService/rejectDecisionInfo?applicationNumber=" + str(
        #     applicationNumber) + "&accessKey=" + str(api_key)

        url = f'http://plus.kipris.or.kr/openapi/rest/IntermediateDocumentOPService/additionRejectInfo?applicationNumber={applicationNumber}&accessKey={api_key}'

        try:
            response_contents = urllib.request.urlopen(url)
        except urllib.error.URLError as e:
            print(e)
            print('urllib.error.URLError. Plz wait for 5 minutes')
            sleep(300)
            try:
                response_contents = urllib.request.urlopen(url)
            except TimeoutError as e:
                print(e)
                print('TimeoutError agiain. Plz wait for 10 minute')
                sleep(600)
                response_contents = urllib.request.urlopen(url)
                pass
            pass
        except urllib.error.HTTPError as e:
            print(e)
            print('urllib.error.HTTPError. Plz wait for 5 minutes')
            sleep(300)
            response_contents = urllib.request.urlopen(url)
            pass
        except http.client.HTTPException as e:
            print(e)
            print('http.client.HTTPException. Plz wait for 5 minutes')
            sleep(300)
            response_contents = urllib.request.urlopen(url)
            pass
        except requests.exceptions.ConnectionError as e:
            print(e)
            print('requests.exceptions.ConnectionError. Plz wait for 5 minutes')
            sleep(300)
            response_contents = urllib.request.urlopen(url)
            pass
        except TimeoutError as e:
            print(e)
            print('TimeoutError. Plz wait for 5 minutes')
            sleep(300)
            response_contents = urllib.request.urlopen(url)
            pass
        except urllib3.exceptions.NewConnectionError as e:
            print(e)
            print('urllib3.exceptions.NewConnectionError. Plz wait for 5 minutes')
            sleep(300)
            response_contents = urllib.request.urlopen(url)
            pass
        except urllib3.exceptions.MaxRetryError as e:
            print(e)
            print('urllib3.exceptions.MaxRetryError. Plz wait for 5 minutes')
            sleep(300)
            response_contents = urllib.request.urlopen(url)
            pass
        except http.client.RemoteDisconnected as e:
            print(e)
            print('http.client.RemoteDisconnected. Plz wait for 5 minutes')
            sleep(300)
            response_contents = urllib.request.urlopen(url)
            pass
        except ConnectionResetError as e:
            print(e)
            print('ConnectionResetError: [WinError 10054] 현재 연결은 원격 호스트에 의해 강제로 끊겼습니다')
            sleep(300)
            response_contents = urllib.request.urlopen(url)
            pass
        except xml.parsers.expat.ExpatError as e:
            print(e)
            print('xml.parsers.expat.ExpatError')
            sleep(300)
            response_contents = urllib.request.urlopen(url)
            pass
        except  ConnectionError as e:
            print(e)
            sleep(300)
            response_contents = urllib.request.urlopen(url)



        responseData_contents = response_contents.read()
        responseData_contents = xmltodict.parse(responseData_contents)
        responseData_contents = json.dumps(responseData_contents)
        responseData_contents = json.loads(responseData_contents)



        if not responseData_contents['response']['body']['items'] == None:

            if type(responseData_contents['response']['body']['items']['additionRejectInfo']) == dict:
                print(applicationNumber, 'additionRejectInfo 있음', 'dict')

                rejectionContentDetail = responseData_contents['response']['body']['items']['additionRejectInfo']['additionRejectionContent']
                try:

                    rejectionContentDetail = re.sub(r'[^ ㄱ-ㅣ가-힣A-Za-z]', '', rejectionContentDetail)
                except Exception as ex:
                    print('특수기호 제거시 에러가 발생했습니다.',ex)

                url = f'http://plus.kipris.or.kr/kipo-api/kipi/designInfoSearchService/getBibliographyDetailInfoSearch?applicationNumber={applicationNumber}&ServiceKey={api_key}'
                try:
                    response_title = urllib.request.urlopen(url)
                except urllib.error.URLError as e:
                    print(e)
                    print('urllib.error.URLError. Plz wait for 5 minutes')
                    sleep(300)
                    try:
                        response_title = urllib.request.urlopen(url)
                    except TimeoutError as e:
                        print(e)
                        print('TimeoutError agiain. Plz wait for 10 minute')
                        sleep(600)
                        response_title = urllib.request.urlopen(url)
                        pass
                    pass
                except urllib.error.HTTPError as e:
                    print(e)
                    print('urllib.error.HTTPError. Plz wait for 5 minutes')
                    sleep(300)
                    response_title = urllib.request.urlopen(url)
                    pass
                except http.client.HTTPException as e:
                    print(e)
                    print('http.client.HTTPException. Plz wait for 5 minutes')
                    sleep(300)
                    response_title = urllib.request.urlopen(url)
                    pass
                except requests.exceptions.ConnectionError as e:
                    print(e)
                    print('requests.exceptions.ConnectionError. Plz wait for 5 minutes')
                    sleep(300)
                    response_title = urllib.request.urlopen(url)
                    pass
                except TimeoutError as e:
                    print(e)
                    print('TimeoutError. Plz wait for 5 minutes')
                    sleep(300)
                    response_title = urllib.request.urlopen(url)
                    pass
                except urllib3.exceptions.NewConnectionError as e:
                    print(e)
                    print('urllib3.exceptions.NewConnectionError. Plz wait for 5 minutes')
                    sleep(300)
                    response_title = urllib.request.urlopen(url)
                    pass
                except urllib3.exceptions.MaxRetryError as e:
                    print(e)
                    print('urllib3.exceptions.MaxRetryError. Plz wait for 5 minutes')
                    sleep(300)
                    response_title = urllib.request.urlopen(url)
                    pass
                except http.client.RemoteDisconnected as e:
                    print(e)
                    print('http.client.RemoteDisconnected. Plz wait for 5 minutes')
                    sleep(300)
                    response_title = urllib.request.urlopen(url)
                    pass
                except ConnectionResetError as e:
                    print(e)
                    print('ConnectionResetError: [WinError 10054] 현재 연결은 원격 호스트에 의해 강제로 끊겼습니다')
                    sleep(300)
                    response_title = urllib.request.urlopen(url)
                    pass
                except  ConnectionError as e:
                    print(e)
                    sleep(300)
                    response_title = urllib.request.urlopen(url)


                responseData_title = response_title.read()
                responseData_title = xmltodict.parse(responseData_title)
                responseData_title = json.dumps(responseData_title)
                responseData_title = json.loads(responseData_title)
                try:

                    korean=responseData_title['response']['body']['item']['biblioSummaryInfoArray']['biblioSummaryInfo'][
                              'inventionTitle']
                    english=responseData_title['response']['body']['item']['biblioSummaryInfoArray']['biblioSummaryInfo'][
                              'inventionTitleEng']

                    # df = df.append(pd.DataFrame([[applicationNumber,korean,english, rejectionContentDetail]],
                    #                             columns=['patent_id','korean','english','rejectionContentDetail']), ignore_index=True)


                    # df.to_csv(save_path+filename, header=True, index=False, encoding='euc-kr')
                except Exception as ex:
                    error_df = error_df.append(
                        pd.DataFrame([[applicationNumber, ex]], columns=['patent_id', 'error']),
                        ignore_index=True)
                    error_df.to_csv(error_save_path + error_filename, header=True, index=False, encoding='euc-kr')
                try:
                    basket = {'patent_id': [applicationNumber], 'korean': [korean], 'english': [english],
                              'rejectionContentDetail': [rejectionContentDetail]}
                    basket = pd.DataFrame(basket)
                    print('basket:',basket)
                    basket.to_csv(save_path + filename, mode='a', index=False, header=False, encoding='euc-kr')
                    # df.to_csv(save_path + filename, header=True, index=False, encoding='euc-kr')
                except Exception as ex:
                    error_df = error_df.append(
                        pd.DataFrame([[applicationNumber, ex]], columns=['patent_id', 'error']),
                        ignore_index=True)
                    error_df.to_csv(error_save_path + error_filename, header=True, index=False, encoding='euc-kr')


            elif type(responseData_contents['response']['body']['items']['additionRejectInfo']) == list:
                print(applicationNumber, 'additionRejectInfo 있음', 'list')

                for k in range(len(responseData_contents['response']['body']['items']['additionRejectInfo'])):
                    rejectionContentDetail = responseData_contents['response']['body']['items']['additionRejectInfo'][k]['additionRejectionContent']
                    try:

                        rejectionContentDetail = re.sub(r'[^ ㄱ-ㅣ가-힣A-Za-z]', '', rejectionContentDetail)
                    except Exception as ex:
                        print('특수기호 제거시 에러가 발생했습니다.',ex)


                    url = f'http://plus.kipris.or.kr/kipo-api/kipi/patUtiModInfoSearchSevice/getBibliographyDetailInfoSearch?applicationNumber={applicationNumber}&ServiceKey={api_key}'
                    try:
                        response_title = urllib.request.urlopen(url)
                    except urllib.error.URLError as e:
                        print(e)
                        print('urllib.error.URLError. Plz wait for 5 minutes')
                        sleep(300)
                        try:
                            response_title = urllib.request.urlopen(url)
                        except TimeoutError as e:
                            print(e)
                            print('TimeoutError agiain. Plz wait for 10 minute')
                            sleep(600)
                            response_title = urllib.request.urlopen(url)
                            pass
                        pass
                    except urllib.error.HTTPError as e:
                        print(e)
                        print('urllib.error.HTTPError. Plz wait for 5 minutes')
                        sleep(300)
                        response_title = urllib.request.urlopen(url)
                        pass
                    except http.client.HTTPException as e:
                        print(e)
                        print('http.client.HTTPException. Plz wait for 5 minutes')
                        sleep(300)
                        response_title = urllib.request.urlopen(url)
                        pass
                    except requests.exceptions.ConnectionError as e:
                        print(e)
                        print('requests.exceptions.ConnectionError. Plz wait for 5 minutes')
                        sleep(300)
                        response_title = urllib.request.urlopen(url)
                        pass
                    except TimeoutError as e:
                        print(e)
                        print('TimeoutError. Plz wait for 5 minutes')
                        sleep(300)
                        response_title = urllib.request.urlopen(url)
                        pass
                    except urllib3.exceptions.NewConnectionError as e:
                        print(e)
                        print('urllib3.exceptions.NewConnectionError. Plz wait for 5 minutes')
                        sleep(300)
                        response_title = urllib.request.urlopen(url)
                        pass
                    except urllib3.exceptions.MaxRetryError as e:
                        print(e)
                        print('urllib3.exceptions.MaxRetryError. Plz wait for 5 minutes')
                        sleep(300)
                        response_title = urllib.request.urlopen(url)
                        pass
                    except http.client.RemoteDisconnected as e:
                        print(e)
                        print('http.client.RemoteDisconnected. Plz wait for 5 minutes')
                        sleep(300)
                        response_title = urllib.request.urlopen(url)
                        pass
                    except ConnectionResetError as e:
                        print(e)
                        print('ConnectionResetError: [WinError 10054] 현재 연결은 원격 호스트에 의해 강제로 끊겼습니다')
                        sleep(300)
                        response_title = urllib.request.urlopen(url)
                        pass
                    except  ConnectionError as e:
                        print(e)
                        sleep(300)
                        response_title = urllib.request.urlopen(url)

                    responseData_title = response_title.read()
                    responseData_title = xmltodict.parse(responseData_title)
                    responseData_title = json.dumps(responseData_title)
                    responseData_title = json.loads(responseData_title)

                    try:
                        korean = responseData_title['response']['body']['item']['biblioSummaryInfoArray'][
                            'biblioSummaryInfo']['inventionTitle']


                        english = responseData_title['response']['body']['item']['biblioSummaryInfoArray'][
                            'biblioSummaryInfo']['inventionTitleEng']

                        # df = df.append(pd.DataFrame([[applicationNumber, korean, english, rejectionContentDetail]],
                        #                             columns=['patent_id', 'korean', 'english',
                        #                                      'rejectionContentDetail']), ignore_index=True)



                        # df.to_csv(save_path + filename, header=True, index=False, encoding='euc-kr')

                    except Exception as ex:
                        error_df = error_df.append(
                            pd.DataFrame([[applicationNumber, ex]], columns=['patent_id', 'error']),
                            ignore_index=True)
                        error_df.to_csv(error_save_path + error_filename, header=True, index=False, encoding='euc-kr')


                    try:
                        basket = {'patent_id': [applicationNumber], 'korean': [korean], 'english': [english],
                                  'rejectionContentDetail': [rejectionContentDetail]}
                        basket = pd.DataFrame(basket)
                        # print('basket:', basket)
                        basket.to_csv(save_path + filename, mode='a', index=False, header=False, encoding='euc-kr')

                        # df.to_csv(save_path + filename, header=True, index=False, encoding='euc-kr')
                    except Exception as ex:
                        error_df = error_df.append(
                            pd.DataFrame([[applicationNumber, ex]], columns=['patent_id', 'error']),
                            ignore_index=True)
                        error_df.to_csv(error_save_path + error_filename, header=True, index=False, encoding='euc-kr')



        else :

            print(year, applicationNumber, ': contents 없음')

            url = f'http://plus.kipris.or.kr/kipo-api/kipi/designInfoSearchService/getBibliographyDetailInfoSearch?applicationNumber={applicationNumber}&ServiceKey={api_key}'
            try:
                response_title = urllib.request.urlopen(url)
            except urllib.error.URLError as e:
                print(e)
                print('urllib.error.URLError. Plz wait for 5 minutes')
                sleep(300)
                try:
                    response_title = urllib.request.urlopen(url)
                except TimeoutError as e:
                    print(e)
                    print('TimeoutError agiain. Plz wait for 10 minute')
                    sleep(600)
                    response_title = urllib.request.urlopen(url)
                    pass
                pass
            except urllib.error.HTTPError as e:
                print(e)
                print('urllib.error.HTTPError. Plz wait for 5 minutes')
                sleep(300)
                response_title = urllib.request.urlopen(url)
                pass
            except http.client.HTTPException as e:
                print(e)
                print('http.client.HTTPException. Plz wait for 5 minutes')
                sleep(300)
                response_title = urllib.request.urlopen(url)
                pass
            except requests.exceptions.ConnectionError as e:
                print(e)
                print('requests.exceptions.ConnectionError. Plz wait for 5 minutes')
                sleep(300)
                response_title = urllib.request.urlopen(url)
                pass
            except TimeoutError as e:
                print(e)
                print('TimeoutError. Plz wait for 5 minutes')
                sleep(300)
                response_title = urllib.request.urlopen(url)
                pass
            except urllib3.exceptions.NewConnectionError as e:
                print(e)
                print('urllib3.exceptions.NewConnectionError. Plz wait for 5 minutes')
                sleep(300)
                response_title = urllib.request.urlopen(url)
                pass
            except urllib3.exceptions.MaxRetryError as e:
                print(e)
                print('urllib3.exceptions.MaxRetryError. Plz wait for 5 minutes')
                sleep(300)
                response_title = urllib.request.urlopen(url)
                pass
            except http.client.RemoteDisconnected as e:
                print(e)
                print('http.client.RemoteDisconnected. Plz wait for 5 minutes')
                sleep(300)
                response_title = urllib.request.urlopen(url)
                pass
            except ConnectionResetError as e:
                print(e)
                print('ConnectionResetError: [WinError 10054] 현재 연결은 원격 호스트에 의해 강제로 끊겼습니다')
                sleep(300)
                response_title = urllib.request.urlopen(url)
                pass
            except  ConnectionError as e:
                print(e)
                sleep(300)
                response_title = urllib.request.urlopen(url)

            responseData_title = response_title.read()
            responseData_title = xmltodict.parse(responseData_title)
            responseData_title = json.dumps(responseData_title)
            responseData_title = json.loads(responseData_title)
            try:
                korean = responseData_title['response']['body']['item']['biblioSummaryInfoArray']['biblioSummaryInfo'][
                    'inventionTitle']
                english = responseData_title['response']['body']['item']['biblioSummaryInfoArray']['biblioSummaryInfo'][
                    'inventionTitleEng']

            except Exception as ex:
                print(ex, '해당번호 특허OA 없음')
                korean = 'None'
                english = 'None'

            rejectionContentDetail = 'None'
            # df = df.append(pd.DataFrame([[applicationNumber, korean, english, rejectionContentDetail]],
            #                             columns=['patent_id', 'korean', 'english',
            #                                      'rejectionContentDetail']), ignore_index=True)
            # df.to_csv(save_path + filename, header=True, index=False, encoding='euc-kr')
            basket = {'patent_id': [str(applicationNumber)], 'korean': [korean], 'english': [english],
                      'rejectionContentDetail': [rejectionContentDetail]}
            # print('basket:', basket)
            basket = pd.DataFrame(basket)
            # print('basket:', basket)
            basket.to_csv(save_path + filename, mode='a', index=False, header=False, encoding='euc-kr')











