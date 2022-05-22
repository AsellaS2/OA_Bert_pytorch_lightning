import json

vocab_path = "./oa-6000-wpm-32000-vocab.txt"

vocab_file = './oa-6000-wpm-32000-vocab_pre.txt'
f = open(vocab_file,'w',encoding='utf-8')
with open(vocab_path, 'r', encoding='utf-8') as json_file:
    json_data = json.load(json_file)
    for item in json_data["model"]["vocab"].keys():
        f.write(item+'\n')
    f.close()

# import chardet
#
# with open("./oa-6000-wpm-32000-vocab.txt", "r", encoding="utf-8") as f:
#     file_data = f.readline()
# print(file_data)
# print(chardet.detect(file_data.encode()))

# # 입력 스트림과 출력 스트림을 연다
# input = open("D:/project/OA_paper/DATA/Part/OA_corpus_test.txt", "rt", encoding="utf-8")
# output = open("D:/project/OA_paper/DATA/Part/OA_corpus_test_output.txt", "wt", encoding="utf-8")
#
# # 유니코드 데이터 조각들을 스트리밍한다
# with input, output:
#     while True:
#         # 데이터 조각을 읽고
#         chunk = input.read(4096)
#         if not chunk:
#             break
#         # 수직 탭을 삭제한다
#         chunk = chunk.replace("\u000B", "")
#         # 데이터 조각을 쓴다
#         output.write(chunk)