import numpy as np
from tqdm import tqdm

from torch.utils.data import TensorDataset

from sklearn import metrics
from sklearn.preprocessing import MultiLabelBinarizer

import matplotlib.pyplot as plt
import seaborn as sns

from bert_op import *


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

inputpath = 'D:/project/OA_paper/DATA/Part/k_fold/5/'
savepath = 'D:/project/OA_paper/output/5/'

df = pd.read_csv(inputpath + 'Test5.csv', encoding='cp949')

df['label'] = [i.replace('발음유사', '칭호유사') for i in df['label']]

x = list(df['rejectionContentDetail'])
y = list(map(lambda x: x.split(', '), df['label']))

mlb = MultiLabelBinarizer()
yt = mlb.fit_transform(y)
print(mlb.classes_)

# Initialize the Bert tokenizer
BERT_MODEL_NAME = "bert-base-multilingual-cased"
# Bert_tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)   # 사전학습 토크나이저
Bert_tokenizer = BertTokenizer(vocab_file='./oa-6000-wpm-32000-vocab.txt', do_lower_case=False)     # custom 토크나이저

# Initialize the parameters that will be use for training
BATCH_SIZE = 32
MAX_LEN = 512
LR = 2e-05

# Instantiate and set up the data_module
OAdata_module = TestOADataModule(x_test=x, y_test=yt, tokenizer=Bert_tokenizer,
                                 batch_size=BATCH_SIZE, max_token_len=MAX_LEN)
OAdata_module.setup()

# model_path
model_path = 'D:/project/OA_paper/lightning_logs/version_k_fold_5/checkpoints/OA-epoch=19-val_loss=0.11.ckpt'
opt_thresh = float(open(savepath + "opt_thresh.txt", 'r').readline())

# Instantiate the classifier model
steps_per_epoch = len(x)//BATCH_SIZE
model = OAClassifier.load_from_checkpoint(model_path)

# Tokenize all contents in x_test
input_ids = []
attention_masks = []

for content in x:
    encoded_con = Bert_tokenizer.encode_plus(
        content,
        None,
        add_special_tokens=True,
        max_length=MAX_LEN,
        padding='max_length',
        return_token_type_ids=False,
        return_attention_mask=True,
        truncation=True,
        return_tensors='pt'
    )

    # Add the input_ids from encoded content to the list.
    input_ids.append(encoded_con['input_ids'])
    # Add its attention mask
    attention_masks.append(encoded_con['attention_mask'])

# Now convert the lists into tensors.
input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels = torch.tensor(yt)

# Create the DataLoader.
pred_data = TensorDataset(input_ids, attention_masks, labels)
pred_sampler = SequentialSampler(pred_data)
pred_dataloader = DataLoader(pred_data, sampler=pred_sampler, batch_size=BATCH_SIZE)

flat_pred_outs = 0
flat_true_labels = 0

# Put model in test mode
model = model.to(device)
model.eval()

# Tracking variables
pred_outs, true_labels = [], []

# Predict
for batch in tqdm(pred_dataloader):
    # Add batch to GPU
    batch = tuple(t.to(device).long() for t in batch)

    # Unpack the inputs from our dataloader
    b_input_ids, b_attn_mask, b_labels = batch

    with torch.no_grad():
        # Forward pass, calculate logit predictions
        pred_out = model(b_input_ids, b_attn_mask)
        pred_out = torch.sigmoid(pred_out)
        # Move predicted output and labels to CPU
        pred_out = pred_out.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
    # Store predictions and true labels
    pred_outs.append(pred_out)
    true_labels.append(label_ids)

# Combine the results across all batches.
flat_pred_outs = np.concatenate(pred_outs, axis=0)

# Combine the correct labels for each batch into a single list.
flat_true_labels = np.concatenate(true_labels, axis=0)

# convert probabilities into 0 or 1 based on a threshold value
def classify(pred_prob, thresh):
    y_pred = []
    for tag_label_row in pred_prob:
        temp = []
        for tag_label in tag_label_row:
            if tag_label >= thresh:
                temp.append(1)  # Infer tag value as 1 (present)
            else:
                temp.append(0)  # Infer tag value as 0 (absent)
        y_pred.append(temp)
    return y_pred

# convert labels to 1D array
y_true = flat_true_labels.ravel()

# predictions for optimal threshold
y_pred_labels = classify(flat_pred_outs, opt_thresh)
y_pred = np.array(y_pred_labels).ravel()  # Flatten

# report 작성
report = metrics.classification_report(y_true=y_true, y_pred=y_pred, output_dict=True)
report = pd.DataFrame(report).transpose()

report2 = metrics.classification_report(y_true=flat_true_labels, y_pred=y_pred_labels,
                                        target_names=['관념유사', '기타', '발음유사', '상품 불명확', '식별력', '외관유사'], output_dict=True)
report2 = pd.DataFrame(report2).transpose()

confusion = metrics.multilabel_confusion_matrix(y_true=flat_true_labels, y_pred=y_pred_labels)

# Creating multilabel confusion matrix
confusion = metrics.multilabel_confusion_matrix(flat_true_labels, y_pred_labels, labels=[0, 1, 2, 3, 4, 5])

# Plot confusion matrix
# 한글 폰트 사용을 위한 세팅
from matplotlib import font_manager, rc
font_path = "C:/Windows/Fonts/NGULIM.TTF"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)

labels = ['관념유사', '기타', '칭호유사', '상품 불명확', '식별력', '외관유사']

def print_confusion_matrix(confusion_matrix, axes, class_label, class_names, fontsize=14):

    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names,
    )

    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cbar=True, ax=axes, linewidths=.5)
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    axes.set_ylabel('True label')
    axes.set_xlabel('Predicted label')
    axes.set_title("class - " + class_label)

fig, ax = plt.subplots(2, 3, figsize=(12, 7))

for axes, cfs_matrix, label in zip(ax.flatten(), confusion, labels):
    print_confusion_matrix(cfs_matrix, axes, label, ["N", "Y"])

fig.tight_layout()

# confusion matrix plot 저장
plt.savefig(savepath + 'confusion_matrix.png')

y_pred = mlb.inverse_transform(np.array(y_pred_labels))
y_act = mlb.inverse_transform(flat_true_labels)

# 본문과 실제라벨, 예측라벨 결과
output = pd.DataFrame({'Body':x, 'Actual labels':y_act, 'Predicted labels':y_pred})

# 결과 저장
report.to_csv(savepath + 'report_test.csv', index=True, encoding='cp949')
report2.to_csv(savepath + 'report2_test.csv', index=True, encoding='cp949')
output.to_csv(savepath + 'output_test.csv', index=False, encoding='cp949')
