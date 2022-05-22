import pandas as pd
import numpy as np
from tqdm import tqdm

from torch.utils.data import TensorDataset

from pytorch_lightning.callbacks import ModelCheckpoint

from sklearn import metrics
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split

from bert_op import *


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

inputpath = 'D:/project/OA_paper/DATA/Part/'
savepath = 'D:/project/OA_paper/output/'

df = pd.read_csv(inputpath + 'Test.csv', encoding='cp949')

x = list(df['rejectionContentDetail'])
y = list(map(lambda x: x.split(', '), df['label']))

x = x[:1000]
y = y[:1000]

mlb = MultiLabelBinarizer()
yt = mlb.fit_transform(y)
print(mlb.classes_)

# Initialize the Bert tokenizer
BERT_MODEL_NAME = "bert-base-multilingual-cased"
Bert_tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)

# Initialize the parameters that will be use for training
BATCH_SIZE = 8
MAX_LEN = 300
LR = 2e-05

# Instantiate and set up the data_module
OAdata_module = TestOADataModule(x_test=x, y_test=yt, tokenizer=Bert_tokenizer,
                                 batch_size=BATCH_SIZE, max_token_len=MAX_LEN)
OAdata_module.setup()

# model_path
model_path = './lightning_logs/version_20/checkpoints/OA-epoch=00-val_loss=0.75.ckpt'
opt_thresh = float(open("./output/opt_thresh.txt", 'r').readline())

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

y_pred = mlb.inverse_transform(np.array(y_pred_labels))
y_act = mlb.inverse_transform(flat_true_labels)

output = pd.DataFrame({'Body':x, 'Actual labels':y_act, 'Predicted labels':y_pred})

# 결과 저장
report.to_csv(savepath + 'report_test.csv', index=True, encoding='cp949')
output.to_csv(savepath + 'output_test.csv', index=False, encoding='cp949')
