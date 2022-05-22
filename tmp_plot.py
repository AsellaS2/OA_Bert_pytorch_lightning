import pandas as pd
import numpy as np

from torch.utils.data import TensorDataset

from pytorch_lightning.callbacks import ModelCheckpoint

from sklearn import metrics
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from bert_op import *


RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

inputpath = 'D:/project/OA_paper/DATA/Part/'
savepath = 'D:/project/OA_paper/output/'

df = pd.read_csv(inputpath + 'Train.csv', encoding='cp949')

df['label'] = [i.replace('발음유사', '칭호유사') for i in df['label']]

x = list(df['rejectionContentDetail'])
y = list(map(lambda x: x.split(', '), df['label']))

x = x[:1000]
y = y[:1000]

mlb = MultiLabelBinarizer()
yt = mlb.fit_transform(y)
print(mlb.classes_)

# Split Train in to training and validation
x_tr, x_val, y_tr, y_val = train_test_split(x, yt, test_size=0.2, random_state=RANDOM_SEED, shuffle=True)

# Initialize the Bert tokenizer
BERT_MODEL_NAME = "bert-base-multilingual-cased"
Bert_tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)

# Initialize the parameters that will be use for training
N_EPOCHS = 1
BATCH_SIZE = 8
MAX_LEN = 300
LR = 2e-05

# Instantiate and set up the data_module
OAdata_module = TrainOADataModule(x_tr=x_tr, y_tr=y_tr, x_val=x_val, y_val=y_val, tokenizer=Bert_tokenizer,
                                  batch_size=BATCH_SIZE, max_token_len=MAX_LEN)
OAdata_module.setup()

# Instantiate the classifier model
steps_per_epoch = len(x_tr)//BATCH_SIZE
model = OAClassifier(n_classes=6, steps_per_epoch=steps_per_epoch,n_epochs=N_EPOCHS,lr=LR)
# model = OAClassifier.load_from_checkpoint('D:/project/OA_paper/lightning_logs/version_20/checkpoints/OA-epoch=00-val_loss=0.75.ckpt', n_classes=6, steps_per_epoch=steps_per_epoch, n_epochs=N_EPOCHS, lr=LR)

# saves a file like: OA-epoch=02-val_loss=0.32.ckpt
checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',                         # monitored quantity
    filename='OA-{epoch:02d}-{val_loss:.2f}',
    save_top_k=3,                               # save the top 3 models
    mode='min',                                 # mode of the monitored quantity for optimization
)

# Instantiate the Model Trainer
trainer = pl.Trainer(max_epochs=N_EPOCHS, gpus=1, callbacks=[checkpoint_callback], progress_bar_refresh_rate=30,
                     resume_from_checkpoint='D:/project/OA_paper/lightning_logs/version_20/checkpoints/OA-epoch=00-val_loss=0.75.ckpt')
# trainer = pl.Trainer(resume_from_checkpoint='D:/project/OA_paper/lightning_logs/version_20/checkpoints/OA-epoch=00-val_loss=0.75.ckpt')

# Train the Classifier Model
trainer.fit(model, OAdata_module)

# Evaluate the model performance on the validation dataset
trainer.validate(model, datamodule=OAdata_module)

# Retreive the checkpoint path for best model
model_path = checkpoint_callback.best_model_path

# Tokenize all contents in x_test
input_ids = []
attention_masks = []

for content in x_val:
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
labels = torch.tensor(y_val)

# Create the DataLoader.
pred_data = TensorDataset(input_ids, attention_masks, labels)
pred_sampler = SequentialSampler(pred_data)
pred_dataloader = DataLoader(pred_data, sampler=pred_sampler, batch_size=BATCH_SIZE)

flat_pred_outs = 0
flat_true_labels = 0

# Put model in evaluation mode
model = model.to(device)
model.eval()

# Tracking variables
pred_outs, true_labels = [], []

# Predict
for batch in pred_dataloader:
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

# define candidate threshold values
threshold = np.arange(0.4, 0.51, 0.01)

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

scores = []  # Store the list of f1 scores for prediction on each threshold

# convert labels to 1D array
y_true = flat_true_labels.ravel()
y_score = flat_pred_outs.ravel()

for thresh in threshold:
    # classes for each threshold
    pred_bin_label = classify(flat_pred_outs, thresh)

    # convert to 1D array
    y_pred = np.array(pred_bin_label).ravel()

    scores.append(metrics.f1_score(y_true, y_pred))

# find and save the optimal threshold
opt_thresh = threshold[scores.index(max(scores))]
# f = open(savepath + "opt_thresh.txt", 'w')
# f.write(str(opt_thresh))
# f.close()
print(f'Optimal Threshold Value = {opt_thresh}')

# predictions for optimal threshold
y_pred_labels = classify(flat_pred_outs, opt_thresh)
y_pred = np.array(y_pred_labels).ravel()  # Flatten

# report 작성
report = metrics.classification_report(y_true=y_true, y_pred=y_pred, output_dict=True)
report = pd.DataFrame(report).transpose()

# roc_auc = metrics.roc_auc_score(y_true=y_true, y_score=flat_pred_outs)
# print(roc_auc)

y_pred = mlb.inverse_transform(np.array(y_pred_labels))
y_act = mlb.inverse_transform(flat_true_labels)

output = pd.DataFrame({'Body':x_val,'Actual labels':y_act,'Predicted labels':y_pred})

# # 결과 저장
# report.to_csv(savepath + 'report_train.csv', index=True, encoding='cp949')
# output.to_csv(savepath + 'output_train.csv', index=False, encoding='cp949')
