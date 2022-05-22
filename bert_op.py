import pandas as pd

import transformers
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup

import torch
from torch import nn, cuda
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler

import pytorch_lightning as pl


class OADataset(Dataset):
    def __init__(self, content, label, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.text = content
        self.labels = label
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, item_idx):
        text = self.text[item_idx]
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,  # Add [CLS] [SEP]
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=False,
            return_attention_mask=True,  # Differentiates padded vs normal token
            truncation=True,  # Truncate data beyond max length
            return_tensors='pt'  # PyTorch Tensor format
        )

        input_ids = inputs['input_ids'].flatten()
        attn_mask = inputs['attention_mask'].flatten()
        # token_type_ids = inputs["token_type_ids"]

        return {
            'input_ids': input_ids,
            'attention_mask': attn_mask,
            'label': torch.tensor(self.labels[item_idx], dtype=torch.float)
        }


class TrainOADataModule(pl.LightningDataModule):
    def __init__(self, x_tr, y_tr, x_val, y_val, tokenizer, batch_size=16, max_token_len=200):
        super().__init__()
        self.tr_text = x_tr
        self.tr_label = y_tr
        self.val_text = x_val
        self.val_label = y_val
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_token_len = max_token_len

    def setup(self):
        self.train_dataset = OADataset(content=self.tr_text, label=self.tr_label, tokenizer=self.tokenizer,
                                       max_len=self.max_token_len)
        self.val_dataset = OADataset(content=self.val_text, label=self.val_label, tokenizer=self.tokenizer,
                                     max_len=self.max_token_len)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=16)


class TestOADataModule(pl.LightningDataModule):
    def __init__(self, x_test, y_test, tokenizer, batch_size=16, max_token_len=200):
        super().__init__()
        self.test_text = x_test
        self.test_label = y_test
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_token_len = max_token_len

    def setup(self):
        self.test_dataset = OADataset(content=self.test_text, label=self.test_label, tokenizer=self.tokenizer,
                                      max_len=self.max_token_len)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=16)


class OAClassifier(pl.LightningModule):
    # Set up the classifier
    def __init__(self, n_classes=6, steps_per_epoch=None, n_epochs=3, lr=2e-5):
        super().__init__()

        self.bert = BertModel.from_pretrained("bert-base-multilingual-cased", return_dict=True)
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)  # outputs = number of labels
        self.steps_per_epoch = steps_per_epoch
        self.n_epochs = n_epochs
        self.lr = lr
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, input_ids, attn_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attn_mask)
        output = self.classifier(output.pooler_output)

        return output

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['label']

        outputs = self(input_ids, attention_mask)
        loss = self.criterion(outputs, labels)
        self.log('train_loss', loss, prog_bar=True, logger=True)

        return {"loss": loss, "predictions": outputs, "labels": labels}

    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['label']

        outputs = self(input_ids, attention_mask)
        loss = self.criterion(outputs, labels)
        self.log('val_loss', loss, prog_bar=True, logger=True)

        return loss

    def test_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['label']

        outputs = self(input_ids, attention_mask)
        loss = self.criterion(outputs, labels)
        self.log('test_loss', loss, prog_bar=True, logger=True)

        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr)
        warmup_steps = self.steps_per_epoch // 3
        total_steps = self.steps_per_epoch * self.n_epochs - warmup_steps

        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

        return [optimizer], [scheduler]
