import logging as log

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split

import pytorch_lightning as pl
from test_tube import HyperOptArgumentParser
from sklearn.metrics import accuracy_score, f1_score

from transformers import glue_convert_examples_to_features as convert_examples_to_features
from transformers import BertModel, BertTokenizer
from transformers import DistilBertModel, DistilBertTokenizer
from transformers import AlbertModel, AlbertTokenizer

from inputs import text_reader


class BERT(pl.LightningModule):

    def __init__(self, hparams):
        super(BERT, self).__init__()
        self.hparams = hparams

        self.__load_dataset()
        self.__build_model()

    def __load_dataset(self):

        if self.hparams.pretrained_model.startswith('distilbert'):
            tokenizer = DistilBertTokenizer.from_pretrained(self.hparams.pretrained_model)
        elif self.hparams.pretrained_model.startswith('bert'):
            tokenizer = BertTokenizer.from_pretrained(self.hparams.pretrained_model)
        elif self.hparams.pretrained_model.startswith('albert'):
            tokenizer = AlbertTokenizer.from_pretrained(self.hparams.pretrained_model)
        else:
            raise ValueError('Unrecognized model name.')

        train_data, test_data, self.n_classes = text_reader.NewData(
            dataset_name=self.hparams.dataset,
            model='bert', root=self.hparams.data_dir,
            n_sample=self.hparams.n_sample, n_aug=self.hparams.n_aug, seed=self.hparams.seed,
        )

        train_features = convert_examples_to_features(train_data,
                                                      tokenizer,
                                                      label_list=range(self.n_classes),
                                                      max_length=self.hparams.max_length,
                                                      output_mode='classification',
                                                      pad_on_left=False,
                                                      pad_token=tokenizer.pad_token_id,
                                                      pad_token_segment_id=0)
        train_dataset = TensorDataset(torch.tensor([f.input_ids for f in train_features], dtype=torch.long),
                                      torch.tensor([f.attention_mask for f in train_features], dtype=torch.long),
                                      torch.tensor([f.token_type_ids for f in train_features], dtype=torch.long),
                                      torch.tensor([f.label for f in train_features], dtype=torch.long))

        n_train = int(0.90 * len(train_dataset))
        n_dev = len(train_dataset) - n_train

        self.train_dataset, self.dev_dataset = random_split(train_dataset, [n_train, n_dev])

        test_features = convert_examples_to_features(test_data,
                                                     tokenizer,
                                                     label_list=range(self.n_classes),
                                                     max_length=self.hparams.max_length,
                                                     output_mode='classification',
                                                     pad_on_left=False,
                                                     pad_token=tokenizer.pad_token_id,
                                                     pad_token_segment_id=0)

        self.test_dataset = TensorDataset(torch.tensor([f.input_ids for f in test_features], dtype=torch.long),
                                          torch.tensor([f.attention_mask for f in test_features], dtype=torch.long),
                                          torch.tensor([f.token_type_ids for f in test_features], dtype=torch.long),
                                          torch.tensor([f.label for f in test_features], dtype=torch.long))

    def __build_model(self):
        if self.hparams.pretrained_model.startswith('distilbert'):
            self.bert = DistilBertModel.from_pretrained(self.hparams.pretrained_model, output_attentions=True)
        elif self.hparams.pretrained_model.startswith('bert'):
            self.bert = BertModel.from_pretrained(self.hparams.pretrained_model, output_attentions=True)
        elif self.hparams.pretrained_model.startswith('albert'):
            self.bert = AlbertModel.from_pretrained(self.hparams.pretrained_model, output_attentions=True)
        else:
            raise ValueError('Unrecognized model name.')

        self.classifier = nn.Sequential(
            # nn.Dropout(self.hparams.dropout),
            # nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size),
            nn.Dropout(self.hparams.dropout),
            nn.Linear(self.bert.config.hidden_size, self.n_classes),
        )

    def forward(self, input_ids, attention_mask, token_type_ids):
        if self.hparams.pretrained_model.startswith('distilbert'):
            h, attn = self.bert(input_ids=input_ids,
                                attention_mask=attention_mask)
        elif self.hparams.pretrained_model.startswith('bert'):
            h, _, attn = self.bert(input_ids=input_ids,
                                   attention_mask=attention_mask,
                                   token_type_ids=token_type_ids)
        elif self.hparams.pretrained_model.startswith('albert'):
            h, _, attn = self.bert(input_ids=input_ids,
                                   attention_mask=attention_mask,
                                   token_type_ids=token_type_ids)
        else:
            raise ValueError('Unrecognized model name.')

        h_cls = h[:, 0]
        logits = self.classifier(h_cls)
        return logits, attn

    def training_step(self, batch, batch_idx):
        # REQUIRED
        input_ids, attention_mask, token_type_ids, y = batch
        y_hat, attn = self.forward(input_ids, attention_mask, token_type_ids)

        # calculate loss
        loss_train = F.cross_entropy(y_hat, y)

        tqdm_dict = {'train_loss': loss_train}
        output = {
            'loss': loss_train,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        }
        return output

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, token_type_ids, y = batch
        y_hat, attn = self.forward(input_ids, attention_mask, token_type_ids)

        loss_val = F.cross_entropy(y_hat, y)

        a, y_hat = torch.max(y_hat, dim=1)
        val_acc = accuracy_score(y_hat.cpu(), y.cpu())
        val_acc = torch.tensor(val_acc)

        return {'val_loss': loss_val, 'val_acc': val_acc}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_val_acc = torch.stack([x['val_acc'] for x in outputs]).mean()

        tqdm_dict = {'val_loss': avg_loss, 'val_acc': avg_val_acc}
        result = {'progress_bar': tqdm_dict, 'val_loss': avg_loss, 'log': tqdm_dict}

        return result

    def test_step(self, batch, batch_idx):
        input_ids, attention_mask, token_type_ids, y = batch
        y_hat, attn = self.forward(input_ids, attention_mask, token_type_ids)

        a, y_hat = torch.max(y_hat, dim=1)
        test_acc = accuracy_score(y.cpu(), y_hat.cpu())
        test_f1 = f1_score(y.cpu(), y_hat.cpu(), average='macro')

        output = {
            'test_acc': torch.tensor(test_acc),
            'test_f1': torch.tensor(test_f1)
        }

        return output

    def test_end(self, outputs):
        avg_test_acc = torch.stack([x['test_acc'] for x in outputs]).mean()
        avg_test_f1 = torch.stack([x['test_f1'] for x in outputs]).mean()

        log.info('test acc: %s' % avg_test_acc.cpu())
        log.info('test f1: %s' % avg_test_f1.cpu())
        tqdm_dict = {
            'test_acc': avg_test_acc,
            'test_f1': avg_test_f1
        }
        result = {'progress_bar': tqdm_dict, 'log': tqdm_dict}

        return result

    def configure_optimizers(self):
        parameters = [
            {
                "params": self.classifier.parameters(),
                "lr": self.hparams.learning_rate,
            },
            {
                "params": self.bert.parameters(),
                "lr": self.hparams.learning_rate,
            },
        ]
        return torch.optim.Adam(parameters, lr=self.hparams.learning_rate)

    def train_dataloader(self):
        log.info('Training data loader called.')
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size)

    def val_dataloader(self):
        log.info('Development data loader called.')
        return DataLoader(self.dev_dataset, batch_size=self.hparams.batch_size)

    def test_dataloader(self):
        log.info('Test data loader called.')
        return DataLoader(self.test_dataset, batch_size=self.hparams.batch_size)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = HyperOptArgumentParser(strategy=parent_parser.strategy, parents=[parent_parser])

        # dataset params
        parser.add_argument('--data_dir', default=None)
        parser.add_argument('--dataset', default=None)
        parser.add_argument('--n_sample', default=-1, type=float)
        parser.add_argument('--n_aug', default=-1, type=int)
        parser.add_argument('--seed', default=-1, type=int)
        parser.add_argument('--max_length', default=128, type=int)

        # training specific
        parser.add_argument('--max_epochs', default=5, type=int)
        parser.opt_list('--learning_rate', default=1e-3, type=float, options=[1e-3, 5e-3, 1e-4], tunable=True)
        parser.opt_list('--batch_size', default=32, type=int, options=[16, 32, 64], tunable=True)

        # model
        parser.add_argument('--pretrained_model', default='bert-base-uncased', type=str)
        parser.add_argument('--dropout', default=0.1, type=float)

        return parser

