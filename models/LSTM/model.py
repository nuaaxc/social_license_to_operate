import logging as log

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split

import pytorch_lightning as pl
from test_tube import HyperOptArgumentParser
from sklearn.metrics import accuracy_score, f1_score

from inputs import text_reader


class LSTM(pl.LightningModule):

    def __init__(self, hparams):
        super(LSTM, self).__init__()
        self.hparams = hparams

        self.__load_dataset()
        self.__build_model()

    def __load_dataset(self):
        self.train_dataset, self.test_dataset, self.pretrained_weight = \
            text_reader.NewData(
                dataset_name=self.hparams.dataset,
                word_vec_dir=self.hparams.word_vec_dir,
                model='LSTM',
                root=self.hparams.data_dir, ngrams=1, vocab=None,
                n_sample=self.hparams.n_sample, n_aug=self.hparams.n_aug, seed=self.hparams.seed,
                pwe=self.hparams.pwe
            )
        self.vocab_size = len(self.train_dataset.get_vocab())
        self.n_classes = len(self.train_dataset.get_labels())

        train_len = int(len(self.train_dataset) * 0.9)
        self.train_dataset, self.dev_dataset = random_split(
            self.train_dataset, [train_len, len(self.train_dataset) - train_len]
        )

    def __build_model(self):
        self.embedding = nn.Embedding(self.vocab_size, self.hparams.embed_dim, scale_grad_by_freq=True,
                                      padding_idx=0)
        if self.pretrained_weight is not None:
            self.embedding.weight.data.copy_(self.pretrained_weight)
        self.lstm = nn.LSTM(self.hparams.embed_dim,
                            self.hparams.hidden_dim,
                            num_layers=self.hparams.n_layers,
                            # dropout=dropout,
                            bidirectional=True,
                            batch_first=True,
                            bias=True)
        self.fc1 = nn.Linear(self.hparams.hidden_dim * 2, self.hparams.hidden_dim)
        self.fc2 = nn.Linear(self.hparams.hidden_dim, self.n_classes)
        self.dropout = nn.Dropout(self.hparams.dropout)

    def forward(self, x, x_len):
        x = self.embedding(x)
        x = self.dropout(x)
        x = torch.nn.utils.rnn.pack_padded_sequence(x, x_len, batch_first=True)
        out, _ = self.lstm(x)
        out, input_sizes = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        out = torch.mean(out, dim=1)
        out = self.fc1(out)
        feat = out
        out = self.dropout(out)
        out = self.fc2(out)
        logits = F.log_softmax(out, dim=1)

        return logits, feat

    def training_step(self, batch, batch_idx):
        x, x_len, y = batch
        logits, _ = self.forward(x, x_len)

        # calculate loss
        loss_train = F.nll_loss(logits, y)

        tqdm_dict = {'train_loss': loss_train}
        output = {
            'loss': loss_train,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        }
        return output

    def validation_step(self, batch, batch_idx):
        x, x_len, y = batch
        logits, _ = self.forward(x, x_len)
        loss_val = F.nll_loss(logits, y)
        a, y_hat = torch.max(logits, dim=1)
        val_acc = accuracy_score(y_hat.cpu(), y.cpu())
        val_acc = torch.tensor(val_acc)
        return {'val_loss': loss_val, 'val_acc': val_acc}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_val_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        tqdm_dict = {'val_loss': avg_loss, 'val_acc': avg_val_acc}
        result = {'progress_bar': tqdm_dict, 'val_loss': avg_loss}
        return result

    def test_step(self, batch, batch_idx):
        x, x_len, y = batch
        logits, _ = self.forward(x, x_len)

        a, y_hat = torch.max(logits, dim=1)
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
        optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=self.hparams.scheduler_gamma)
        return [optimizer], [scheduler]

    def __dataloader(self, _type):
        if _type == 'train':
            dataset = self.train_dataset
        elif _type == 'dev':
            dataset = self.dev_dataset
        elif _type == 'test':
            dataset = self.test_dataset
        else:
            raise ValueError('Unrecognized dataset type. should be one of [train, dev, test]')

        if _type == 'train':
            loader = DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=True,
                                collate_fn=collate_batch)
        else:
            loader = DataLoader(dataset, batch_size=len(dataset), collate_fn=collate_batch)
        return loader

    @pl.data_loader
    def train_dataloader(self):
        log.info('Training data loader called.')
        return self.__dataloader(_type='train')

    @pl.data_loader
    def val_dataloader(self):
        log.info('Development data loader called.')
        return self.__dataloader(_type='dev')

    @pl.data_loader
    def test_dataloader(self):
        log.info('Test data loader called.')
        return self.__dataloader(_type='test')

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = HyperOptArgumentParser(strategy=parent_parser.strategy, parents=[parent_parser])

        # dataset params
        parser.add_argument('--data_dir', default=None)
        parser.add_argument('--word_vec_dir', default=None)
        parser.add_argument('--dataset', default=None)
        parser.add_argument('--n_sample', default=-1, type=float)
        parser.add_argument('--n_aug', default=-1, type=int)
        parser.add_argument('--pwe', default=0, type=int)  # pretrained word embedding

        # model params
        parser.add_argument('--embed_dim', default=200, type=int)
        parser.opt_list('--hidden_dim', default=64, type=int, options=[64, 128, 256], tunable=True)
        parser.add_argument('--n_layers', default=1, type=int)
        parser.opt_list('--dropout', default=0.1, type=float, options=[0.1, 0.2, 0.3, 0.5], tunable=True)

        # training params
        parser.add_argument('--max_epochs', default=5, type=int)
        parser.opt_list('--learning_rate', default=4.0, type=float, options=[0.05, 0.5, 5], tunable=True)
        parser.opt_list('--scheduler_gamma', default=0.9, type=float, options=[0.9, 0.5, 0.1], tunable=True)
        parser.opt_list('--batch_size', default=32, type=int, options=[16, 32, 64], tunable=True)
        parser.add_argument('--seed', default=-1, type=int)

        return parser


def collate_batch(batch):
    sorted_batch = sorted(batch, key=lambda x: x[1].shape[0], reverse=True)
    sequences = [x[1] for x in sorted_batch]
    sequences_padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)
    lengths = torch.LongTensor([len(x) for x in sequences])
    labels = torch.LongTensor([x[0] for x in sorted_batch])
    return sequences_padded, lengths, labels

