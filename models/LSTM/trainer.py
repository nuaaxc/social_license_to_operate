import os
import torch
import random
import numpy as np
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.logging import TestTubeLogger
from test_tube import HyperOptArgumentParser
from sklearn.metrics import accuracy_score, f1_score
from models.LSTM.model import LSTM
import config as cfg

seed = 2020
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)


def _train(_hparams):
    model = LSTM(_hparams)
    trainer = Trainer(logger=TestTubeLogger(save_dir=".", name="logs_{}".format(_hparams.name)),
                      gpus=_hparams.gpus,
                      row_log_interval=10,
                      log_save_interval=100,
                      # fast_dev_run=True,
                      # overfit_pct=0.01,
                      early_stop_callback=EarlyStopping(patience=3),
                      )
    trainer.fit(model)
    trainer.test()


def _prediction(_hparams):
    pretrained_model = LSTM.load_from_metrics(
        weights_path=_hparams.weight_path,
        tags_csv=_hparams.cfg_path)
    pretrained_model.eval()
    pretrained_model.freeze()

    if _hparams.save_feature:
        for phase, loader in [('train', pretrained_model.train_dataloader()),
                              ('val', pretrained_model.val_dataloader()[0]),
                              ('test', pretrained_model.test_dataloader()[0])]:
            print(phase, '...')
            y_all, y_hat_all, feature_all = [], [], []
            for x, _len, y in loader:
                y_hat, feat = pretrained_model(x, _len)
                a, y_hat = torch.max(y_hat, dim=1)
                y_all.extend(y.cpu().numpy())
                y_hat_all.extend(y_hat.cpu().numpy())
                feature_all.extend(feat.cpu().numpy())
            print('saving ...')
            torch.save({'labels': y_all,
                        'features': feature_all},
                       os.path.join(_hparams.data_dir,
                                    cfg.LOC[_hparams.dataset]['dir'],
                                    '{}_{}_features.th'.format(_hparams.name, phase)))
            print('done.')

    y_all, y_hat_all = [], []
    for x, _len, y in pretrained_model.test_dataloader()[0]:
        y_hat, _ = pretrained_model(x, _len)
        a, y_hat = torch.max(y_hat, dim=1)
        y_all.extend(y.cpu().numpy())
        y_hat_all.extend(y_hat.cpu().numpy())
    test_acc = accuracy_score(y_all, y_hat_all)
    test_f1 = f1_score(y_all, y_hat_all, average='macro')
    print(test_acc, test_f1)


if __name__ == '__main__':
    parser = HyperOptArgumentParser(strategy='grid_search', add_help=False)
    parser.add_argument('--gpus', type=str, default=None)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--predict', action='store_true')
    parser.add_argument('--name', default='LSTM', type=str)
    parser.add_argument('--save_feature', action='store_true')
    parser.add_argument('--weight_path', type=str, default=None)
    parser.add_argument('--cfg_path', type=str, default=None)
    parser = LSTM.add_model_specific_args(parser)
    hparams = parser.parse_args()

    if not torch.cuda.is_available():
        hparams.gpus = None

    if hparams.train:
        _train(hparams)

    if hparams.predict:
        _prediction(hparams)
