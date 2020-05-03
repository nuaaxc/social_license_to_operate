import torch
import random
import numpy as np
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.logging import TestTubeLogger
from test_tube import HyperOptArgumentParser
from models.CNN.model import CNN
import config as cfg

seed = 2020
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)


def _train(_hparams):
    model = CNN(_hparams)

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


if __name__ == '__main__':
    parser = HyperOptArgumentParser(strategy='grid_search', add_help=False)
    parser.add_argument('--gpus', type=str, default=None)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--name', default='CNN', type=str)
    parser = CNN.add_model_specific_args(parser)
    hparams = parser.parse_args()

    if not torch.cuda.is_available():
        hparams.gpus = None

    if hparams.train:
        _train(hparams)
