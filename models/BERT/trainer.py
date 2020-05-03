import torch
import os
import random
import numpy as np
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.logging import TestTubeLogger
from test_tube import HyperOptArgumentParser
from sklearn.metrics import accuracy_score, f1_score
from models.BERT.model import BERT
from transformers import BertTokenizer
from transformers import DistilBertTokenizer
from transformers import AlbertTokenizer
from config import MiningConfig, LABEL_MAP


def seed_everything(seed=2020):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


seed_everything()


def _train(_hparams):
    model = BERT(_hparams)
    trainer = Trainer(logger=TestTubeLogger(save_dir=".", name="logs_{}_{}".format(_hparams.name, _hparams.dataset)),
                      gpus=hparams.gpus,
                      # row_log_interval=10,
                      # log_save_interval=100,
                      # fast_dev_run=True,
                      # overfit_pct=0.01,
                      use_amp=hparams.use_amp,
                      early_stop_callback=EarlyStopping(patience=3),
                      )
    trainer.fit(model)
    trainer.test()


def _test(_hparams):
    model = BERT.load_from_checkpoint(
        checkpoint_path=_hparams.weight_path,
        tags_csv=_hparams.cfg_path
    )
    print('model loaded.')
    model.eval()
    model.freeze()

    if _hparams.pretrained_model.startswith('distilbert'):
        tokenizer = DistilBertTokenizer.from_pretrained(_hparams.pretrained_model)
    elif _hparams.pretrained_model.startswith('bert'):
        tokenizer = BertTokenizer.from_pretrained(_hparams.pretrained_model)
    elif _hparams.pretrained_model.startswith('albert'):
        tokenizer = AlbertTokenizer.from_pretrained(_hparams.pretrained_model)
    else:
        raise ValueError('Unrecognized model name.')

    y_all, y_hat_all = [], []

    error_analysis_f = None
    if _hparams.error_analysis:
        error_analysis_f = open(MiningConfig.error_analysis_path % (_hparams.name, _hparams.dataset), 'w')

    for input_ids, attention_mask, token_type_ids, y in model.test_dataloader():
        y_hat, attn = model(input_ids, attention_mask, token_type_ids)
        a, y_hat = torch.max(y_hat, dim=1)
        for i in range(input_ids.size(0)):
            y_single = y.cpu().numpy()[i]
            y_hat_single = y_hat.cpu().numpy()[i]
            text = tokenizer.decode(input_ids[i]).\
                replace('[CLS]', '').replace('[SEP]', '').replace('[PAD]', '').\
                replace('\t', '').replace('\n', '').strip()
            y_all.append(y_single)
            y_hat_all.append(y_hat_single)

            if 'STANCE' not in _hparams.dataset:
                if _hparams.error_analysis:
                    if y_single == 0 and y_hat_single == 1:
                        error_analysis_f.write('FN' + '\t' + text + '\n')
                    if y_single == 1 and y_hat_single == 0:
                        error_analysis_f.write('FP' + '\t' + text + '\n')
            else:
                if _hparams.error_analysis:
                    if y_single != y_hat_single:
                        error_analysis_f.write('%s-->%s' % (LABEL_MAP['STANCE'][y_single], LABEL_MAP['STANCE'][y_hat_single])
                                               + '\t' + text + '\n')

    if _hparams.error_analysis:
        error_analysis_f.close()

    test_acc = accuracy_score(y_all, y_hat_all)
    test_f1 = f1_score(y_all, y_hat_all, average='macro')
    print(test_acc, test_f1)


if __name__ == '__main__':
    parser = HyperOptArgumentParser(strategy='grid_search', add_help=False)
    parser.add_argument('--gpus', type=str, default=None)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--error_analysis', action='store_true')
    parser.add_argument('--name', default='BERT', type=str)
    parser.add_argument('--use_amp', action="store_true")
    parser.add_argument('--weight_path', type=str, default=None)
    parser.add_argument('--cfg_path', type=str, default=None)
    parser = BERT.add_model_specific_args(parser)
    hparams = parser.parse_args()

    if not torch.cuda.is_available():
        hparams.gpus = None

    if hparams.train:
        _train(hparams)

    if hparams.test:
        _test(hparams)
