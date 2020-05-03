import fasttext
import numpy as np
from config import MiningConfig
from pprint import pprint
from sklearn.metrics import f1_score, accuracy_score
from scipy import stats


def train(fold, save=False):
    model = fasttext.train_supervised(
        MiningConfig.stance_train_fold_path + '%s.txt' % fold,
        lr=0.1,
        epoch=10)
    result = model.test_label(MiningConfig.stance_test_fold_path + '%s.txt' % fold)
    f1 = []
    p = []
    r = []
    for label, scores in result.items():
        print(label)
        pprint(scores)
        f1.append(scores['f1score'])
        p.append(scores['precision'])
        r.append(scores['recall'])
    print(np.mean(f1), np.mean(p), np.mean(r))

    if save:
        model.save_model(MiningConfig.slo_case_study_stance_model_path)
        print('model saved.')


def test_human(_fold):
    model = fasttext.load_model(MiningConfig.stance_model_fold_path + '%s.bin' % _fold)
    y_true = []
    y_predict = []
    for line in open(MiningConfig.test_human_valued_path):
        line = line.strip().split('\t')
        text = line[-1]
        # profile = line[-2]
        stance = line[-3]

        result = model.predict(text)
        y_predict.append(result[0][0])
        y_true.append('__label__' + stance)
    macro = f1_score(y_true, y_predict, average='macro')
    micro = f1_score(y_true, y_predict, average='micro')
    accuracy = accuracy_score(y_true, y_predict)
    print('macro:', macro)
    print('micro:', micro)
    print('accuracy:', accuracy)
    return macro, micro, accuracy


if __name__ == '__main__':
    # train('1', save=True)
    # train('2', save=True)
    # train('3', save=True)
    # train('4', save=True)
    # train('5', save=True)

    # =============== Testing =====================
    ma_all = []
    mi_all = []
    acc_all = []
    for _fold in [1, 2, 3, 4, 5]:
        ma, mi, acc = test_human(_fold)
        ma_all.append(ma)
        mi_all.append(mi)
        acc_all.append(acc)
    print('=========================')
    print('macro:', np.mean(ma_all), np.std(ma_all))
    print('micro:', np.mean(mi_all), np.std(mi_all))
    print('accuracy:', np.mean(acc_all), np.std(acc_all))

    pass
