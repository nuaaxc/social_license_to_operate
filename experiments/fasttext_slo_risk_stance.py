import fasttext
import numpy as np
from config import MiningConfig
from pprint import pprint
from sklearn.metrics import f1_score, accuracy_score
# from scipy import stats


def train(slo_value, fold, save=False):
    model = fasttext.train_supervised(
        MiningConfig.slo_valued_stance_train_fold_path % slo_value + '%s.txt' % fold,
        lr=0.1,
        epoch=10)
    result = model.test_label(MiningConfig.slo_valued_stance_test_fold_path % slo_value + '%s.txt' % fold)
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
        model_path = MiningConfig.fasttext_slo_valued_stance_model_fold_path % slo_value + '%s.bin' % fold
        print('saving model %s ...' % model_path)
        model.save_model(model_path)
        print('done.')


def test_human(topic, _fold):
    models = {}
    for slo_value in ['social', 'economic', 'environmental', 'other']:
        models[slo_value] = fasttext.load_model(
            MiningConfig.fasttext_slo_valued_stance_model_fold_path % (topic, slo_value) + '%s.bin' % _fold
        )

    y_true = []
    y_predict = []
    for line in open(MiningConfig.test_human_valued_path):
        line = line.strip().split('\t')
        text = line[-1]
        # profile = line[-2]
        stance = line[-3]
        aspects = line[:-3]
        y_true.append('__label__' + stance)

        prediction = None
        predicted_score = 0
        for aspect in aspects:
            aspect = aspect[len('__label__'):]
            result = models[aspect].predict(text)
            label = result[0][0]
            score = result[1][0]
            if score > predicted_score:
                prediction = label
        y_predict.append(prediction)

    assert len(y_true) == len(y_predict)

    macro = f1_score(y_true, y_predict, average='macro')
    micro = f1_score(y_true, y_predict, average='micro')
    accuracy = accuracy_score(y_true, y_predict)
    print('macro:', macro)
    print('micro:', micro)
    print('accuracy:', accuracy)
    return macro, micro, accuracy


def test_human_per_slo_value(slo_value, _fold):
    model = fasttext.load_model(MiningConfig.fasttext_slo_valued_stance_model_fold_path % slo_value + '%s.bin' % _fold)
    # model = fasttext.load_model(MiningConfig.stance_model_fold_path % topic + '%s.bin' % _fold)
    y_true = []
    y_predict = []
    for line in open(MiningConfig.test_human_valued_path):
        line = line.strip().split('\t')
        text = line[-1]
        aspects = line[:-3]

        if '__label__' + slo_value in aspects:
            # profile = line[-2]
            stance = line[-3]
            result = model.predict(text)
            label = result[0][0]

            y_true.append('__label__' + stance)
            y_predict.append(label)
    print('# examples:', len(y_true))
    macro = f1_score(y_true, y_predict, average='macro')
    micro = f1_score(y_true, y_predict, average='micro')
    accuracy = accuracy_score(y_true, y_predict)
    print('macro:', macro)
    print('micro:', micro)
    print('accuracy:', accuracy)
    return macro, micro, accuracy


def experiment_test_human():
    # ===================== Testing ===================
    # macro: 0.5058818121580428 0.01042497624641105
    # micro: 0.5978102189781023 0.007776378011920616
    # accuracy: 0.5978102189781023 0.007776378011920616
    ma_all = []
    mi_all = []
    acc_all = []
    for fold in [1, 2, 3, 4, 5]:
        ma, mi, acc = test_human('mining3', fold)
        ma_all.append(ma)
        mi_all.append(mi)
        acc_all.append(acc)
    print('=========================')
    print('macro:', np.mean(ma_all), np.std(ma_all))
    print('micro:', np.mean(mi_all), np.std(mi_all))
    print('accuracy:', np.mean(acc_all), np.std(acc_all))


def experiment_test_human_per_category():
    # slo_value = 'social'
    # slo_value = 'economic'
    slo_value = 'environmental'
    # slo_value = 'other'
    ma_all = []
    mi_all = []
    acc_all = []
    for fold in [1, 2, 3, 4, 5]:
        ma, mi, acc = test_human_per_slo_value('mining3', slo_value, fold)
        ma_all.append(ma)
        mi_all.append(mi)
        acc_all.append(acc)
    print('=========================')
    print('macro:', np.mean(ma_all), np.std(ma_all))
    print('micro:', np.mean(mi_all), np.std(mi_all))
    print('accuracy:', np.mean(acc_all), np.std(acc_all))


if __name__ == '__main__':
    # train('mining3', 'social', '1', save=True)
    # train('mining3', 'social', '2', save=True)
    # train('mining3', 'social', '3', save=True)
    # train('mining3', 'social', '4', save=True)
    # train('mining3', 'social', '5', save=True)
    # (standard error of the mean)

    # train('mining3', 'economic', '1', save=True)
    # train('mining3', 'economic', '2', save=True)
    # train('mining3', 'economic', '3', save=True)
    # train('mining3', 'economic', '4', save=True)
    # train('mining3', 'economic', '5', save=True)
    #

    # train('mining3', 'environmental', '1', save=True)
    # train('mining3', 'environmental', '2', save=True)
    # train('mining3', 'environmental', '3', save=True)
    # train('mining3', 'environmental', '4', save=True)
    # train('mining3', 'environmental', '5', save=True)

    # train('mining3', 'other', '1', save=True)
    # train('mining3', 'other', '2', save=True)
    # train('mining3', 'other', '3', save=True)
    # train('mining3', 'other', '4', save=True)
    # train('mining3', 'other', '5', save=True)

    # experiment_test_human()
    experiment_test_human_per_category()

    pass
