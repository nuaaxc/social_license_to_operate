import fasttext
import os
import numpy as np
from config import MiningConfig
from sklearn.metrics import accuracy_score
from data.utils import prepare_fasttext_data
from sklearn.metrics import confusion_matrix
from pprint import pprint


def train_test(train_path, test_path, save=False):
    model = fasttext.train_supervised(train_path, lr=0.1, epoch=10)
    if save:
        model.save_model(MiningConfig.slo_case_study_relevance_model_path)
        print('model saved.')
    true_labels = []
    pred_labels = []
    for line in open(test_path):
        label, text = line.strip().split('\t')
        pred = model.predict(text)[0][0]
        true_labels.append(label)
        pred_labels.append(pred)
    acc = accuracy_score(true_labels, pred_labels)
    return acc


def train_predict(train_path, test_path, save_path=None):
    model = fasttext.train_supervised(train_path, lr=1, epoch=10)
    true_labels = []
    pred_labels = []

    results = []
    for line in open(test_path):
        label, text = line.strip().split('\t')
        pred = model.predict(text)[0][0]
        true_labels.append(label)
        pred_labels.append(pred)
        if label != pred:
            print(label)
            print(pred)
            print(text)
            print('-------------')
            if label == '__label__relevance' and pred == '__label__irrelevance':
                results.append('FN' + '\t' + text + '\n')
            if label == '__label__irrelevance' and pred == '__label__relevance':
                results.append('FP' + '\t' + text + '\n')

    if save_path:
        with open(save_path, 'w', encoding='utf-8') as f:
            f.writelines(results)

    acc = accuracy_score(true_labels, pred_labels)
    tn, fp, fn, tp = confusion_matrix(true_labels, pred_labels).ravel()
    print(acc)
    print(tn, fp, fn, tp)
    print('fpr:', fp / (fp + tn))
    print('fnr:', fn / (fn + tp))


def portion_train():
    # ratio = ''
    ratio = '_0.25'
    acc_all = []
    for fold in ['1', '2', '3', '4', '5']:
        acc = train_test(MiningConfig.rc_train_fold_path + '%s%s.txt' % (fold, ratio),
                         MiningConfig.rc_test_fold_path + '%s.txt' % fold)
        # MiningConfig.rc_dev_fold_path + '%s.txt' % fold)
        acc_all.append(acc)
    print(np.mean(acc_all), np.std(acc_all))


def full_train():
    acc_all = []
    for fold in ['1', '2', '3', '4', '5']:
        acc = train_test(train_path=MiningConfig.rc_train_fold_path + '%s.txt' % fold,
                         test_path=MiningConfig.rc_test_fold_path + '%s.txt' % fold,
                         save=True)
        acc_all.append(acc)
    print(np.mean(acc_all), np.std(acc_all))


def prediction():
    fold = '1'
    train_predict(MiningConfig.rc_train_fold_path + '%s.txt' % fold,
                  MiningConfig.rc_test_fold_path + '%s.txt' % fold,
                  MiningConfig.error_analysis_path % ('FASTTEXT', 'RELEVANCE')
                  )
    # train_predict(train_path=MiningConfig.rc_train_fold_path + '%s.txt' % fold,
    #               test_path=MiningConfig.rc_test_gold_path)


if __name__ == '__main__':
    # portion_train()
    # full_train()
    prediction()
