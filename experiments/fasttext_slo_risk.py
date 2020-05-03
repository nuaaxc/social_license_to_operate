import fasttext
import numpy as np
from config import MiningConfig
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import confusion_matrix


def train_test_single_label(slo_value, _fold, save=False):
    model = fasttext.train_supervised(MiningConfig.slo_value_train_fold_path % slo_value + '%s.txt' % _fold,
                                      lr=0.1,
                                      epoch=25)
    if save:
        model.save_model(MiningConfig.slo_case_study_category_model_path % slo_value)
        print('model saved.')
    true_labels = []
    pred_labels = []
    for line in open(MiningConfig.slo_value_test_gold_path % slo_value):
        label, text = line.strip().split('\t')
        pred = model.predict(text)[0][0]
        true_labels.append(label)
        pred_labels.append(pred)
        if label != pred:
            print(label)
            print(pred)
            print(text)
            print('-------------')
    acc = accuracy_score(true_labels, pred_labels)
    micro = f1_score(true_labels, pred_labels, average='micro')
    macro = f1_score(true_labels, pred_labels, average='macro')
    tn, fp, fn, tp = confusion_matrix(true_labels, pred_labels).ravel()
    print('acc:', acc)
    print('micro:', micro)
    print('macro:', macro)
    print(tn, fp, fn, tp)
    print('fpr:', fp / (fp + tn))
    print('fnr:', fn / (fn + tp))
    return macro


def train_test_multi_label(train_path, test_path, test_path_output,
                           topic, fold, mode):
    model = fasttext.train_supervised(train_path % (topic, 'multi_label') + '%s.txt' % fold,
                                      lr=0.1,
                                      epoch=10,
                                      # wordNgrams=2,
                                      loss='ova')
    if mode == 'predict':
        with open(test_path_output, 'w', encoding='utf-8') as f:
            for line in open(test_path, encoding='utf-8'):
                _, _, text = line.strip().split('\t')
                result = model.predict(text, k=-1, threshold=0.5)
                aspect_label = '\t'.join(result[0]).strip()
                f.write(aspect_label + '\t' + line)
        print('done.')

    elif mode == 'test':
        result = model.test(test_path % (topic, 'multi_label') + '%s.txt' % fold, k=1)
        print(result)

    elif mode == 'exact':
        total = 0
        correct = 0
        for line in open(test_path % (topic, 'multi_label') + '%s.txt' % fold):
            line = line.strip().split('\t')
            label = line[:-1]
            text = line[-1]
            label = set(label)
            predict_label = set(model.predict(text, k=-1, threshold=0.5)[0])
            if label == predict_label:
                correct += 1
            total += 1
        print(correct / total)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    all_result = []
    for fold in ['1', '2', '3', '4', '5']:
        result = train_test_single_label('social', fold, save=True)
        # result = train_test_single_label('economic', fold, save=True)
        # result = train_test_single_label('environmental', fold, save=True)
        # result = train_test_single_label('other', fold, save=True)
        all_result.append(result)
    print('Averaging ...')
    print(np.mean(all_result), np.std(all_result))

    # train_test_multi_label(MiningConfig.slo_value_train_fold_path,
    #                                            MiningConfig.slo_value_test_fold_path,
    #                                            MiningConfig.slo_value_test_fold_path,
    #                                            'mining3', '1', 'exact')  # 0.9842
    # train_test_multi_label(MiningConfig.slo_value_train_fold_path,
    #                                            MiningConfig.slo_value_test_fold_path,
    #                                            MiningConfig.slo_value_test_fold_path,
    #                                            'mining3', '2', 'exact')  # 0.9875
    # train_test_multi_label(MiningConfig.slo_value_train_fold_path,
    #                                            MiningConfig.slo_value_test_fold_path,
    #                                            MiningConfig.slo_value_test_fold_path,
    #                                            'mining3', '3', 'exact')  # 0.9893
    # train_test_multi_label(MiningConfig.slo_value_train_fold_path,
    #                                            MiningConfig.slo_value_test_fold_path,
    #                                            MiningConfig.slo_value_test_fold_path,
    #                                            'mining3', '4', 'exact')  # 0.9858
    # train_test_multi_label(MiningConfig.slo_value_train_fold_path,
    #                                            MiningConfig.slo_value_test_fold_path,
    #                                            MiningConfig.slo_value_test_fold_path,
    #                                            'mining3', '5', 'exact')  # 0.9760

    # evaluate "test human file"
    # train_test_multi_label(MiningConfig.slo_value_train_fold_path,
    #                        MiningConfig.test_human_norm_path,
    #                        MiningConfig.test_human_valued_path,
    #                        'mining3', '1', 'predict')
    pass
