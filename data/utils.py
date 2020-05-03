import io
import numpy as np
import random

from sklearn.model_selection import StratifiedKFold, train_test_split
from config import MiningConfig


def split_training_dev_data(input_file_path,
                            train_file_path,
                            dev_file_path,
                            index_label,
                            test_size):
    X = []
    y = []
    header = None
    with io.open(input_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('ID\t'):
                header = line
                continue
            X.append(line)
            y.append(line.strip().split('\t')[index_label])

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=777, stratify=y)

    with io.open(train_file_path, 'w', encoding='utf-8') as f:
        if header:
            f.write(header)
        f.writelines(X_train)

    with io.open(dev_file_path, 'w', encoding='utf-8') as f:
        if header:
            f.write(header)
        f.writelines(X_val)

    print('saved.')


def split_training_dev_test_data(input_file_path,
                                 train_file_path,
                                 dev_file_path,
                                 test_file_path,
                                 index_label,
                                 test_size):
    X = []
    y = []
    header = None
    with io.open(input_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('ID\t'):
                header = line
                continue
            X.append(line)
            y.append(line.strip().split('\t')[index_label])

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=test_size, random_state=777, stratify=y)
    X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train,
                                                      test_size=test_size, random_state=777, stratify=y_train)

    with io.open(train_file_path, 'w', encoding='utf-8') as f:
        if header:
            f.write(header)
        f.writelines(X_train)

    with io.open(dev_file_path, 'w', encoding='utf-8') as f:
        if header:
            f.write(header)
        f.writelines(X_dev)

    with io.open(test_file_path, 'w', encoding='utf-8') as f:
        if header:
            f.write(header)
        f.writelines(X_test)

    print('saved.')


def prepare_fasttext_data(input_file_path, output_file_path, index_text, index_label, has_header):
    print('reading from %s' % input_file_path)
    print('saving to %s' % output_file_path)
    with open(input_file_path, encoding='utf-8') as f_in, \
            open(output_file_path, 'w', encoding='utf-8') as f_out:
        if has_header:
            next(f_in)
        for line in f_in:
            line = line.strip().split('\t')
            text = line[index_text]
            label = line[index_label]
            ft_label = '__label__' + label

            f_out.write(text + '\t' + ft_label + '\n')
    print('saved.')


def prepare_cv_data(x, y, n_fold,
                    train_path, dev_path, test_path,
                    clean_text):

    x = np.array(x)
    y = np.array(y)

    skf = StratifiedKFold(n_splits=n_fold)

    index = 1
    for train_index, test_index in skf.split(x, y):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        x_train, x_dev, y_train, y_dev = train_test_split(x_train, y_train,
                                                          test_size=0.1, random_state=42,
                                                          stratify=y_train)

        assert len(x_train) == len(y_train)
        assert len(x_dev) == len(y_dev)
        assert len(x_test) == len(y_test)

        n_train = len(x_train)
        with open(train_path + '%s.txt' % index, 'w', encoding='utf-8') as f:
            for i in range(n_train):
                f.write(y_train[i] + '\t' + clean_text(x_train[i], remove_hashtag=True) + '\n')
        print('Saved %s to %s.' % (n_train, train_path + '%s.txt' % index))

        n_dev = len(x_dev)
        with open(dev_path + '%s.txt' % index, 'w', encoding='utf-8') as f:
            for i in range(n_dev):
                f.write(y_dev[i] + '\t' + clean_text(x_dev[i], remove_hashtag=True) + '\n')
        print('Saved %s to %s.' % (n_dev, dev_path + '%s.txt' % index))

        n_test = len(x_test)
        with open(test_path + '%s.txt' % index, 'w', encoding='utf-8') as f:
            for i in range(n_test):
                f.write(y_test[i] + '\t' + clean_text(x_test[i], remove_hashtag=True) + '\n')
        print('Saved %s to %s.' % (n_test, test_path + '%s.txt' % index))
        index += 1


def sample_data(input_file_path, output_file_path, ratio):
    print(input_file_path)
    print(output_file_path)

    with open(input_file_path) as f:
        lines = f.readlines()

    samples = random.sample(lines, int(len(lines) * ratio))

    with open(output_file_path, 'w', encoding='utf-8') as f:
        f.writelines(samples)
    print('done')


if __name__ == '__main__':
    ratio = 0.75
    for fold in ['1', '2', '3', '4', '5']:
        sample_data(input_file_path=MiningConfig.rc_train_fold_path % 'mining3' + '%s.txt' % fold,
                    output_file_path=MiningConfig.rc_train_fold_path % 'mining3' + '%s_%s.txt' % (fold, ratio),
                    ratio=ratio)
