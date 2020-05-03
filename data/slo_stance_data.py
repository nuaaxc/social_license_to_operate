import hashlib
from collections import Counter
from config import MiningConfig
from data.pre_processing import clean_dummy_text, clean_tweet_text
from data.utils import prepare_cv_data


def norm(in_file_path, out_file_path, clean_text, skip_header):
    print(in_file_path)
    print(out_file_path)
    unique_text = set()
    with open(in_file_path, 'r', encoding='utf-8') as f_in, \
            open(out_file_path, 'w', encoding='utf-8') as f_out:
        if skip_header:
            next(f_in)
        for line in f_in:
            line = line.strip().split('\t')
            if 'test_human' in in_file_path:
                text = line[2]
                profile = line[3]
                label = line[4]
            else:
                text = line[4]
                profile = line[3]
                label = line[5]

            text = clean_text(text, remove_hashtag=True, remove_stopword=False)
            text_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()    # compute a unique hash to deduplicate
            if text_hash not in unique_text:
                unique_text.add(text_hash)
                profile = clean_text(profile, remove_hashtag=True, remove_stopword=False)
                if len(text) > 0:
                    f_out.write(label + '\t' + profile + '\t' + text + '\n')
    print('done')


def sample_debug(in_file_path, out_file_path):
    global sample_size

    print(in_file_path)
    print(out_file_path)
    fn = an = nn = int(sample_size / 3)
    with open(in_file_path, 'r', encoding='utf-8') as f_in, \
            open(out_file_path, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            label = line.strip().split('\t')[0]
            if label == 'FAVOR' and fn > 0:
                f_out.write(line)
                fn -= 1
            if label == 'AGAINST' and an > 0:
                f_out.write(line)
                an -= 1
            if label == 'NONE' and nn > 0:
                f_out.write(line)
                nn -= 1

            if fn == 0 and an == 0 and nn == 0:
                break
    print('done')


def prepare_stance_training_data(input_file_path):
    print('Loading from', input_file_path)
    x = []
    y = []
    for line in open(input_file_path):
        label, profile, text = line.strip().split('\t')
        # x.append(profile + '\t' + text)
        x.append(text)
        y.append('__label__' + label)
    prepare_cv_data(x, y,
                    5,
                    MiningConfig.stance_train_fold_path,
                    MiningConfig.stance_dev_fold_path,
                    MiningConfig.stance_test_fold_path,
                    clean_dummy_text)
    print('done')


def data_stats():
    labels = []
    for line in open(MiningConfig.full_norm_path % 'mining3'):
        label, profile, text = line.strip().split('\t')
        labels.append(label)
    c = Counter(labels)
    print(c)


def slo_valued_stance_human_test_data():
    social = []
    economic = []
    environmental = []
    other = []

    for line in open(MiningConfig.test_human_valued_path):
        if '__label__social' in line:
            social.append(line)
        if '__label__economic' in line:
            economic.append(line)
        if '__label__environmental' in line:
            environmental.append(line)
        if '__label__other' in line:
            other.append(line)

    print(len(social), len(economic), len(environmental), len(other))

    with open(MiningConfig.test_human_valued_social_path, 'w', encoding='utf-8') as f:
        for line in social:
            line = line.strip().split('\t')
            tweet = line[-1]
            profile = line[-2]
            stance_label = line[-3]
            f.write('__label__' + stance_label + '\t' + tweet + '\n')

    with open(MiningConfig.test_human_valued_economic_path, 'w', encoding='utf-8') as f:
        for line in economic:
            line = line.strip().split('\t')
            tweet = line[-1]
            profile = line[-2]
            stance_label = line[-3]
            f.write('__label__' + stance_label + '\t' + tweet + '\n')

    with open(MiningConfig.test_human_valued_environmental_path, 'w', encoding='utf-8') as f:
        for line in environmental:
            line = line.strip().split('\t')
            tweet = line[-1]
            profile = line[-2]
            stance_label = line[-3]
            f.write('__label__' + stance_label + '\t' + tweet + '\n')

    with open(MiningConfig.test_human_valued_other_path, 'w', encoding='utf-8') as f:
        for line in other:
            line = line.strip().split('\t')
            tweet = line[-1]
            profile = line[-2]
            stance_label = line[-3]
            f.write('__label__' + stance_label + '\t' + tweet + '\n')

    print('done')


if __name__ == '__main__':
    # norm(MiningConfig.full_raw_path % topic, MiningConfig.full_norm_path % topic,
    #      clean_text=clean_tweet_text, skip_header=True)

    # norm(MiningConfig.test_raw_path, MiningConfig.test_norm_path,
    #      clean_text=clean_tweet_text, skip_header=True)

    # sample_size = 1000
    # sample_debug(MiningConfig.train_norm_path % topic,
    #              MiningConfig.train_sample_path % (topic, sample_size))

    # data_stats()

    # prepare_stance_training_data(MiningConfig.full_norm_path)
    # slo_valued_stance_human_test_data()

    pass
