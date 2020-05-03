import json
import random
import hashlib
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
from config import MiningConfig
from data.pre_processing import clean_tweet_text
from data.utils import prepare_cv_data


def prepare_gold_test_set():
    with open(MiningConfig.rc_test_gold_path, 'w', encoding='utf-8') as f:
        for line in open(MiningConfig.rc_test_gold_raw_path, encoding='windows-1252', errors='ignore'):
            text, label = line.strip().split('\t')
            label = '__label__irrelevance' if label == 'Not relevant' else '__label__relevance'
            text = clean_tweet_text(text, remove_hashtag=False, remove_stopword=False)
            f.write(label + '\t' + text + '\n')
    print('done')


def get_company_data(topic):
    """
    3644
    910
    2
    394
    0
    940
    730
    6620
    ---
    3644
    19353
    2
    10267
    59
    23948
    3873
    61146
    """
    adani = []
    bhp = []
    riotinto = []
    woodside = []
    whitehavencoal = []
    santos = []
    fortescue = []
    _all = []

    # load relevant
    for line in open(MiningConfig.full_norm_path % topic):
        _, _, text = line.strip().split('\t')
        label = '__label__relevant'
        if 'adani' in text:
            adani.append((text, label))
            _all.append((text, label))
        elif 'bhp' in text:
            bhp.append((text, label))
            _all.append((text, label))
        elif 'riotinto' in text:
            riotinto.append((text, label))
            _all.append((text, label))
        elif 'woodside' in text:
            woodside.append((text, label))
            _all.append((text, label))
        elif 'whitehavencoal' in text:
            whitehavencoal.append((text, label))
            _all.append((text, label))
        elif 'santos' in text:
            santos.append((text, label))
            _all.append((text, label))
        elif 'fortescue' in text:
            fortescue.append((text, label))
            _all.append((text, label))

    for line in open(MiningConfig.irrelevance_path):
        text = line.strip()
        label = '__label__irrelevant'
        if 'adani' in text:
            adani.append((text, label))
            _all.append((text, label))
        elif 'bhp' in text:
            bhp.append((text, label))
            _all.append((text, label))
        elif 'riotinto' in text:
            riotinto.append((text, label))
            _all.append((text, label))
        elif 'woodside' in text:
            woodside.append((text, label))
            _all.append((text, label))
        elif 'whitehavencoal' in text:
            whitehavencoal.append((text, label))
            _all.append((text, label))
        elif 'santos' in text:
            santos.append((text, label))
            _all.append((text, label))
        elif 'fortescue' in text:
            fortescue.append((text, label))
            _all.append((text, label))

    print(len(adani))
    print(len(bhp))
    print(len(riotinto))
    print(len(woodside))
    print(len(whitehavencoal))
    print(len(santos))
    print(len(fortescue))
    print(len(_all))


def full_data_cv(topic):
    """
    1) combine relevance and irrelevance
    2) cv
    3) train/dev/test split
    """
    # load relevant
    relevance = []
    for line in open(MiningConfig.full_norm_path % topic):
        _, _, text = line.strip().split('\t')
        relevance.append(text)

    # load irrelevant
    irrelevance = []
    for line in open(MiningConfig.irrelevance_path):
        irrelevance.append(line.strip())

    x = relevance + irrelevance
    y = ['__label__relevance'] * len(relevance) + ['__label__irrelevance'] * len(irrelevance)

    prepare_cv_data(x, y,
                    5,
                    MiningConfig.rc_train_fold_path % topic,
                    MiningConfig.rc_dev_fold_path % topic,
                    MiningConfig.rc_test_fold_path % topic,
                    clean_tweet_text)

    print('done')


def irrelevance_data():
    unique_text = set()
    total_num = 62883
    with open(MiningConfig.json_file_path) as f_in, \
            open(MiningConfig.irrelevance_path, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            record = json.loads(line)
            if record.get('text') is None:
                continue

            label = record['relevance']
            if label != 'irrelevant':
                continue

            text = record['text']
            text = text.replace('\n', ' ')
            text = text.replace('\r', ' ')
            text = clean_tweet_text(text, remove_hashtag=True, remove_stopword=False)

            text_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()
            if text_hash not in unique_text:
                unique_text.add(text_hash)
                # non-english
                try:
                    lang = detect(text)
                except LangDetectException:
                    continue
                if lang != 'en':
                    continue
                # keywords
                if 'auspol' in text or 'adani' in text or \
                        'qldpol' in text or 'nswpol' in text \
                        or 'pollution' in text or 'riotinto' in text:
                    continue
                # randomness
                p = random.random()
                if p > 0.5:
                    continue

                f_out.write(text + '\n')

                total_num -= 1
                if total_num % 5000 == 0:
                    print(total_num)
                if total_num == 0:
                    break
    print('done')


if __name__ == '__main__':
    # irrelevance_data()
    # full_data_cv('mining3')
    # prepare_gold_test_set()
    # get_company_data('mining3')
    pass
