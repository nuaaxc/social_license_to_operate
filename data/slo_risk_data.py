from pprint import pprint
import hashlib
from collections import defaultdict, Counter
from config import MiningConfig
from data.utils import prepare_cv_data
from data.pre_processing import clean_tweet_text, clean_dummy_text

social_value = ['culture', 'support', 'live', 'land', 'public', 'approve', 'traditional', 'humanity', 'licence',
                'human', 'labor', 'moral', 'national', 'land', 'donate', 'local', 'farm', 'vote', 'trust',
                'life', 'multinational', 'regional', 'party', 'generation']

economic_value = ['fund', 'financial', 'business', 'work', 'economic', 'import', 'money',
                  'job', 'employ', 'invest', 'spend', 'pay', 'market', 'cost', 'productivity',
                  'deposit', 'donation', 'asset']

environmental_value = ['environment', 'destroy', 'approve', 'reef', 'insanity', 'enviro', 'climate', 'danger',
                       'greatbarrierreef', 'climatechange', 'reefnotcoal', 'flood', 'renewable', 'river',
                       'groundwater', 'poison', 'agriculture', 'save', 'protect', 'threaten']

slo_value_queries = {
    # 7930
    # 'social': ['community', 'wellbeing', 'social', 'families', 'communication', 'member', 'support',
    #            'movement', 'family', 'donation', 'culture', 'citizen', 'education', 'people'],
    'social': social_value,
    # 10135
    # 'economic': ['job', 'economy', 'economic', 'employ', 'business', 'finance',
    #              'industry', 'training', 'investment', 'career', 'energy'],
    'economic': economic_value,
    # 6256
    # 'environmental': ['environment', 'climate', 'green', 'pollution', 'pollute',
    #                   'health', 'disease', 'sick', 'death', 'life', 'unhealthy'],
    'environmental': environmental_value,

    'other': social_value + economic_value + environmental_value,
}


def gold_test_stats():
    value_map = defaultdict(list)
    label_count = []
    social_c = 0
    economic_c = 0
    environmental_c = 0
    for line in open(MiningConfig.slo_value_test_gold_raw_path, encoding='windows-1252', errors='ignore'):
        line = line.strip().split('\t')
        text = line[0]
        text = clean_tweet_text(text, remove_hashtag=False, remove_stopword=True)
        if len(line) == 1 or len(line) == 2:
            label_count.append(1)
        elif len(line) == 3:
            label_count.append(2)
            if 'social' in line:
                social_c += 1
            if 'economic' in line:
                economic_c += 1
            if 'environmental' in line:
                environmental_c += 1
        else:
            label_count.append(3)
            if 'social' in line:
                social_c += 1
            if 'economic' in line:
                economic_c += 1
            if 'environmental' in line:
                environmental_c += 1

        if len(line) == 1:
            value_map['none'].append(text)
        if 'social' in line:
            value_map['social'].append(text)
        if 'economic' in line:
            value_map['economic'].append(text)
        if 'environmental' in line:
            value_map['environmental'].append(text)
    print(len(value_map['social']))
    print(len(value_map['economic']))
    print(len(value_map['environmental']))
    print(len(value_map['none']))

    print(social_c)
    print(economic_c)
    print(environmental_c)

    print(Counter(label_count))


def prepare_gold_test_set():
    value_map = defaultdict(list)

    for line in open(MiningConfig.slo_value_test_gold_raw_path, encoding='windows-1252', errors='ignore'):
        line = line.strip().split('\t')
        text = line[0]
        text = clean_tweet_text(text, remove_hashtag=False, remove_stopword=False)
        if len(line) == 1:
            value_map['other'].append(text)
        if 'social' in line:
            value_map['social'].append(text)
        if 'economic' in line:
            value_map['economic'].append(text)
        if 'environmental' in line:
            value_map['environmental'].append(text)
    print('social:', len(value_map['social']))
    print('economic:', len(value_map['economic']))
    print('environmental:', len(value_map['environmental']))
    print('other:', len(value_map['other']))

    for value in ['social', 'economic', 'environmental']:
        print('saving to "%s" ...' % value)
        with open(MiningConfig.slo_value_test_gold_path % value, 'w', encoding='utf-8') as f:
            for line in value_map[value]:
                f.write('__label__' + value + '\t' + line + '\n')
            unique_text = set()
            for other_value in ['social', 'economic', 'environmental']:
                if other_value != value:
                    for line in value_map[other_value]:
                        text_hash = hashlib.sha256(line.encode('utf-8')).hexdigest()
                        if text_hash not in unique_text:
                            unique_text.add(text_hash)
                            f.write('__label__none' + '\t' + line + '\n')
        print('done')

    print('saving to "other" ...')
    with open(MiningConfig.slo_value_test_gold_path % 'other', 'w', encoding='utf-8') as f:
        for line in value_map['other']:
            f.write('__label__other' + '\t' + line + '\n')
        unique_text = set()
        for other_value in ['social', 'economic', 'environmental']:
            for line in value_map[other_value]:
                text_hash = hashlib.sha256(line.encode('utf-8')).hexdigest()
                if text_hash not in unique_text:
                    unique_text.add(text_hash)
                    f.write('__label__none' + '\t' + line + '\n')
    print('done')


def query_data(text, queries):
    text = text.lower()
    if any(q in text for q in queries):
        return True
    else:
        return False


def prepare_training_data(input_file_path,
                          queries,
                          _slo_value):
    print('Loading from', input_file_path)
    x = []
    y = []
    with open(input_file_path, 'r', encoding='utf-8') as f_in:
        next(f_in)
        for line in f_in:
            _, _, text = line.strip().split('\t')
            x.append(text)
            if query_data(text=text, queries=queries[_slo_value]):
                if _slo_value == 'other':
                    y.append('__label__none')  # hit all categories
                else:
                    y.append('__label__' + _slo_value)
            else:
                if _slo_value == 'other':
                    y.append('__label__other')
                else:
                    y.append('__label__none')

    print('Hits:', len([l for l in y if l == '__label__' + _slo_value]))

    prepare_cv_data(x, y,
                    5,
                    MiningConfig.slo_value_train_fold_path % _slo_value,
                    MiningConfig.slo_value_dev_fold_path % _slo_value,
                    MiningConfig.slo_value_test_fold_path % _slo_value,
                    clean_dummy_text)

    print('done')


def label_stats(input_file_path, queries):
    c_social = 0
    c_economic = 0
    c_environmental = 0
    c_none = 0
    with open(input_file_path, 'r', encoding='utf-8') as f_in:
        for line in f_in:
            _, _, text = line.strip().split('\t')
            has_label = False
            if query_data(text=text, queries=queries['social']):
                c_social += 1
                has_label = True
            if query_data(text=text, queries=queries['economic']):
                c_economic += 1
                has_label = True
            if query_data(text=text, queries=queries['environmental']):
                c_environmental += 1
                has_label = True
            if has_label is False:
                c_none += 1
    print(c_social)
    print(c_economic)
    print(c_environmental)
    print(c_none)


def multi_label_stats(input_file_path, queries):
    from collections import Counter
    y = []
    with open(input_file_path, 'r', encoding='utf-8') as f_in:
        next(f_in)
        for line in f_in:
            _, _, text = line.strip().split('\t')
            label = ''
            if query_data(text=text, queries=queries['social']):
                label = '__label__social' + '\t' + label
            if query_data(text=text, queries=queries['economic']):
                label = '__label__economic' + '\t' + label
            if query_data(text=text, queries=queries['environmental']):
                label = '__label__environmental' + '\t' + label
            label = label.strip()
            if label == '':
                label = '__label__none'
            y.append(label)

    c = Counter([len(_y.strip().split('\t')) for _y in y])
    print(c)


def prepare_slo_valued_stance_data(input_file_path,
                                   queries,
                                   _slo_value):
    print('Loading from', input_file_path)
    x = []
    y = []
    with open(input_file_path, 'r', encoding='utf-8') as f_in:
        next(f_in)
        for line in f_in:
            label, profile, text = line.strip().split('\t')
            if _slo_value != 'other':
                if query_data(text=text, queries=queries[_slo_value]):
                    # x.append(profile + '\t' + text)
                    x.append(text)
                    y.append('__label__' + label)
            else:
                if not query_data(text=text, queries=queries[_slo_value]):
                    # x.append(profile + '\t' + text)
                    x.append(text)
                    y.append('__label__' + label)

    assert len(x) == len(y)
    print(len(x), len(y))

    prepare_cv_data(x, y,
                    5,
                    MiningConfig.slo_valued_stance_train_fold_path % slo_value,
                    MiningConfig.slo_valued_stance_dev_fold_path % slo_value,
                    MiningConfig.slo_valued_stance_test_fold_path % slo_value,
                    clean_dummy_text)
    print('done')


if __name__ == '__main__':
    slo_value = 'social'
    # slo_value = 'economic'
    # slo_value = 'environmental'
    # slo_value = 'other'
    # prepare_training_data(input_file_path=MiningConfig.full_norm_path % slo_value,
    #                       queries=slo_value_queries,
    #                       _slo_value=slo_value)
    #
    prepare_gold_test_set()
    #
    # prepare_slo_valued_stance_data(input_file_path=MiningConfig.full_norm_path,
    #                                queries=slo_value_queries,
    #                                _slo_value=slo_value)

    # label_stats(input_file_path=MiningConfig.full_norm_path % topic,
    #             queries=slo_value_queries)

    # multi_label_train_stats(input_file_path=MiningConfig.full_norm_path % topic,
    #                   queries=slo_value_queries)  # {1: 56605, 2: 5654, 3: 623}

    # gold_test_stats()


    pass
