import os

LOC = {

    'REL': {
        'dir': '',
        'train': 'mining3_rc_train_fold_4.txt',
        'test': 'mining3_rc_test_fold_4.txt',
    },
    # -------------------------------------
    # -------------------------------------
    # -------------------------------------
    'CATE_SOCIAL': {
        'dir': '',
        'train': 'mining3_slo_value_social_train_fold_1.txt',
        'test': 'slo_value_gold_test_set_social.txt',
    },
    'CATE_ECONOMIC': {
        'dir': '',
        'train': 'mining3_slo_value_economic_train_fold_1.txt',
        'test': 'slo_value_gold_test_set_economic.txt',
    },
    'CATE_ENVIRONMENTAL': {
        'dir': '',
        'train': 'mining3_slo_value_environmental_train_fold_1.txt',
        'test': 'slo_value_gold_test_set_environmental.txt',
    },
    'CATE_OTHER': {
        'dir': '',
        'train': 'mining3_slo_value_other_train_fold_1.txt',
        'test': 'slo_value_gold_test_set_other.txt',
    },
    # -------------------------------------
    # -------------------------------------
    # -------------------------------------
    'STANCE_SOCIAL': {
        'dir': '',
        'train': 'mining3_stance_train_fold_5.txt',
        'test': 'test_human_valued_social.txt'
    },
    'STANCE_ECONOMIC': {
        'dir': '',
        'train': 'mining3_stance_train_fold_5.txt',
        'test': 'test_human_valued_economic.txt'
    },
    'STANCE_ENVIRONMENTAL': {
        'dir': '',
        'train': 'mining3_stance_train_fold_5.txt',
        'test': 'test_human_valued_environmental.txt'
    },
    'STANCE_OTHER': {
        'dir': '',
        'train': 'mining3_stance_train_fold_5.txt',
        'test': 'test_human_valued_other.txt'
    },
    # -------------------------------------
    # -------------------------------------
    # -------------------------------------
    'CATE_STANCE_SOCIAL': {
        'dir': '',
        'train': 'mining3_slo_valued_stance_social_train_fold_1.txt',
        'test': 'test_human_valued_social.txt',
    },
    'CATE_STANCE_ECONOMIC': {
        'dir': '',
        'train': 'mining3_slo_valued_stance_economic_train_fold_1.txt',
        'test': 'test_human_valued_economic.txt',
    },
    'CATE_STANCE_ENVIRONMENTAL': {
        'dir': '',
        'train': 'mining3_slo_valued_stance_environmental_train_fold_1.txt',
        'test': 'test_human_valued_environmental.txt',
    },
    'CATE_STANCE_OTHER': {
        'dir': '',
        'train': 'mining3_slo_valued_stance_other_train_fold_1.txt',
        'test': 'test_human_valued_other.txt',
    }
}

LABELS = {
    'REL': {'__label__relevance': 1,
            '__label__irrelevance': 2},
    'CATE_SOCIAL': {'__label__social': 1,
                    '__label__none': 2},
    'CATE_ECONOMIC': {'__label__economic': 1,
                      '__label__none': 2},
    'CATE_ENVIRONMENTAL': {'__label__environmental': 1,
                           '__label__none': 2},
    'CATE_OTHER': {'__label__other': 1,
                   '__label__none': 2},
    'STANCE_SOCIAL': {'__label__FAVOR': 1,
                      '__label__AGAINST': 2,
                      '__label__NONE': 3},
    'STANCE_ECONOMIC': {'__label__FAVOR': 1,
                        '__label__AGAINST': 2,
                        '__label__NONE': 3},
    'STANCE_ENVIRONMENTAL': {'__label__FAVOR': 1,
                             '__label__AGAINST': 2,
                             '__label__NONE': 3},
    'STANCE_OTHER': {'__label__FAVOR': 1,
                     '__label__AGAINST': 2,
                     '__label__NONE': 3},
    'CATE_STANCE_SOCIAL': {'__label__FAVOR': 1,
                           '__label__AGAINST': 2,
                           '__label__NONE': 3},
    'CATE_STANCE_ECONOMIC': {'__label__FAVOR': 1,
                             '__label__AGAINST': 2,
                             '__label__NONE': 3},
    'CATE_STANCE_ENVIRONMENTAL': {'__label__FAVOR': 1,
                                  '__label__AGAINST': 2,
                                  '__label__NONE': 3},
    'CATE_STANCE_OTHER': {'__label__FAVOR': 1,
                          '__label__AGAINST': 2,
                          '__label__NONE': 3},
}

LABEL_MAP = {
    'STANCE': {
        0: 'FAVOR',
        1: 'AGAINST',
        2: 'NONE'
    }
}


class DirConfig(object):
    project_name = 'social_license_to_operate'
    home = str(os.path.expanduser('~'))
    if 'C:' in home:
        W2V_DIR = os.path.join(home, 'Downloads/dataset/word_vec/')
        GLOVE_840B_300D = os.path.join(W2V_DIR, 'glove.840B.300d.txt')
        GLOVE_TWITTER_27B_200D = os.path.join(W2V_DIR, 'glove.twitter.27B.200d.txt')
        BERT_VOC = os.path.join(home, 'Downloads/dataset/bert/bert-base-uncased-vocab.txt')
        BERT_MODEL = os.path.join(home, 'Downloads/dataset/bert/bert-base-uncased.tar.gz')
        ROOT_DIR = ''
        ROOT_LOCAL_DIR = ''
    elif 'chang' in home:
        W2V_DIR = '/home/xu052/glove/'
        GLOVE_840B_300D = ''
        BERT_VOC = os.path.join(home, 'Dropbox/resources/pretrained/bert/bert-base-uncased-vocab.txt')
        BERT_MODEL = os.path.join(home, 'Dropbox/resources/pretrained/bert/bert-base-uncased.tar.gz')
        BERT_VOC_TWITTER = os.path.join(home, 'Dropbox/resources/pretrained/TwitterBERT/vocab.txt')
        BERT_MODEL_TWITTER = os.path.join(home, 'Dropbox/resources/pretrained/TwitterBERT/')
        ROOT_DIR = os.path.join(home, 'Dropbox/project/%s/')
        ROOT_LOCAL_DIR = os.path.join(home, 'project/%s/')
    else:
        W2V_DIR = '/Users/xu052/Documents/project/glove/'
        GLOVE_840B_300D = ''
        BERT_VOC = ''
        BERT_MODEL = ''
        ROOT_DIR = ''
        ROOT_LOCAL_DIR = ''


class MiningConfig(DirConfig):
    corpus_name = 'mining'
    max_vocab_size = 10000
    max_seq_len = 30

    hp = {
        'mining3': {
            'lr': 0.00001,
            # 'lr': 0.00005,
            'epochs': 99,
            'batch_size': 32,
            'patience': 1,
            'd_hidden': 768,
            'dropout': 0.1,
            'cuda_device': 1,
            'max_pieces': 128,
            'tune_bert': True
        },
    }

    labels = [
        'FAVOR',
        'AGAINST',
        'NONE'
    ]

    n_label = len(labels)

    root = DirConfig.ROOT_DIR % DirConfig.project_name
    root_local = DirConfig.ROOT_LOCAL_DIR % DirConfig.project_name

    data_dir = os.path.join(root, 'data')

    model_dir = os.path.join(root_local, 'models')
    result_dir = os.path.join(root_local, 'results')
    raw_tweet_dir = os.path.join(root_local, 'raw_tweets')

    # data

    full_raw_path = os.path.join(data_dir, 'mining3.txt')
    full_norm_path = os.path.join(data_dir, 'mining3_norm.txt')
    json_file_path = os.path.join(data_dir, 'chang_slo.json')

    irrelevance_path = os.path.join(data_dir, 'irrelevance.txt')
    rc_train_fold_path = os.path.join(data_dir, 'mining3_rc_train_fold_')
    rc_dev_fold_path = os.path.join(data_dir, 'mining3_rc_dev_fold_')
    rc_test_fold_path = os.path.join(data_dir, 'mining3_rc_test_fold_')
    rc_test_gold_raw_path = os.path.join(data_dir, 'cecile_amt_agreed_relevance_training_set.tsv')
    rc_test_gold_path = os.path.join(data_dir, 'relevance_gold_test_set.txt')

    slo_value_train_fold_path = os.path.join(data_dir, 'mining3_slo_value_%s_train_fold_')
    slo_value_dev_fold_path = os.path.join(data_dir, 'mining3_slo_value_%s_dev_fold_')
    slo_value_test_fold_path = os.path.join(data_dir, 'mining3_slo_value_%s_test_fold_')
    slo_value_test_gold_raw_path = os.path.join(data_dir, 'tbl_training_set.tsv')
    slo_value_test_gold_path = os.path.join(data_dir, 'slo_value_gold_test_set_%s.txt')

    slo_valued_stance_train_fold_path = os.path.join(data_dir, 'mining3_slo_valued_stance_%s_train_fold_')
    slo_valued_stance_dev_fold_path = os.path.join(data_dir, 'mining3_slo_valued_stance_%s_dev_fold_')
    slo_valued_stance_test_fold_path = os.path.join(data_dir, 'mining3_slo_valued_stance_%s_test_fold_')

    stance_train_fold_path = os.path.join(data_dir, 'mining3_stance_train_fold_')
    stance_train_sample_path = os.path.join(data_dir, 'mining3_stance_train_sample_%s.txt')
    stance_dev_fold_path = os.path.join(data_dir, 'mining3_stance_val_fold_')
    stance_test_fold_path = os.path.join(data_dir, 'mining3_stance_test_fold_')

    test_human_path = os.path.join(data_dir, 'test_human.txt')
    test_human_norm_path = os.path.join(data_dir, 'test_human_norm.txt')
    test_human_valued_path = os.path.join(data_dir, 'test_human_valued.txt')
    test_human_valued_social_path = os.path.join(data_dir, 'test_human_valued_social.txt')
    test_human_valued_economic_path = os.path.join(data_dir, 'test_human_valued_economic.txt')
    test_human_valued_environmental_path = os.path.join(data_dir, 'test_human_valued_environmental.txt')
    test_human_valued_other_path = os.path.join(data_dir, 'test_human_valued_other.txt')

    slo_time_series_json_path = os.path.join(data_dir, 'slo_scores.json')
    slo_time_series_csv_path = os.path.join(data_dir, 'slo_scores.csv')
    slo_time_series_company_csv_path = os.path.join(data_dir, 'slo_scores_company.csv')
    slo_time_series_bhp_csv_path = os.path.join(data_dir, 'slo_scores_bhp.csv')
    slo_text_series_path = os.path.join(data_dir, 'slo_text_series.txt')
    slo_case_study_relevance_model_path = os.path.join(model_dir, 'slo_case_study_relevance_model.bin')
    slo_case_study_category_model_path = os.path.join(model_dir, 'slo_case_study_category_model_%s.bin')
    slo_case_study_stance_model_path = os.path.join(model_dir, 'slo_case_study_stance_model.bin')
    slo_case_study_bhp_mariana_path = os.path.join(data_dir, 'slo_case_study_bhp_mariana.txt')
    slo_case_study_bhp_brumadinho_path = os.path.join(data_dir, 'slo_case_study_bhp_brumadinho.txt')
    slo_case_study_bhp_mariana_wc_path = os.path.join(data_dir, 'slo_case_study_bhp_mariana_wc.png')
    slo_case_study_bhp_brumadinho_wc_path = os.path.join(data_dir, 'slo_case_study_bhp_brumadinho_wc.png')

    error_analysis_path = os.path.join(data_dir, 'error_analysis_%s_%s.txt')

    train_meta_path = os.path.join(result_dir, 'train_meta_%s_%s.th')
    test_meta_path = os.path.join(result_dir, 'test_meta_%s_%s.th')
    img_gen_feature_path = os.path.join(result_dir, 'img', 'gen_feature_%s.png')

    # model
    fasttext_slo_valued_stance_model_fold_path = os.path.join(model_dir,
                                                              'fasttext_mining3_slo_valued_stance_%s_model_fold_'
                                                              )
    stance_model_fold_path = os.path.join(model_dir, 'mining3_stance_model_fold_')
