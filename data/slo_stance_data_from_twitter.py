# coding=utf-8

import tweepy
import datetime
import io
import glob
import os

from config import MiningConfig
from data.utils import split_training_dev_data

date = datetime.datetime.today().strftime('%Y_%m_%d')

# input your credentials here
consumer_key = 'ohxFuI7vLTsIt3x7IJ8xyIDHz'
consumer_secret = 'hiMpRZ3R3Z2lnsYEbEGRDymWc0HIwC0y8Wip65QCQbhJu7mbyj'
access_token = '494744329-hORjipZ6hHIMkvq4AJngtlqfl1K2U3iLdOq4fOj0'
access_token_secret = 'r4ZwUH0w0eTNrygOYqpXUCGnP0KRzcMUNzxQQ1uGWKxKG'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)


def parse(tweet):
    if hasattr(tweet, 'retweeted_status'):
        tweet = tweet.retweeted_status
    _id = tweet.id_str
    created_at = str(tweet.created_at)
    text = tweet.full_text
    text = text.replace('\n', ' ')
    text = text.replace('\r', ' ')
    user_name = tweet.user.screen_name
    user_name = user_name.replace('\n', ' ')
    user_name = user_name.replace('\r', ' ')
    user_description = tweet.user.description
    user_description = user_description.replace('\n', ' ')
    user_description = user_description.replace('\r', ' ')

    return _id, created_at, user_name, user_description, text


def get_tweet_by_hashtag(tag, since):
    # Open/Create a file to append data
    id_set = set()
    with io.open(os.path.join(MiningConfig.raw_tweet_dir, 'hashtag_%s_%s.txt' % (tag, date)),
                 'w', encoding='utf-8') as f:
        for tweet in tweepy.Cursor(api.search, q='#%s' % tag,
                                   rpp=100,
                                   lang='en',
                                   since=since,
                                   tweet_mode='extended').items():
            _id, created_at, user_name, user_description, text = parse(tweet)
            if _id in id_set:
                continue
            id_set.add(_id)
            write_line = '\t'.join([_id, created_at, user_name, user_description, text])
            print(write_line)
            f.write(write_line)
            f.write('\n')


def get_tweet_by_user_id(tag, since=None):
    # Open/Create a file to append data
    with io.open(os.path.join(MiningConfig.raw_tweet_dir, 'user_%s_%s.txt' % (tag, date)),
                 'w', encoding='utf-8') as f:
        for tweet in tweepy.Cursor(api.user_timeline,
                                   id='%s' % tag,
                                   tweet_mode='extended').items():
            _id, created_at, user_name, user_description, text = parse(tweet)
            if since is not None:
                created_at_dt = datetime.datetime.strptime(created_at.split()[0], '%Y-%m-%d')
                since_dt = datetime.datetime.strptime(since, '%Y-%m-%d')
                if created_at_dt < since_dt:
                    break

            write_line = '\t'.join([_id, created_at, user_name, user_description, text])
            print(write_line)
            f.write(write_line)
            f.write('\n')


def prepare_training_data(input_dir, output_file_path):
    """
    Merge all tweets about different users/hashtags into one training file
    10/23/19: 65818 (20296, 25194, 20328)
    """
    id_tweet_against = {}
    id_tweet_favor = {}
    id_tweet_none = {}
    for filename in glob.glob(
            os.path.join(input_dir, 'user_*.txt')) + glob.glob(os.path.join(input_dir, 'hashtag_*.txt')
                                                               ):
        print('reading %s ...' % filename)
        if len(os.path.basename(filename).split('_')) != 5:
            raise Exception('Invalid filename format: %s' % filename)

        with io.open(filename, 'r', encoding='utf-8') as f:
            basename = os.path.basename(filename)
            for _line in f:
                _line = _line.strip().split('\t')
                if len(_line) != 5:
                    continue
                _id, dt, user_name, user_profile, text = _line
                if 'stop' in basename.lower() or basename.split('_')[1] in ['nonewcoal',
                                                                            'NoNewCoalMines']:
                    label = 'AGAINST'
                    # print(label, filename)
                    id_tweet_against[_id] = (dt, user_name, user_profile, text, label, basename)
                elif basename.split('_')[1] in ['GoAdani', 'AdaniOnline', 'bhp',
                                                'RioTinto', 'SantosLtd',
                                                'FortescueNews', 'kennecottutah',
                                                'NSWMC', 'CMEWA', 'QRCouncil', 'WoodsideEnergy']:
                    label = 'FAVOR'
                    # print(label, filename)
                    id_tweet_favor[_id] = (dt, user_name, user_profile, text, label, basename)
                elif basename.split('_')[1] in ['MiningNewsNet', 'ozmining',
                                                'miningcomau', 'MiningEnergySA',
                                                'AUMiningMonthly', 'MineralsCouncil',
                                                'Austmine', 'MiningWeeklyAUS',
                                                'AuMiningReview']:
                    label = 'NONE'
                    # print(label, filename)
                    id_tweet_none[_id] = (dt, user_name, user_profile, text, label, basename)
                else:
                    raise Exception('Twitter account not recognised.')

    all_id = set(id_tweet_against.keys()) | set(id_tweet_favor.keys()) | set(id_tweet_none.keys())
    print('A total of', len(all_id), 'tweets collected.')
    selected_id = all_id - \
                  (set(id_tweet_against.keys()) & set(id_tweet_favor.keys())) - \
                  (set(id_tweet_against.keys()) & set(id_tweet_none.keys())) - \
                  (set(id_tweet_favor.keys()) & set(id_tweet_none.keys()))

    print(len(selected_id), 'tweets selected.')

    print('building training data ...')
    n_against = n_favor = n_none = 0
    with io.open(output_file_path, 'w', encoding='utf-8') as f:
        f.write('ID\tTarget\tName\tProfile\tTweet\tStance\tSource\n')
        for _id, tweet in id_tweet_against.items():
            if _id in selected_id:
                f.write('\t'.join([_id, 'Mining Project', tweet[1], tweet[2], tweet[3], tweet[4], tweet[5]]))
                f.write('\n')
                n_against += 1
        for _id, tweet in id_tweet_favor.items():
            if _id in selected_id:
                f.write('\t'.join([_id, 'Mining Project', tweet[1], tweet[2], tweet[3], tweet[4], tweet[5]]))
                f.write('\n')
                n_favor += 1
        for _id, tweet in id_tweet_none.items():
            if _id in selected_id:
                f.write('\t'.join([_id, 'Mining Project', tweet[1], tweet[2], tweet[3], tweet[4], tweet[5]]))
                f.write('\n')
                n_none += 1
    print('done.')
    print(n_against, n_favor, n_none)
    assert sum([n_against, n_favor, n_none]) == len(selected_id)


def get_tweet_by_user_id_batch():
    with open('query_user') as f:
        for line in f:
            user_id = line.strip()
            if len(user_id) > 0:
                print(user_id)
                get_tweet_by_user_id(user_id, since='2019-03-20')


if __name__ == '__main__':
    # get_tweet_by_hashtag('StopBHP', since='2019-10-22')
    # get_tweet_by_user_id('WoodsideEnergy', since=None)

    # prepare_training_data(
    #     input_dir=MiningConfig.raw_tweet_dir,
    #     output_file_path=os.path.join(MiningConfig.data_dir, 'mining3.txt')
    # )

    # split_training_dev_data(input_file_path=os.path.join(MiningConfig.data_dir, 'mining3.txt'),
    #                         train_file_path=MiningConfig.train_raw_path % 'mining3',
    #                         dev_file_path=MiningConfig.dev_raw_path % 'mining3',
    #                         test_size=0.1)

    pass

