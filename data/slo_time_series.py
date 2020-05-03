import json
from config import MiningConfig
from datetime import datetime, timedelta
from data.pre_processing import clean_tweet_text
from collections import defaultdict, OrderedDict


def json2csv():
    with open(MiningConfig.slo_time_series_csv_path, 'w', encoding='utf-8') as fout:
        fout.write('date,company,mean,overallMean,lower,upper\n')
        fin = open(MiningConfig.slo_time_series_json_path)
        for d in eval(fin.read()):
            d = dict(d)
            date = datetime(1970, 1, 1) + timedelta(milliseconds=d['date'])
            company = d['company']
            if company == 'adani':
                company = 'Adani Group'
            elif company == 'bhp':
                company = 'BHP'
            elif company == 'fortescue':
                company = 'Fortescue'
            elif company == 'riotinto':
                company = 'Rio Tinto'
            elif company == 'santos':
                company = 'Santos Limited'
            elif company == 'whitehavencoal':
                company = 'Whitehaven Coal'
            elif company == 'woodside':
                company = 'Woodside Energy Ltd'
            elif company == 'overall':
                company = 'Market trend (Mean)'
            else:
                continue

            mean = d.get('mean', '')
            overallMean = d.get('overallMean', '')
            lower = d.get('lower', '')
            upper = d.get('upper', '')
            fout.write(','.join([str(date.date()), company,
                                 str(mean), str(overallMean),
                                 str(lower), str(upper)]))
            fout.write('\n')
    print('done')


def csv2company():
    data = defaultdict(dict)
    for line in open(MiningConfig.slo_time_series_csv_path):
        if line.startswith('date'):
            continue
        line = line.strip().split(',')
        dt = line[0]
        company = line[1]
        mean = line[2]
        data[dt][company] = mean

    with open(MiningConfig.slo_time_series_company_csv_path, 'w') as f:
        f.write(','.join(['date', 'Adani Group', 'BHP',
                          'Fortescue', 'Rio Tinto', 'Santos Limited',
                          'Whitehaven Coal', 'Woodside Energy Ltd', 'Market trend (Mean)']) + '\n')
        for dt, company in data.items():
            f.write(','.join([dt, company['Adani Group'], company['BHP'], company['Fortescue'],
                              company['Rio Tinto'], company['Santos Limited'], company['Whitehaven Coal'],
                              company['Woodside Energy Ltd'], company['Market trend (Mean)']]) + '\n')

    print('done')


def csv2bhp():
    with open(MiningConfig.slo_time_series_bhp_csv_path, 'w') as f:
        for line in open(MiningConfig.slo_time_series_csv_path):
            if line.startswith('date'):
                f.write(line)
            company = line.strip().split(',')[1]
            if company != 'BHP':
                continue
            f.write(line)
    print('done.')


def extract_slo_text_series():
    with open(MiningConfig.slo_text_series_path, 'w', encoding='utf-8') as f:
        for line in open(MiningConfig.json_file_path):
            record = json.loads(line)
            company = record['company']
            if company != 'bhp':
                continue

            if record.get('text') is None:
                continue

            text = record['text']
            text = text.replace('\n', ' ')
            text = text.replace('\r', ' ')
            text = clean_tweet_text(text, remove_hashtag=False, remove_stopword=True)

            if len(text) == 0:
                continue

            publishedDate = record['publishedDate']

            f.write(publishedDate + '\t' + company + '\t' + text + '\n')

    print('done')


if __name__ == '__main__':
    json2csv()
    csv2company()
    # csv2bhp()
    # extract_slo_text_series()
