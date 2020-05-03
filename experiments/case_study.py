from datetime import datetime, timedelta, date
from collections import Counter
from wordcloud import WordCloud
import fasttext
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from config import MiningConfig


def extract_case_study_texts(start, end, img_name):
    print('loading models ...')
    model_rel = fasttext.load_model(MiningConfig.slo_case_study_relevance_model_path)
    model_cate_social = fasttext.load_model(MiningConfig.slo_case_study_category_model_path % 'social')
    model_cate_economic = fasttext.load_model(MiningConfig.slo_case_study_category_model_path % 'economic')
    model_cate_environmental = fasttext.load_model(MiningConfig.slo_case_study_category_model_path % 'environmental')
    model_cate_other = fasttext.load_model(MiningConfig.slo_case_study_category_model_path % 'other')
    model_stance = fasttext.load_model(MiningConfig.slo_case_study_stance_model_path)
    print('done.')

    f = open(MiningConfig.slo_text_series_path)
    lines = f.readlines()

    all_text = []
    all_category = []
    all_stance = []

    for line in lines:
        timestamp, company, text = line.strip().split('\t')

        rel = model_rel.predict(text)[0][0]
        social = model_cate_social.predict(text)[0][0]
        economic = model_cate_economic.predict(text)[0][0]
        environmental = model_cate_environmental.predict(text)[0][0]
        other = model_cate_other.predict(text)[0][0]
        stance = model_stance.predict(text)[0][0]

        # if rel == '__label__irrelevance':
        #     continue

        try:
            _date = (datetime(1970, 1, 1) + timedelta(milliseconds=int(timestamp))).date()
        except ValueError:
            _date = datetime.strptime(timestamp.split('T')[0], '%Y-%m-%d').date()

        text = text.replace('bhp', '')
        text = text.replace('billiton', '')
        text = text.replace('URL', '')
        text = text.replace('via', '')

        if start <= _date <= end:
            all_text.append(text)

            if social != '__label__none':
                all_category.append(social)
            if economic != '__label__none':
                all_category.append(economic)
            if environmental != '__label__none':
                all_category.append(environmental)
            if other != '__label__none':
                all_category.append(other)
            all_stance.append(stance)

    print(Counter(all_category))
    print(Counter(all_stance))
    print(len(all_text))

    wordcloud = WordCloud().generate(' '.join(all_text))

    plt.imshow(wordcloud, interpolation='bilinear')
    plt.tight_layout()
    plt.axis("off")
    plt.savefig(img_name, dpi=600)
    plt.show()


def draw_full_time_series():
    df = pd.read_csv(MiningConfig.slo_time_series_company_csv_path, index_col=0)
    print(df)
    for col in df.columns:
        df[col] = df[col].ewm(alpha=0.025).mean()
    ax = df.plot(legend=True, grid=False, figsize=(8, 5), fontsize=16)
    for i, line in enumerate(ax.lines):
        if i == len(ax.lines) - 1:
            line.set_linewidth(4)
            line.set_color('black')
            line.set_linestyle('-')
        else:
            line.set_linewidth(2)
            line.set_linestyle('-')

    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles=handles, labels=labels, fontsize=12, loc='upper left', bbox_to_anchor=(1.01, 1))
    labels = [item.get_text() for item in ax.get_xticklabels()]
    labels[1] = '2016'
    labels[2] = '2017'
    labels[3] = '2018'
    labels[4] = '2019'
    labels[5] = '2020'
    ax.set_xlabel('')
    ax.set_ylabel('EWMA', fontsize=20)
    ax.set_xticklabels(labels, rotation=45, fontsize=16)
    _, _, ymin, ymax = ax.axis()
    # print([a for a in ax.get_xticks()])
    ax.vlines(105, ymin=ymin, ymax=ymax, linestyles='--', color='grey', linewidth=2)
    ax.grid(b=True, which='major', axis='y', color='grey', linestyle='--', alpha=1)
    plt.tight_layout()
    plt.show()


def draw_bhp_time_series():
    df = pd.read_csv(MiningConfig.slo_time_series_bhp_csv_path)

    for col in df.columns:
        if col != 'company' and col != 'date':
            df[col] = df[col].ewm(alpha=0.05).mean()

    df = df.drop(columns=['company'])
    df = df.rename(columns={'mean': 'SLO score',
                            'overallMean': 'Mean',
                            'lower': 'LCL', 'upper': 'UCL'})

    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    plt.figure(figsize=(6, 6))
    plt.plot_date(df['date'], df['LCL'], '--', color='grey', alpha=0.2)
    plt.plot_date(df['date'], df['UCL'], '--', color='grey', alpha=0.2)
    plt.plot_date(df['date'], df['SLO score'], '-', color='orange', linewidth=2, label='BHP')
    plt.plot_date(df['date'], df['Mean'], '-', color='black', linewidth=4, label='Mean')
    plt.fill_between(df['date'], df['LCL'], df['UCL'],
                     where=df['UCL'] >= df['LCL'],
                     facecolor='grey', alpha=0.2, interpolate=True)
    plt.xlim((pd.to_datetime('2016-01-01', format='%Y-%m-%d', errors='ignore'),
              pd.to_datetime('2016-06-30', format='%Y-%m-%d', errors='ignore')))
    _, _, ymin, ymax = plt.axis()
    plt.vlines([pd.to_datetime('2016-03-01', format='%Y-%m-%d', errors='ignore')],
               ymin=ymin, ymax=ymax, linestyles='--', color='grey', linewidth=2)
    plt.xticks(rotation=45, fontsize=16)
    plt.yticks(fontsize=16)
    plt.ylabel('EWMA', fontsize=20)
    plt.legend(loc='lower right', fontsize=16)
    plt.tight_layout()
    plt.grid(b=True, which='major', axis='y', color='grey', linestyle='--', alpha=1)
    plt.show()


if __name__ == '__main__':
    extract_case_study_texts(date(2016, 1, 1), date(2016, 3, 1),
                             MiningConfig.slo_case_study_bhp_mariana_wc_path)      # mariana
    extract_case_study_texts(date(2019, 1, 25), date(2019, 2, 25),
                             MiningConfig.slo_case_study_bhp_brumadinho_wc_path)   # brumadinho
    draw_full_time_series()
    draw_bhp_time_series()
