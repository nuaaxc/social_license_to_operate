import os
from config import MiningConfig


def label_distribution(all_labels, label_types):
    lens = {}
    ratios = {}
    for label_type in label_types:
        sub_labels = [label for label in all_labels if label == label_type]
        lens[label_type] = len(sub_labels)
        ratios[label_type] = len(sub_labels) / len(all_labels)
    print(lens)
    print(ratios)
    print(sum(lens.values()))


def slo_label_distribution(input_file_path, index_stance, label_types):
    labels = []
    with open(input_file_path, encoding='utf-8') as f:
        for line in f:
            label = line.strip().split('\t')[index_stance]
            labels.append(label)
    label_distribution(all_labels=labels, label_types=label_types)


if __name__ == '__main__':
    # slo_value = 'environment'
    slo_value = 'community'
    # slo_value = 'health'
    # slo_value = 'economy'
    # slo_label_distribution(input_file_path=os.path.join(MiningConfig.data_dir, 'mining3_%s.txt' % slo_value),
    #                        index_stance=-2)
    slo_label_distribution(input_file_path=os.path.join(MiningConfig.data_dir, 'mining3_slo_values.txt'),
                           index_stance=-1,
                           label_types=['community', 'environment', 'health', 'economy'])
    slo_label_distribution(input_file_path=os.path.join(MiningConfig.data_dir, 'mining3_slo_values_train.txt'),
                           index_stance=-1,
                           label_types=['community', 'environment', 'health', 'economy'])
    slo_label_distribution(input_file_path=os.path.join(MiningConfig.data_dir, 'mining3_slo_values_dev.txt'),
                           index_stance=-1,
                           label_types=['community', 'environment', 'health', 'economy'])
    slo_label_distribution(input_file_path=os.path.join(MiningConfig.data_dir, 'mining3_slo_values_test.txt'),
                           index_stance=-1,
                           label_types=['community', 'environment', 'health', 'economy'])
