import logging
import torch
import io
import os
from torch.utils.data import Dataset
import torchtext
from torchtext.utils import download_from_url, extract_archive, unicode_csv_reader
from torchtext.data.utils import ngrams_iterator, get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.vocab import Vocab
from tqdm import tqdm

from transformers.data.processors.utils import InputExample
from config import LOC, LABELS


def _csv_iterator(data_path, ngrams, dataset_name=None, yield_cls=False):
    # tokenizer = get_tokenizer("basic_english")
    tokenizer = get_tokenizer("spacy", language="en_core_web_sm")
    with io.open(data_path, encoding="utf8") as f:
        reader = unicode_csv_reader(f, delimiter='\t')
        for row in reader:
            tokens = row[1]
            tokens = tokenizer(tokens)
            if yield_cls:
                label = int(LABELS[dataset_name][row[0]]) - 1
                yield label, ngrams_iterator(tokens, ngrams)
            else:
                yield ngrams_iterator(tokens, ngrams)


def _create_data_from_iterator(vocab, iterator, include_unk):
    data = []
    labels = []
    with tqdm(unit_scale=0, unit='lines') as t:
        for cls, tokens in iterator:
            if include_unk:
                tokens = torch.tensor([vocab[token] for token in tokens])
            else:
                token_ids = list(filter(lambda x: x is not Vocab.UNK, [vocab[token]
                                                                       for token in tokens]))
                tokens = torch.tensor(token_ids)
            if len(tokens) == 0:
                logging.info('Row contains no tokens.')
            data.append((cls, tokens))
            labels.append(cls)
            t.update(1)
    return data, set(labels)


class TextClassificationDataset(torch.utils.data.Dataset):
    """Defines an abstract text classification datasets.
       Currently, we only support the following datasets:
             - AG_NEWS
             - SogouNews
             - DBpedia
             - YelpReviewPolarity
             - YelpReviewFull
             - YahooAnswers
             - AmazonReviewPolarity
             - AmazonReviewFull
    """

    def __init__(self, vocab, data, labels):
        """Initiate text-classification dataset.
        Arguments:
            vocab: Vocabulary object used for dataset.
            data: a list of label/tokens tuple. tokens are a tensor after
                numericalizing the string tokens. label is an integer.
                [(label1, tokens1), (label2, tokens2), (label2, tokens3)]
                label: a set of the labels.
                {label1, label2}
        Examples:
            See the examples in examples/text_classification/
        """

        super(TextClassificationDataset, self).__init__()
        self._data = data
        self._labels = labels
        self._vocab = vocab

    def __getitem__(self, i):
        return self._data[i]

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        for x in self._data:
            yield x

    def get_labels(self):
        return self._labels

    def get_vocab(self):
        return self._vocab


def _create_examples(lines, set_type):
    examples = []
    label_set = set()
    for (i, line) in enumerate(lines):
        guid = "%s-%s" % (set_type, i)
        text = ' '.join(list(line[1]))
        label = line[0]
        label_set.add(label)
        examples.append(InputExample(guid=guid, text_a=text, label=label))
    return examples, len(label_set)


def _get_examples(data_path, ngrams, dataset_name, _type):
    return _create_examples(list(_csv_iterator(data_path, ngrams, dataset_name, yield_cls=True)), _type)


def _setup_datasets_bert(dataset_name, root=None, n_sample=None, n_aug=None, seed=None):
    if n_sample >= 0 and n_aug >= 0 and seed >= 0:
        train_csv_path = os.path.join(root, LOC[dataset_name]['train'].format(seed, n_sample, n_aug))
    else:
        train_csv_path = os.path.join(root, LOC[dataset_name]['train'])
    test_csv_path = os.path.join(root, LOC[dataset_name]['test'])

    logging.info('Loading training data ...')
    train_data, n_classes = _get_examples(train_csv_path, 1, dataset_name, 'train')
    logging.info('[{}] loaded'.format(len(train_data)))

    logging.info('Loading testing data ...')
    test_data, _ = _get_examples(test_csv_path, 1, dataset_name, 'test')
    logging.info('[{}] loaded'.format(len(test_data)))

    logging.info('Number of classes: [{}]'.format(n_classes))
    return train_data, test_data, n_classes


def _setup_datasets(dataset_name, word_vec_dir, root=None, ngrams=1, vocab=None,
                    include_unk=False, n_sample=None, n_aug=None, seed=None, pwe=0):
    if isinstance(LOC[dataset_name], str) and LOC[dataset_name].startswith('https'):
        dataset_tar = download_from_url(LOC[dataset_name], root=root)
        extracted_files = extract_archive(dataset_tar)
        for fname in extracted_files:
            if fname.endswith('train.csv'):
                train_csv_path = fname
            if fname.endswith('test.csv'):
                test_csv_path = fname
    else:
        if n_sample >= 0 and n_aug >= 0 and seed >= 0:
            train_csv_path = os.path.join(root, LOC[dataset_name]['train'].format(seed, n_sample, n_aug))
        else:
            train_csv_path = os.path.join(root, LOC[dataset_name]['train'])
        test_csv_path = os.path.join(root, LOC[dataset_name]['test'])

    word_emb_pretrained = None
    if vocab is None:
        logging.info('Building Vocab based on {}'.format(train_csv_path))
        vocab = build_vocab_from_iterator(_csv_iterator(train_csv_path, ngrams))
        if pwe:
            vocab.load_vectors('glove.twitter.27B.200d', cache=word_vec_dir)
            word_emb_pretrained = vocab.vectors
    else:
        if not isinstance(vocab, Vocab):
            raise TypeError("Passed vocabulary is not of type Vocab")
    logging.info('Vocab has {} entries'.format(len(vocab)))
    logging.info('Creating training data')
    train_data, train_labels = _create_data_from_iterator(
        vocab, _csv_iterator(train_csv_path, ngrams, dataset_name, yield_cls=True), include_unk)
    logging.info('Creating testing data')
    test_data, test_labels = _create_data_from_iterator(
        vocab, _csv_iterator(test_csv_path, ngrams, dataset_name, yield_cls=True), include_unk)
    if len(train_labels ^ test_labels) > 0:
        raise ValueError("Training and test labels don't match")
    return (TextClassificationDataset(vocab, train_data, train_labels),
            TextClassificationDataset(vocab, test_data, test_labels),
            word_emb_pretrained)


def NewData(**kwargs):
    if kwargs['model'] == 'bert':
        return _setup_datasets_bert(dataset_name=kwargs['dataset_name'],
                                    root=kwargs['root'],
                                    n_sample=kwargs['n_sample'],
                                    n_aug=kwargs['n_aug'],
                                    seed=kwargs['seed'])
    else:
        return _setup_datasets(dataset_name=kwargs['dataset_name'],
                               word_vec_dir=kwargs['word_vec_dir'],
                               root=kwargs['root'],
                               ngrams=kwargs['ngrams'],
                               vocab=kwargs['vocab'],
                               include_unk=False,
                               n_sample=kwargs['n_sample'],
                               n_aug=kwargs['n_aug'],
                               seed=kwargs['seed'],
                               pwe=kwargs['pwe'])


