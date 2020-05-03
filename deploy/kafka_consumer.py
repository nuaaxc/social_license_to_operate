import json
import os
import torch
import langdetect
from kafka import KafkaConsumer, KafkaProducer
from langdetect.lang_detect_exception import LangDetectException
from kafka.structs import OffsetAndMetadata, TopicPartition
import time

from data import preprocess_tweet
from data.utils import *
from data.tokenization import tweet_tokenizer
from data.semeval import get_vocab
from config import SEConfig
from models.DStance import DStance

consumer = KafkaConsumer(bootstrap_servers=['130.155.204.198:9094'],
                         value_deserializer=lambda m: json.loads(m.decode('utf8')),
                         auto_offset_reset="earliest",
                         group_id='Chang-stance')
producer = KafkaProducer(bootstrap_servers='130.155.204.198:9094',
                         value_serializer=lambda v: json.dumps(v).encode('utf-8'))

consumer.subscribe(['slo-start'])

model_path = 'grid_search_best_checkpoint_DStance_mining2_0_50.pt'
vocab_path = 'vocab_mining2.pt'


class Opt:
    model_path = model_path
    tgt = 'mining'
    n_rnn_layers = 1


opt = Opt()

voc = get_vocab(vocab_path)
setattr(opt, 'd_voc', voc.vectors.shape[0])
setattr(opt, 'd_emb', voc.vectors.shape[1])
setattr(opt, 'v_emb', voc.vectors)

print('loading model ...')
best_checkpoint = torch.load(os.path.join(opt.model_path), map_location=lambda storage, location: storage)
setattr(opt, 'dropout', best_checkpoint['dropout'])
setattr(opt, 'd_rnn', best_checkpoint['d_rnn'])
setattr(opt, 'd_fc', best_checkpoint['d_fc'])

model = DStance(opt)
model.load_state_dict(best_checkpoint['model'])
model.eval()


def rule(_text):
    score = {}
    label = None
    try:
        if langdetect.detect(_text) != 'en':
            return 'NONE', {}

        _text = preprocess_tweet.preprocess(_text, remove=False)
        if len(_text) == 0:
            return 'NONE', {}

        for word in _text.split(' '):
            if 'stopadani' in word \
                    or 'climatechange' in word \
                    or 'climatestrike' in word \
                    or 'climateemergency' in word \
                    or 'nonewcoal' in word \
                    or 'stopbhp' in word \
                    or 'stopriotinto' in word \
                    or 'renewables' in word \
                    or 'stopfortescue' in word \
                    or 'stopsantos' in word:
                label = 'AGAINST'
                score[word] = 1.0
            elif 'goadani' in word:
                label = 'FAVOR'
                score[word] = 1.0

        return label, score

    except LangDetectException:
        return 'NONE', {}


def predict(text):
    label, score = rule(text)
    if label:
        return label, score
    batch = Batch(field_types=['text', 'text'], batch_size=1, voc=voc)
    tokenized_target = tweet_tokenizer(opt.tgt)
    tokenized_text = tweet_tokenizer(text)
    if len(tokenized_text) == 0:
        return 'None', {}

    batch_processed = batch.add([tokenized_target, tokenized_text])
    if batch_processed:
        (t, mask_t, lengths_t), (u, mask_u, lengths_u) = batch_processed
        y_, attention_score = model([t, u])
        attention_score = [s[0] for s in attention_score[0]]

        assert len(tokenized_text.split()) == len(attention_score)

        score = dict(zip(tokenized_text.split(), attention_score))
        label = SEConfig.stance_label_index_rev[y_.max(1)[1].cpu().numpy()[0]]
        return label, score


for msg in consumer:
    # do some work
    text = msg.value['text']

    print(text)
    stance_label, score = predict(text)
    print(stance_label)
    print(score)
    print('-----------------------------------------------')

    tp = TopicPartition(msg.topic, msg.partition)
    offsets = {tp: OffsetAndMetadata(msg.offset, None)}
    consumer.commit(offsets=offsets)

    # send a msg to next topic
    msg.value['chang-stance'] = 'supportive'

    time.sleep(1)
