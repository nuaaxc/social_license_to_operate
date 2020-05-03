
# SLO Prototyping - Mining

 
## Description   
This project is about developing a prototype system (SIRTA - Social Insight via Real-Time Text Analytics) 
for assessing and monitoring an organisation's Social License to Operate 
(currently in the mining domain). Particularly, the methodology described here is
 designed with a focus on how to swiftly performing SLO assessment and monitoring 
 on a new domain/market of interest.
 
## SIRTA's Architecture
SIRTA consists of two processing modules: the SLO assessment engine and the SLO monitoring 
engine. The assessment engine is responsible for extracting opinions from social feed via text 
analytics. Specifically, it performs the pipeline of various text classification tasks to 
extract the opinions from a stream of posts. The opinions are then aggregated 
regularly (e.g., weekly) to produce the SLO scores, which are stored in a dedicated 
database for the next step in SIRTA. The monitoring engine keeps track of the time 
series of different organisations' scores as well as that of the overall market over 
time. It first computes a benchmark SLO time series representing the context (market) 
where those organisations operate. This allows one to see when an organisation's score 
departs significantly from the benchmark. To spot the departure, the monitoring engine 
applies quality control techniques to compute control limits for bounding 
an organisation's time series and a departure occurs when the benchmark falls out of the bound.

## Front-End Dashboard
1. **SLO Weekly Overview panel**: shows a list of real-time (weekly) numerical scores representing the SLO levels for the organisations under consideration.

2. **SLO Trend panel**: plots the long-term trend of the SLO level of a selected organisation (e.g., Adani Group), compared with the general trend of the market (all the organisations together).

3. **Social Feed panel**: lists the most recent social media posts (e.g., tweets) about the selected organisation, with both the stances and SLO risk categories (e.g., environment, social) identified.

##Back-End Dashboard
1. **Relevance classification**: finding posts contributing to the SLO assessment.

2. **Risk classification**: identifying SLO risk categories discussed in a post.

3. **Risk-aware stance classification**: detecting stances in the posts (risk-specific stances) and converting the stances into numerical SLO levels/scores.

## Data

### Stance classification
#### Training set
- Standard stance classification: data/files/mining3_stance_train_fold_{1-5}.txt
- Risk-aware stance classification:
    - Social: data/files/mining3_slo_valued_stance_social_train_fold_{1-5}.txt
    - Economic: data/files/mining3_slo_valued_stance_economic_train_fold_{1-5}.txt
    - Environmental: data/files/mining3_slo_valued_stance_environmental_train_fold_{1-5}.txt
    - Other: data/files/mining3_slo_valued_stance_other_train_fold_{1-5}.txt

#### Test set
- Standard stance classification: data/files/test_human_norm.txt
- Risk-aware stance classification:
    - Social: data/files/test_human_valued_social.txt
    - Economic: data/files/test_human_valued_economic.txt
    - Environmental: data/files/test_human_valued_environmental.txt
    - Other: data/files/test_human_valued_other.txt

#### Dataset preparation
- Standard stance classification: two steps are needed to prepare the above training/test data:
    - Raw data collection: the code is in data/slo_stance_data_from_twitter.py. There are two modes to query with Twitter APIs: by hashtag and by user_id, which can be done by running the function get_tweet_by_hashtag and get_tweet_by_user_id, respectively. After that, run prepare_training_data to merge data from all the hashtags and user_ids as a single file training file.
    - Training/test set creation: the code is in data/slo_stance_data.py. Use prepare_stance_training_data to get the training data, and use slo_valued_stance_human_test_data to get the test data.
- Risk-aware stance classification: the code is in data/slo_risk_data.py. Use prepare_slo_valued_stance_data to get training data for this task.

### Risk classification
#### Training set
- Social: data/files/mining3_slo_value_social_train_fold_{1-5}.txt
- Economic: data/files/mining3_slo_value_economic_train_fold_{1-5}.txt
- Environmental: data/files/mining3_slo_value_environmental_train_fold_{1-5}.txt
- Other: data/files/mining3_slo_value_other_train_fold_{1-5}.txt

#### Test set
- Social: data/files/slo_value_gold_test_set_social.txt
- Economic: data/files/slo_value_gold_test_set_economic.txt
- Environmental: data/files/slo_value_gold_test_set_environmental.txt
- Other: data/files/slo_value_gold_test_set_other.txt

#### Dataset preparation
The code in data/slo_risk_data.py. Use prepare_training_data to get the training data, 
and use prepare_gold_test_set to get the test data.

### Relevance classification
#### Training set
- 25% data: data/files/mining3_rc_train_fold_{1-5}_0.25.txt
- 50% data: data/files/mining3_rc_train_fold_{1-5}_0.5.txt
- 75% data: data/files/mining3_rc_train_fold_{1-5}_0.75.txt
- 100% data: data/files/mining3_rc_train_fold_{1-5}.txt

#### Test set
data/files/mining3_rc_test_fold_{1-5}.txt

#### Dataset preparation
The code is in slo_relevance_data.py. Use irrelevance_data to get irrelevance data and use 
full_data_cv to get the full dataset (containing both relevance and irrelevance data) for training 
the relevance classifier. Here the relevance means the data is relevant for the stance 
classification task (i.e., the training data of size 62883 collected for stance classification).

## Data Collection for New Domains
It is easy to get data for a new SLO domain from Twitter, such as AI and automation, by reusing the above source codes, which involves the following steps:

- Change the contents of the files query_hashtag and query_user to desirable lists of hashtags and user_ids, respectively;
- Run get_tweet_by_hashtag and get_tweet_by_user_id in data/slo_stance_data_from_twitter.py to get tweets containing any 
of the chosen hashtags or posted by any of the chosen user_ids. Then, run prepare_training_data to obtain the training set file.
- Run codes in slo_relevance_data.py, slo_risk_data.py, and slo_stance_data.py to prepare the training/test datasets for the respective text classification tasks.

Usually, if one expects an average performance, e.g., 70%~80%, try creating a large scale silver dataset for training 
(in the order of tens of thousands), which might normally take less than one week to find out good automated annotation rules 
(e.g., certain indicative hashtags). If one expects a good performance, e.g., 90%+, consider creating a gold standard 
human-annotated data (500~1000 would be good, might take 2 or 3 weeks' time to annotate) and use both gold and silver 
datasets for training. In addition, one can also use a lot of unlabelled data at the same time to train a classifier 
in a semi-supervised manner (together with the gold and silver data).

## Classification Model

### Models
Various neural network models can be used to train the classifiers for the above tasks. Here we implemented four such models, including:
1. **fastText**: an efficient classification model trained on word vectors created with subword information;
2. **BiLSTM**: a bidirectional LSTM trained on word vectors pretrained with GloVe word embeddings (glove.twitter.27B, 200d);
3. **CNN**: a convolutional neural network for sentence classification;
4. **BERT**: a general-purpose pre-training contextual model for sentence encoding and classification.

### Training Details

The following configurations were used to train the above classifiers:
1. **fastText**: learning rate of 0.1 was used, and the training did not stop until 10 epochs had passed;
2. **BiLSTM**: the hidden sizes of both LSTM and the followed dense layer were set to 256. A step learning rate scheduler was used, where the learning rate was set to 0.5 initially and then decayed by 10% after each epoch. A dropout layer was placed after the dense layer with a dropout rate of 0.3;
3. **CNN**: four 1D convolutional layers of 256 filters were chained as the sentence encoder, with the sequential filter sizes of 2, 3, 4, and 5. The same learning rate scheduler and dropout layer as those in BiLSTM were used;
4. **BERT**: the BERT(BASE, uncased) was used. The learning rate was set to 10e-5. The maximum number of wordpieces was set to 128. The batch size for each training step was 16 for BERT (due to GPU memory limits) and 128 for others. Early stopping was applied with the patience of 3.

### Implementation
The above four models are implemented in models/LSTM, models/CNN, models/BERT. fastText is implemented in a different fold, i.e., experiments/fasttext_*.py.
1. **fastText**: the codes for the three text classification tasks can be found in fasttext_slo_relevance.py (relevance classification), fasttext_slo_risk.py (risk classification), fasttext_slo_stance.py and fasttext_slo_risk_stance.py (risk-aware stance classification);
2. **BiLSTM**, **CNN**, and **BERT**: for each of these models, its model architecture details are specified in model.py while training-related details are specified in trainer.py;
3. Training scripts: can be found in the folder scripts;
4. IO: the codes for loading data into the models can be found in inputs/text_reader.py.

### Deployment
To deploy trained models onto the server, as the writing of this document (Apr 2020), one may need help from Brian, who is responsible for all the frontend and backend of SIRTA.

The scripts for models deployment can be found in deploy/kafka_consumer.py. Currently, a BiLSTM model is deployed as the classifier used for the stance classification task. To deploy new models, one can refer to the aforementioned script (deploy/kafka_consumer.py) for amendments specific to the target task.

## The trained models for the three text classification tasks
All the trained models for the three classification tasks, relevance classification, risk classification, and stance classification can be found here: https://bitbucket.csiro.au/users/xu052/repos/slo-mining/browse/saved_models

- relevance classification: fastText_relevance_model.bin for fastText model and logs_BERT_REL/version_0/checkpoints/epoch=0.ckpt for BERT model
- risk classification: 
    - logs_BERT_CATE_ECONOMIC/version_0/checkpoints/epoch=7.ckpt
    - logs_BERT_CATE_ENVIRONMENTAL/version_0/checkpoints/epoch=4.ckpt
    - logs_BERT_CATE_OTHER/version_0/checkpoints/epoch=4.ckpt
    - logs_BERT_CATE_SOCIAL/version_0/checkpoints/epoch=2.ckpt
- stance classification:
    - logs_BERT_STANCE_ECONOMIC/version_0/checkpoints/epoch=1.ckpt
    - logs_BERT_STANCE_ENVIRONMENTAL/version_0/checkpoints/epoch=1.ckpt
    - logs_BERT_STANCE_OTHER/version_0/checkpoints/epoch=1.ckpt
    - logs_BERT_STANCE_SOCIAL/version_0/checkpoints/epoch=1.ckpt
- risk-aware stance classification:
    - logs_BERT_CATE_STANCE_ECONOMIC/version_0/checkpoints/epoch=1.ckpt
    - logs_BERT_CATE_STANCE_ENVIRONMENTAL/version_0/checkpoints/epoch=1.ckpt
    - logs_BERT_CATE_STANCE_OTHER/version_0/checkpoints/epoch=1.ckpt
    - logs_BERT_CATE_STANCE_SOCIAL/version_0/checkpoints/epoch=1.ckpt

## Error Analysis
The error analysis results of the three text classification tasks can be found here: https://bitbucket.csiro.au/users/xu052/repos/slo-mining/browse/error_analysis
- Relevance classification: 
    - Format: FN means false-negative and FP mean false-positive
    - error_analysis_BERT_REL.txt (BERT)
    - error_analysis_FASTTEXT_RELEVANCE.txt (fastText)

- Risk classification: 

    - Format: FN means false-negative and FP mean false-positive
    - error_analysis_BERT_CATE_ECONOMIC.txt
    - error_analysis_BERT_CATE_ENVIRONMENTAL.txt
    - error_analysis_BERT_CATE_OTHER.txt
    - error_analysis_BERT_CATE_SOCIAL.txt
    
- Stance classification:

    - Format: True label –> Predicted label
    - error_analysis_BERT_STANCE_ECONOMIC.txt
    - error_analysis_BERT_STANCE_ENVIRONMENTAL.txt
    - error_analysis_BERT_STANCE_OTHER.txt
    - error_analysis_BERT_STANCE_SOCIAL.txt
    
- Risk-aware stance classification:

    - Format: True label –> Predicted label
    - error_analysis_BERT_CATE_STANCE_ECONOMIC.txt
    - error_analysis_BERT_CATE_STANCE_ENVIRONMENTAL.txt
    - error_analysis_BERT_CATE_STANCE_OTHER.txt
    - error_analysis_BERT_CATE_STANCE_SOCIAL.txt
    
## SLO Monitoring Engine
The changing nature of an organisation's operational context can impact its SLO score. Changes could be due, for example, to a change of a company's CEO or changes in the general trend of the overall market (for example, more awareness of climate change might lead to a generally negative stance towards mining overall). Assessing such changes thus requires that we keep track of not only the time series of a company's SLO scores but also the time series of scores of other companies operating in that market sector. This allows us to see when a company's score departs significantly from the average score across similar organisations, which can be seen as a benchmark of the context/market. Such information can drive strategic action at critical points in time. The SLO monitoring engine is then designed to track a comparable set of organisations across time.

To achieve this, it first obtains the market benchmark by averaging over all organisations' SLO time series. This ensures that the larger or more topical organisations (i.e., the ones that are discussed more often) do not dominate in the comparison. All organisations are thus comparable as they have faced the same market conditions over the same period. Then, the engine seeks to monitor the departure of each organisation's time series from the benchmark over a period of time (e.g., one week). The key technique used to achieve this is called quality chart. More details can be found in the paper (Section 2.3). 


