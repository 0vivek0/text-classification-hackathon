import warnings
warnings.filterwarnings("ignore")

import os
import matplotlib.pyplot as plt
import copy
import logging
import evaluate
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder


from tqdm import tqdm
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding

from datasets import Dataset, DatasetDict
from transformers import DataCollatorWithPadding
from text_preprocessing import *

os.environ["TOKENIZERS_PARALLELISM"] = "false"
pd.set_option('max_colwidth', 400)

num_labels = 2
model_checkpoint = "distilroberta-base"
batch_size = 32
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint,use_fast=True)

dict_in_out = {"1":"Allowed", "0":"Restricted"}

def get_test_data_result(dataset, saved_model_name):
    test_data = pd.DataFrame(dataset["test"])
    test_data['Pred_Scope'] = np.nan
    test_data['Result'] = np.nan
    for i in range(len(test_data)):
        test_data['Pred_Scope'][i] = get_prediction_for_query(test_data['Text'][i], saved_model_name)
        if test_data['Pred_Scope'][i] == test_data['Scope'][i]:
            test_data['Result'][i] = "Pass"
        else:
            test_data['Result'][i] = "Fail"
    
    f1_metric = evaluate.load("f1")
    results = f1_metric.compute(predictions=test_data['Pred_Scope'], references=test_data['Scope'])
    logging.info(results)
    print(results)
    
    test_data.to_csv(f'bayer_result_{current_time}.csv')
    acc = (len(test_data[test_data['Result']=='Pass'])/len(test_data))*100
    return acc

def load_model_for_prediction(saved_model_name):
    model = TFAutoModelForSequenceClassification.from_pretrained(saved_model_name,num_labels = num_labels)
    return model

def get_prediction_for_query(details, saved_model_name):
  model = load_model_for_prediction(saved_model_name)
  tokenizer_output = tokenizer(details, truncation=True, return_token_type_ids=True, max_length = 50, return_tensors = 'tf')
  logits = model(**tokenizer_output)["logits"]
  predicted_class_id = int(tf.math.argmax(logits, axis=-1)[0])
  return predicted_class_id
