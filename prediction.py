import warnings
warnings.filterwarnings("ignore")
import os
import numpy as np
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from text_preprocessing import *
from  datetime import datetime

os.environ["TOKENIZERS_PARALLELISM"] = "false"
pd.set_option('max_colwidth', 400)

num_labels = 2
model_checkpoint = "distilroberta-base"
batch_size = 32
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint,use_fast=True)
current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

dict_in_out = {"1":"Allowed", "0":"Restricted"}

def get_test_data_result(dataset, saved_model_name):
    """
    Evaluate the performance of a model on a test dataset by generating predictions 
    and comparing them to the expected outcomes.

    This function takes a dataset containing test examples, computes predictions 
    for each example using a specified model, and determines whether each prediction 
    matches the expected scope, calculating the overall accuracy. The results 
    are saved to a CSV file.

    Parameters:
    dataset (dict): A dictionary containing the test data under the key "test".
    saved_model_name (str): The name of the saved model used for generating predictions.

    Returns:
    float: The accuracy of predictions as a percentage.
    """
    test_data = pd.DataFrame(dataset["test"])
    test_data['Pred_Scope'] = np.nan
    test_data['Result'] = np.nan
    for i in range(len(test_data)):
        test_data['Pred_Scope'][i] = get_prediction_for_query(test_data['Text'][i], saved_model_name)
        if test_data['Pred_Scope'][i] == test_data['Scope'][i]:
            test_data['Result'][i] = "Pass"
        else:
            test_data['Result'][i] = "Fail"

    test_data.to_csv(f'bayer_result_{current_time}.csv')
    acc = len(test_data[test_data['Result']=='Pass'])/len(test_data)*100
    return acc

def load_model_for_prediction(saved_model_name):
    """
    Load a pre-trained sequence classification model for prediction tasks.
    This function retrieves a specified model from the Hugging Face model hub 
    and initializes it for sequence classification with a predefined number 
    of output labels.

    Parameters:
    saved_model_name (str): The name of the pre-trained model to load.

    Returns:
    TFAutoModelForSequenceClassification: The loaded sequence classification model.
    """
    model = TFAutoModelForSequenceClassification.from_pretrained(saved_model_name,num_labels = num_labels)
    return model

def get_prediction_for_query(query, saved_model_name):
    """
    Generate a predicted class ID for a given text query using a pre-trained model.
    This function loads a specified pre-trained model, tokenizes the input query, 
    and computes the logits to determine the predicted class ID by selecting the 
    index of the highest logit.

    Parameters:
    query (str): The text input for which to generate a prediction.
    saved_model_name (str): The name of the pre-trained model to be used.

    Returns:
    int: The predicted class ID corresponding to the input query.
    """
    model = load_model_for_prediction(saved_model_name)
    tokenizer_output = tokenizer(query, truncation=True, return_token_type_ids=True, max_length = 50, return_tensors = 'tf')
    logits = model(**tokenizer_output)["logits"]
    predicted_class_id = int(tf.math.argmax(logits, axis=-1)[0])
    return predicted_class_id
