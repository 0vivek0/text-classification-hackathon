import re
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

from tqdm import tqdm
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from datasets import Dataset, DatasetDict
from transformers import DataCollatorWithPadding

from text_preprocessing import *

os.environ["TOKENIZERS_PARALLELISM"] = "false"
pd.set_option('max_colwidth', 400)

model_checkpoint = "distilroberta-base"

num_labels = 2
batch_size = 32

model = TFAutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels = num_labels)


def model_training_and_saving(dataset, saved_model_name, num_epochs = 3):
    tf_train_dataset, tf_validation_dataset = get_encoded_train_validation_tf_data(dataset)
    num_epochs = num_epochs
    num_train_steps = len(tf_train_dataset) * num_epochs
    lr_scheduler = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=5e-5, end_learning_rate=0.0, decay_steps=num_train_steps, power = 2
    )

    opt = tf.keras.optimizers.Adam(learning_rate=lr_scheduler)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    lr_schedule = []
    for step in range(lr_scheduler.decay_steps):

        decay = (1 - (step / float(lr_scheduler.decay_steps))) ** lr_scheduler.power
        lr_schedule.append(lr_scheduler.initial_learning_rate * decay)

    model.compile(optimizer=opt, loss=loss, metrics=["accuracy"])

    model.fit(tf_train_dataset, validation_data=tf_validation_dataset, epochs=num_epochs)

    model.save_pretrained(f"{saved_model_name}")

    return saved_model_name

