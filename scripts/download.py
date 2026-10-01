"""
This file downloads the following items:
* "distilbert/distilbert-base-cased-distilled-squad"
* 

"""
##### Imports
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from datasets import load_dataset
import os

##### Global vars
MODEL_LOCAL_PATH = os.path.join(
    os.getcwd(),
    'downloads',
    'distilbert'
)
DS_LOCAL_PATH = os.path.join(
    os.getcwd(),
    'downloads',
    'hotpotqa'
)
MODEL_NAME = "distilbert/distilbert-base-cased-distilled-squad"
DATASET_NAME = "hotpotqa/hotpot_qa"

####################################
    # Model Download #
##### Tokenizer and Model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)

##### Storing
tokenizer.save_pretrained(MODEL_LOCAL_PATH)
model.save_pretrained(MODEL_LOCAL_PATH)

##### Usage
# tokenizer = AutoTokenizer.from_pretrained(
#     LOCAL_PATH,
#     local_files_only=True
# )

# model = AutoModelForQuestionAnswering.from_pretrained(
#     LOCAL_PATH,
#     local_files_only=True
# )

#######################################
    # Dataset Download #
##### Download
dataset = load_dataset(
    "hotpotqa/hotpot_qa",
    "distractor"
)

dataset.save_to_disk(DS_LOCAL_PATH)

##### Usage
# dataset = load_from_disk(DS_LOCAL_PATH)
