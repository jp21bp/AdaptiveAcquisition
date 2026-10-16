"""
Executes the pilot_v2 test
"""
##### Imports
#### General
import os
#### Frozen Qs
from ..src.data.questions_loader import load_questions
#### LLM
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering
)
from ..src.models.llm_invocation import LLM
#### Experiment Runner
from ..src.experiment.pilot_v2_exp import ExperimentRunner

##### Load 10 frozen questions
all_questions = load_questions()    #list[Question]

##### Load LLM
#### Model path
MODEL_LOCAL_PATH = os.path.join(
    os.getcwd(),
    'downloads',
    'distilbert'
)
#### Usage
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_LOCAL_PATH,
    local_files_only=True
)
model = AutoModelForQuestionAnswering.from_pretrained(
    MODEL_LOCAL_PATH,
    local_files_only=True
)
#### Initialize
llm = LLM(model, tokenizer)

##### Test pipeline
def test_pipeline():
    ## Experiment runner
    runner = ExperimentRunner(None, "none", 1)

    print(runner.run_state(all_questions[0]))

if __name__ == "__main__":
    test_pipeline()
    print("Success")