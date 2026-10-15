"""
Invokes a given LLM

MODEL_NAME = "distilbert/distilbert-base-cased-distilled-squad"
"""
import torch, os, time
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering
)
from ..data.schemas import AnswerResult
from .parser import parse_answer

##### Model path
MODEL_LOCAL_PATH = os.path.join(
    os.getcwd(),
    'downloads',
    'distilbert'
)

##### Usage
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_LOCAL_PATH,
    local_files_only=True
)

model = AutoModelForQuestionAnswering.from_pretrained(
    MODEL_LOCAL_PATH,
    local_files_only=True
)

##### LLM-specific
class LLM:
    def generate(self, prompt:str, context: str) -> AnswerResult:
        ## Start timer
        start_time = time.perf_counter()
        ## Tokenize inputs
        inputs = tokenizer(prompt, context, return_tensors="pt")
        ## Invoke the LLM
        outputs = model(**inputs)
        ## Parse the answer
        start = torch.argmax(outputs.start_logits)
        end = torch.argmax(outputs.end_logits) + 1
        ## Convert tokens to str
        llm_response = tokenizer.convert_tokens_to_string(
            tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][start:end])
        )
        ## End timer
        end_time = time.perf_counter()
        ## Parsing answers
        answer, confidence = parse_answer(llm_response)
        return AnswerResult(
            answer = answer,
            self_reported_confidence = confidence,
            input_tokens = (len(prompt.split()) + len(context.split())),
            context_tokens = len(context.split()),
            output_tokens = len(answer.split()),
            latency_ms = (end_time - start_time)*1000,
            raw_output = llm_response
        )
