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
        inputs = tokenizer(
            prompt, 
            context, 
            return_tensors="pt", 
                # Returns PT tensors
            return_offsets_mapping=True,
                # Returns offsets of each token
            # truncation=True, 
                # Discards the end of the document outside window size
        )
        ## Invoke the LLM
        with torch.no_grad():
            outputs = model(**inputs)
        ## Parse the answer
        start_idx = torch.argmax(outputs.start_logits)
        end_idx = torch.argmax(outputs.end_logits) + 1
        ## Convert tokens to str
        llm_response = tokenizer.convert_tokens_to_string(
            tokenizer.convert_ids_to_tokens(
                inputs["input_ids"][0][start_idx:end_idx]
            )
        )
        ## End timer
        end_time = time.perf_counter()
        ### Calculating confidence score
        ### Distinguishing between question and context
        sequence_ids = inputs.sequence_ids(0)
        context_mask = torch.tensor(
            [sid == 1 for sid in sequence_ids],
            dtype=torch.bool
        )
        ### Applying context mask to logits
        start_logits = outputs.start_logits[0].clone()
        end_logits = outputs.end_logits[0].clone()
        start_logits[~context_mask] = float("-inf")
        end_logits[~context_mask] = float("-inf")
        ### Convert logits to probability distributions
        start_probs = torch.softmax(start_logits, dim=-1)
        end_probs = torch.softmax(end_logits, dim=-1)
        ### Finding highest score
        best_score = -1.0
        # best_start = best_end = None
        for start in range(len(sequence_ids)):
            ## Only function on context
            if not context_mask[start]:
                continue

            for end in range(start, len(sequence_ids)):
                if not context_mask[end]:
                    continue

                score = (start_probs[start] * end_probs[end]).item()

                if score > best_score:
                    print('iterate', start_probs[start], end_probs[end])
                    best_score = score
                    # best_start, best_end = start, end
        
        return AnswerResult(
            answer = llm_response,
            self_reported_confidence = best_score,
            input_tokens = (len(prompt.split()) + len(context.split())),
            context_tokens = len(context.split()),
            output_tokens = len(llm_response.split()),
            latency_ms = (end_time - start_time)*1000,
            raw_output = llm_response
        )
