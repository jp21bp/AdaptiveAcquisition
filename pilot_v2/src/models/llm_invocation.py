"""
Invokes a given LLM

MODEL_NAME = "distilbert/distilbert-base-cased-distilled-squad"

This model does not generate answers
    It only returns the context chars that best answer a question
"""
import torch, os, time
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering
)
from ..data.schemas import AnswerResult

# ##### Model path
# MODEL_LOCAL_PATH = os.path.join(
#     os.getcwd(),
#     'downloads',
#     'distilbert'
# )

# ##### Usage
# tokenizer = AutoTokenizer.from_pretrained(
#     MODEL_LOCAL_PATH,
#     local_files_only=True
# )

# model = AutoModelForQuestionAnswering.from_pretrained(
#     MODEL_LOCAL_PATH,
#     local_files_only=True
# )

##### LLM-specific
class LLM:
    def generate(self, question:str, context: str) -> AnswerResult:
        # Assuming large prompt => separate chunks 
                # => track sore internally to find best candidate
        ## Start timer
        start_time = time.perf_counter()
        ## Tokenize inputs
            # Configured for long contexts
        all_inputs = tokenizer(
            question,
            context, 
            mex_length=384,
                # Max chunk length
                # Chosen since max window length is 512
                    # First chunk must include the question
            truncation="only_second",
                # When to start truncate
            stride=128,
                # Overlap
            return_overflowing_tokens=True,
                # Ensures tokenization of all chunks
            padding="max_length",
                # Guarantees all chunks are same length size
            return_tensors="pt", 
                # Returns PT tensors
            return_offsets_mapping=True,
                # Returns offsets of each token
        )
        ### Internal scoring
        all_candidates = []
        for i in range(all_inputs["input_ids"].shape[0]):
            ## Selected chunks
            inputs = {
                key: all_inputs[key][i:i+1]
                for key in ("input_ids", "attention_mask")
            }
            ## Chunk details
            offsets = all_inputs["offset_mapping"][i]
            sequence_ids = all_inputs.sequence_ids(i)
            ## Model invocation on chunk
            with torch.no_grad():
                outputs = model(**inputs)
            ## Restrict start/end predictions to context tokens
            valid = torch.tensor(
                [sid == 1 for sid in sequence_ids],
                dtype=torch.bool
            )
            ## Copying logits
            start_logits = outputs.start_logits[0].clone()
            end_logits = outputs.end_logits[0].clone()
            ## Non-context have no vlue
            start_logits[~valid] = float("-inf")
            end_logits[~valid] = float("-inf")
            ## Probabilities upon context logits
            start_probs = torch.softmax(start_logits, dim=-1)
            end_probs = torch.softmax(end_logits, dim=-1)
            ## Find the highest-scoring valid span in this chunk
            best = None
            ## Finding best score
            for start in range(len(sequence_ids)):
                if not valid[start]:
                    continue

                for end in range(start, len(sequence_ids)):
                    if not valid[end]:
                        continue

                    score = (start_probs[start] * end_probs[end]).item()

                    if best is None or score > best["score"]:
                        best = {"start": start, "end": end, "score": score}
            ## Converting score into char tokens
            if best is not None:
                ## Getting index of best context (in this chunk) for llm answer
                char_start = offsets[best["start"]][0].item()
                char_end = offsets[best["end"]][1].item()
                ## Storing best context in current chunk
                if char_end > char_start:
                    all_candidates.append({
                        "answer": context[char_start:char_end],
                        "start": char_start,
                        "end": char_end,
                        "score": best["score"]
                    })

        ## Compare candidates from all chunks
        best_answer = max(all_candidates, key=lambda x: x["score"])
        ## End timer
        end_time = time.perf_counter()
       
        return AnswerResult(
            answer = best_answer["answer"],
            self_reported_confidence = round(best_answer["score"],4),
            input_tokens = (len(question.split()) + len(context.split())),
            context_tokens = len(context.split()),
            output_tokens = len(best_answer["answer"].split()),
            latency_ms = (end_time - start_time)*1000,
            raw_output = best_answer["answer"]
        )
