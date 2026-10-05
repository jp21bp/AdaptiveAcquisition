"""
This file simulates a fake LLM call.

Used to test structural functionality.
"""
import time
from ..data.schemas import AnswerResult

class FakeLLM:
    def generate(self, prompt: str) -> AnswerResult:
        ### Simulating a timer
        start = time.perf_counter()
        time.sleep(0.001)
        ### Returning answer
        fake_answer = 'FAKE LLM TEST'
        return AnswerResult(
            answer=fake_answer,
            self_reported_confidence=50.5,
            input_tokens=len(prompt.split()),
            output_tokens=len(fake_answer.split()),
            latency_ms=(time.perf_counter - start)*1000,
            raw_output=f'ANSWER:{fake_answer}\nCONFIDENCE: 50.5'
        )


