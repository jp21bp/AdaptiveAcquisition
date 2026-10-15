"""
Acquisition action data class
"""

from dataclasses import dataclass

@dataclass
class RetrievalAction:

    method: str
    k: int

    @property
    def action_type(self) -> str:
        return f"{self.method}_top_{self.k}"