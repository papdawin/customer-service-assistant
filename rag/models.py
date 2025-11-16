from typing import Optional

from pydantic import BaseModel


class Query(BaseModel):
    question: Optional[str] = None
    text: Optional[str] = None

    def prompt(self) -> str:
        """Return the first non-empty field, raising if both inputs are blank."""
        value = (self.question or self.text or "").strip()
        if not value:
            raise ValueError("Empty question/text payload")
        return value


__all__ = ["Query"]
