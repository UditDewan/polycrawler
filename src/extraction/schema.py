"""The structured signal the hosted LLM must return for a piece of text.
Pydantic validates every response; bounds here are what trigger corrective retries."""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class SignalType(str, Enum):
    injury_news = "injury_news"   # credible injury / availability report
    rumor = "rumor"               # unconfirmed claim
    lineup = "lineup"             # team selection / starting XI
    transfer = "transfer"
    suspension = "suspension"     # ban / suspension affecting availability
    banter = "banter"             # opinion / noise, no predictive value
    other = "other"


class Sentiment(str, Enum):
    positive = "positive"   # good for the team's match chances
    negative = "negative"   # bad for the team's chances
    neutral = "neutral"


class ExtractedSignal(BaseModel):
    is_relevant: bool                       # carries any match-prediction signal at all?
    signal_type: SignalType
    team: str | None = None                 # team the signal concerns, if identifiable
    player: str | None = None
    sentiment: Sentiment = Sentiment.neutral
    confidence: float = Field(ge=0.0, le=1.0)  # model's certainty in this extraction
    rationale: str | None = None            # one short clause of why
