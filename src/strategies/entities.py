from datetime import date
from pydantic import BaseModel


class StrategyParams(BaseModel):
    name: str
    initial_investment_usd: float
    start_date: date
    end_date: date


class StrategyResults(BaseModel):
    diff_from_baseline_at_end: float
    percent_increase_from_baseline_at_end: float
