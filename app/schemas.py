"""
Pydantic models for the Travel Data Retrieval Agent.

Two things live here:
1. TravelSearchRequest — mirrors the frontend form payload exactly.
2. PointOfInterest / AgentSearchResponse — the structured output we force
   the Anthropic model to return via tool calling.
"""

from pydantic import BaseModel, Field, field_validator
from enum import Enum
from typing import List, Optional


class TransportMode(str, Enum):
    WALKING = "walking"
    PUBLIC_TRANSPORT = "public transport"
    CAR = "by car"


class PacingLevel(str, Enum):
    RELAXED = "Relaxed"
    BALANCED = "Balanced"
    INTENSE = "Intense"


class TravelSearchRequest(BaseModel):
    city: str = Field(..., description="Exact city or area to search in, e.g. 'Bucharest'.")
    start_point: Optional[str] = Field(
        None,
        alias="startPoint",
        description="Optional starting address/landmark the route should begin near.",
    )
    days: int = Field(..., ge=1, le=30, description="Number of days the trip lasts.")
    transport: TransportMode
    max_budget: float = Field(..., gt=0, description="Total budget cap for the whole trip, local currency.")
    pacing: PacingLevel
    categories: List[str] = Field(..., min_length=1, description="Interest tags, e.g. ['History & Culture'].")
    vibe: str = Field(..., description="Requested atmosphere, e.g. 'Romantic', 'Local/Authentic'.")
    hours_per_day: int = Field(..., ge=1, le=16, description="Active hours available per day.")
    meals_per_day: int = Field(..., ge=0, le=6, description="Number of restaurant/cafe stops per day.")
    accessibility_required: bool = Field(False, description="If true, only wheelchair-accessible POIs are returned.")

    model_config = {"populate_by_name": True}

    @field_validator("categories")
    @classmethod
    def categories_not_empty_strings(cls, value: List[str]) -> List[str]:
        cleaned = [c.strip() for c in value if c.strip()]
        if not cleaned:
            raise ValueError("categories must contain at least one non-empty string")
        return cleaned


class PointOfInterest(BaseModel):
    name: str = Field(..., description="Real, existing name of the place.")
    category: str = Field(..., description="e.g. 'museum', 'restaurant', 'park', 'parking'.")
    latitude: float
    longitude: float
    estimated_cost: float = Field(..., ge=0, description="Cost in local currency. 0.0 for free places.")
    schedule_label: str = Field(..., description="e.g. 'Recommended: 2 hours', 'Perfect for Lunch'.")
    is_accessible: bool


class AgentSearchResponse(BaseModel):
    """What POST /api/v1/agent/search actually returns to the caller."""

    city: str
    total_estimated_cost: float
    within_budget: bool
    pois: List[PointOfInterest]