"""
Pydantic models for the Travel Data Retrieval Agent.

Two things live here:
1. TravelSearchRequest — mirrors the frontend form payload exactly.
2. PointOfInterest / AgentSearchResponse — the structured output we force
   the Anthropic model to return via tool calling.
"""

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


# Enums — kept strict so a typo in the frontend fails fast instead of silently confusing the LLM.

class TransportMode(str, Enum):
    WALKING = "walking"
    PUBLIC_TRANSPORT = "public transport"
    CAR = "by car"


class PacingLevel(str, Enum):
    RELAXED = "Relaxed"
    BALANCED = "Balanced"
    INTENSE = "Intense"


class WeatherPreference(str, Enum):
    OUTDOOR = "Mostly Outdoor"
    INDOOR = "Mostly Indoor"


class GroupType(str, Enum):
    SOLO = "Solo"
    COUPLE = "Couple"
    FRIENDS = "Friends"
    FAMILY = "Family"


class ChildAge(str, Enum):
    TODDLER = "Toddler 0-3"
    KIDS = "Kids 4-11"
    TEENS = "Teens 12+"


# Request payload — matches the frontend form 1:1

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

    # New preference axes, mapped from the richer frontend filter set
    tourist_level: int = Field(3, ge=1, le=5, description="1 = famous/iconic spots, 5 = hidden local gems.")
    free_only: bool = Field(False, description="If true, only free (cost 0.0) attractions are returned. Meals may still have a cost.")
    budget_allocation: int = Field(
        50, ge=0, le=100,
        description="0 = spend the budget entirely on food, 100 = entirely on attractions/activities.",
    )
    dining_style: List[str] = Field(default_factory=list, description="e.g. ['Casual Dining', 'Specialty Cafes'].")
    cuisine: Optional[str] = Field(None, description="Specific cuisine preference, e.g. 'Italian'.")
    dietary_restrictions: List[str] = Field(default_factory=list, description="e.g. ['Vegan', 'Gluten-Free'].")
    weather_preference: WeatherPreference = Field(WeatherPreference.OUTDOOR)
    group_type: GroupType = Field(GroupType.SOLO)
    child_age: Optional[ChildAge] = Field(None, description="Set when group_type is Family.")
    pet_friendly: bool = Field(False, description="If true, only pet-friendly venues are included.")

    # Allows the model to be built either from the camelCase frontend payload (via alias) or from snake_case internal calls/tests.
    model_config = {"populate_by_name": True}

    @field_validator("categories")
    @classmethod
    def categories_not_empty_strings(cls, value: List[str]) -> List[str]:
        cleaned = [c.strip() for c in value if c.strip()]
        if not cleaned:
            raise ValueError("categories must contain at least one non-empty value")
        return cleaned


# Response schema — this exact shape is what we force via tool_choice.

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