"""
Standalone FastAPI microservice exposing the Travel Data Retrieval Agent

Run locally with: uvicorn app.main:app --reload --port 8001
"""

import logging

from anthropic import APIError, APIStatusError
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.agent import generate_pois
from app.schemas import TravelSearchRequest, AgentSearchResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("travel_agent")

app = FastAPI(
    title="Travel Data Retrieval Agent",
    description="Standalone microservice that turns trip filters into a structured, real-world POI list.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict:
    """Basic liveness probe for load balancers / orchestrators."""
    return {"status": "ok"}


@app.post("/api/v1/agent/search", response_model=AgentSearchResponse)
async def search_pois(payload: TravelSearchRequest) -> AgentSearchResponse:
    """
    Accepts the frontend's trip-filter payload, asks Claude for a structured
    POI list matching every filter, validates the budget, and returns it.
    """
    try:
        pois = await generate_pois(payload)
    except APIStatusError as exc:
        logger.error("Anthropic API error: %s", exc)
        raise HTTPException(status_code=502, detail="Upstream AI provider error.") from exc
    except APIError as exc:
        logger.error("Anthropic SDK error: %s", exc)
        raise HTTPException(status_code=502, detail="Failed to reach AI provider.") from exc
    except ValueError as exc:
        logger.error("Agent produced an invalid response: %s", exc)
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    total_cost = sum(p.estimated_cost for p in pois)
    return AgentSearchResponse(
        city=payload.city,
        total_estimated_cost=round(total_cost, 2),
        within_budget=total_cost <= payload.max_budget,
        pois=pois,
    )