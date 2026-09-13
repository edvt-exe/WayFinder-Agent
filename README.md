# WayFinder-Agent

A standalone AI microservice that turns a set of trip filters into a structured, real-world list of Points of Interest (POIs).

## What it does

You send it a trip's constraints — city, number of days, transport mode, budget, pacing, interest categories, vibe, active hours per day, meals per day, and accessibility needs — and it returns a validated JSON list of real places (museums, restaurants, parks, landmarks) that match those constraints, each with coordinates, an estimated cost, and a suggested time allocation.

It's built to be plugged into a larger route-generation application as the "data retrieval" layer: it doesn't build the itinerary or the route itself, it just supplies accurate, budget-aware POI data for another system to arrange.

## How it works

- **FastAPI** exposes a single `POST /api/v1/agent/search` endpoint that accepts the trip filters.
- The filters are turned into a prompt sent to **Claude** (Anthropic API), guided by a system prompt that enforces real (non-hallucinated) places, realistic coordinates, category/vibe matching, and per-day meal stops.
- Claude's response is forced into a strict JSON schema via **tool calling**, so the output is always structured POI data — never free-form text.
- A budget-enforcement step then double-checks the total cost of all returned POIs against the trip's max budget, trimming the most expensive non-essential stops if needed, before the response is returned.

More detail on setup, configuration, and API usage will be added here as the project develops.