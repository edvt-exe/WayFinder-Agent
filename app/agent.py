"""
Core agent logic: turns a TravelSearchRequest into a validated list of PointOfInterest objects by calling Claude with a forced tool call.

Why forced tool calling instead of "ask for JSON in the prompt": Claude will always emit a single, schema-valid tool_use block when tool_choice pins it to one tool. That removes the need for regex/JSON repair on the response — Pydantic just validates the tool input directly.
"""

import json
import logging

from anthropic import AsyncAnthropic

from app.config import settings
from app.schemas import PointOfInterest, TravelSearchRequest

logger = logging.getLogger("travel_agent")

client = AsyncAnthropic(api_key=settings.anthropic_api_key)


# System prompt
# This is the agent's entire personality and rulebook. Keep it here (not in main.py) so it can be versioned, unit tested, and swapped independently of the endpoint that calls it.

SYSTEM_PROMPT = """\
You are a Travel Data Retrieval Agent. You research and return real, \
existing Points of Interest (POIs) for a specific city, strictly matching \
the traveler's filters. You are not a creative writer — you are a data \
retrieval system with a JSON output contract.

CORE RULES — NEVER BREAK THESE:

1. NO HALLUCINATIONS.
   Only return places that actually exist. Use your knowledge of real \
   museums, restaurants, parks, landmarks, and venues in the requested city. \
   If you are not confident a place exists, do not include it. Never invent \
   a plausible-sounding name.

2. COORDINATES.
   Every POI needs a latitude and longitude as close to the real location \
   as you can recall. Precision to ~4 decimal places is expected for major \
   or well-known venues. Never return (0, 0) or a placeholder coordinate.

3. BUDGET DISCIPLINE.
   The traveler gave a max_budget for the ENTIRE trip. The sum of \
   estimated_cost across every POI you return must not exceed it. Public \
   parks, viewpoints, walking streets, and other free attractions must be \
   priced at exactly 0.0. Price everything else at a realistic local rate \
   (entry ticket, average meal cost, activity fee). Bias toward leaving \
   headroom under the budget rather than hitting it exactly.

4. PACING AND VOLUME.
   Use `days`, `hours_per_day`, and `pacing` to decide how many POIs to \
   return in total:
   - Relaxed:  roughly 2 stops per active day, longer dwell times.
   - Balanced: roughly 3-4 stops per active day.
   - Intense:  roughly 5-6 stops per active day, tighter schedule_labels.
   Each POI's schedule_label should reflect a realistic time allocation \
   that could fit inside hours_per_day when combined with the others.

5. MEALS.
   Include exactly `meals_per_day * days` POIs categorized as restaurant \
   or cafe. Their cuisine, price point, and setting must match both the \
   `categories` and `vibe` filters (e.g. a 'Romantic' vibe should not \
   surface a fast-food court; a 'Local/Authentic' vibe should avoid tourist \
   chains). Use schedule_label values like "Perfect for Lunch" or \
   "Dinner Recommendation" for these entries.

6. CATEGORY AND VIBE MATCHING.
   Every non-meal POI must plausibly belong to at least one of the \
   requested `categories`. The overall selection should read as a coherent \
   trip matching the requested `vibe`, not a generic top-10 list.

7. TRANSPORT AWARENESS.
   If transport is 'walking', keep POIs geographically clustered and close \
   together. If 'public transport' or 'by car', a wider radius across the \
   city is acceptable, but avoid anything effectively unreachable within \
   the trip's day count.

8. ACCESSIBILITY.
   If accessibility_required is true, only return POIs you are reasonably \
   confident have step-free access, ramps, or flat terrain, and set \
   is_accessible to true for all of them. If it is false, set \
   is_accessible to true or false based on what you actually know about \
   each place — do not default it to true.

9. TOURIST LEVEL (1-5).
   1 means favor famous, iconic, must-see spots. 5 means favor hidden, \
   local-favorite places tourists rarely visit. Interpolate for values in \
   between — 3 is a balanced mix of both.

10. FREE-ONLY MODE.
   If free_only is true, every non-meal POI must have estimated_cost of \
   exactly 0.0 (parks, free viewpoints, free museums/exhibits, walking \
   tours). Meals are exempt from this — they may still have a realistic \
   cost unless the traveler's budget genuinely cannot support any.

11. BUDGET ALLOCATION.
   budget_allocation is 0-100, where 0 means spend the entire max_budget \
   on food and 100 means spend it entirely on attractions/activities. Use \
   it as a soft guide for how you split estimated_cost between meal POIs \
   and non-meal POIs — do not treat it as a hard constraint if it would \
   make the trip incoherent.

12. DINING CONSTRAINTS ARE NON-NEGOTIABLE.
   Every meal POI (restaurant/cafe) must strictly satisfy dietary_restrictions \
   (e.g. a place tagged Vegan must genuinely offer vegan options — do not \
   include a steakhouse). Respect dining_style (e.g. 'Fine Dining' vs \
   'Street/Fast Food') and cuisine when specified. If no real place in the \
   city satisfies all constraints, prefer a venue that satisfies the \
   dietary_restrictions over one that satisfies cuisine — dietary needs \
   take priority.

13. WEATHER PREFERENCE.
   If weather_preference is 'Mostly Indoor', bias toward museums, galleries, \
   indoor markets, and covered venues. If 'Mostly Outdoor', bias toward \
   parks, outdoor landmarks, and open-air venues. Meals are exempt.

14. GROUP TYPE AND COMPANIONS.
   - Solo / Couple / Friends: no special filtering beyond the other rules; \
     for Couple with a 'Romantic' vibe, favor more intimate settings.
   - Family: exclude nightlife, bars, and adult-oriented venues entirely. \
     If child_age is 'Toddler 0-3' or 'Kids 4-11', prioritize venues \
     genuinely suitable for that age (interactive/hands-on museums, \
     playgrounds, short queues) and avoid anything requiring long stillness \
     or unsuitable content. 'Teens 12+' can include most general POIs.

15. PET-FRIENDLY MODE.
   If pet_friendly is true, only include venues you're reasonably confident \
   allow pets (parks, outdoor cafes, pet-friendly patios). Exclude venues \
   that typically prohibit them (indoor museums, fine dining, most indoor \
   attractions) unless you know a specific exception.

16. OUTPUT DISCIPLINE.
   You must respond ONLY by calling the `return_pois` tool with a complete, \
   schema-valid list of POIs. Do not include prose, explanations, \
   disclaimers, or markdown outside of the tool call.
"""


def _build_user_prompt(payload: TravelSearchRequest) -> str:
    """Serializes the filter payload into a compact, unambiguous instruction block."""
    return (
        f"Generate a POI list for the following trip:\n\n"
        f"- City: {payload.city}\n"
        f"- Starting point: {payload.start_point or 'not specified, use city center'}\n"
        f"- Trip length: {payload.days} day(s)\n"
        f"- Transport mode: {payload.transport.value}\n"
        f"- Max total budget: {payload.max_budget} (local currency)\n"
        f"- Pacing: {payload.pacing.value}\n"
        f"- Interest categories: {', '.join(payload.categories)}\n"
        f"- Vibe: {payload.vibe}\n"
        f"- Active hours per day: {payload.hours_per_day}\n"
        f"- Required meal stops per day: {payload.meals_per_day}\n"
        f"- Accessibility required: {payload.accessibility_required}\n"
        f"- Tourist level (1=iconic, 5=hidden gems): {payload.tourist_level}\n"
        f"- Free attractions only: {payload.free_only}\n"
        f"- Budget allocation (0=food, 100=attractions): {payload.budget_allocation}\n"
        f"- Dining style: {', '.join(payload.dining_style) or 'no preference'}\n"
        f"- Cuisine preference: {payload.cuisine or 'no preference'}\n"
        f"- Dietary restrictions: {', '.join(payload.dietary_restrictions) or 'none'}\n"
        f"- Weather preference: {payload.weather_preference.value}\n"
        f"- Group type: {payload.group_type.value}\n"
        f"- Child age (if family): {payload.child_age.value if payload.child_age else 'n/a'}\n"
        f"- Pet-friendly required: {payload.pet_friendly}\n\n"
        f"Call return_pois now with the full list."
    )


# Tool definition — the JSON schema Claude is forced to fill in.
# Built by hand (rather than PointOfInterest.model_json_schema()) so the field-level `description`s can carry the same behavioral hints as the system prompt; the LLM tends to respect descriptions on the tool schema even more literally than prose earlier in context.

RETURN_POIS_TOOL = {
    "name": "return_pois",
    "description": "Return the final, validated list of Points of Interest for this trip.",
    "input_schema": {
        "type": "object",
        "properties": {
            "pois": {
                "type": "array",
                "description": "Every POI for the whole trip, across all days.",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Real name of the place."},
                        "category": {
                            "type": "string",
                            "description": "e.g. museum, restaurant, park, landmark, cafe, parking.",
                        },
                        "latitude": {"type": "number"},
                        "longitude": {"type": "number"},
                        "estimated_cost": {
                            "type": "number",
                            "description": "Local currency. 0.0 for free places.",
                        },
                        "schedule_label": {
                            "type": "string",
                            "description": "e.g. 'Recommended: 2 hours', 'Perfect for Lunch'.",
                        },
                        "is_accessible": {"type": "boolean"},
                    },
                    "required": [
                        "name",
                        "category",
                        "latitude",
                        "longitude",
                        "estimated_cost",
                        "schedule_label",
                        "is_accessible",
                    ],
                },
            }
        },
        "required": ["pois"],
    },
}


async def generate_pois(payload: TravelSearchRequest) -> list[PointOfInterest]:
    """
    Calls Claude with the system prompt + filters, forces the return_pois
    tool call, and returns a validated list of PointOfInterest objects.

    Raises:
        ValueError: if Claude doesn't return a usable tool_use block, or the
            tool input fails Pydantic validation (caught upstream in main.py
            and turned into a 502).
    """
    response = await client.messages.create(
        model=settings.anthropic_model,
        max_tokens=settings.max_tokens,
        system=SYSTEM_PROMPT,
        tools=[RETURN_POIS_TOOL],
        # Pinning tool_choice to this exact tool is what guarantees a structured response instead of a free-text one.
        tool_choice={"type": "tool", "name": "return_pois"},
        messages=[{"role": "user", "content": _build_user_prompt(payload)}],
    )

    tool_use_block = next(
        (block for block in response.content if block.type == "tool_use"),
        None,
    )
    if tool_use_block is None:
        logger.error("Claude response had no tool_use block: %s", response.content)
        raise ValueError("Agent did not return a structured tool call.")

    raw_pois = tool_use_block.input.get("pois", [])
    logger.info("Claude returned %d raw POIs for %s", len(raw_pois), payload.city)

    # Validate every entry individually so one malformed POI doesn't discard an otherwise-good response.
    validated: list[PointOfInterest] = []
    for entry in raw_pois:
        try:
            validated.append(PointOfInterest(**entry))
        except Exception as exc:
            logger.warning("Dropping malformed POI %s: %s", entry.get("name", "?"), exc)

    if not validated:
        raise ValueError("Agent returned zero valid POIs after schema validation.")

    return enforce_budget(validated, payload.max_budget)


def enforce_budget(pois: list[PointOfInterest], max_budget: float) -> list[PointOfInterest]:
    """
    Hard safety net behind the system prompt's budget instructions.
    LLMs are good but not arithmetically perfect — this guarantees the
    contract holds even if Claude's own sum is slightly off.

    Strategy: if the total exceeds max_budget, drop the most expensive
    non-meal POIs first (meals are treated as required), most expensive
    first, until back under budget.
    """
    total = sum(poi.estimated_cost for poi in pois)
    if total <= max_budget:
        return pois

    meal_categories = {"restaurant", "cafe"}
    droppable = sorted(
        (p for p in pois if p.category.lower() not in meal_categories),
        key=lambda p: p.estimated_cost,
        reverse=True,
    )
    kept = list(pois)
    for poi in droppable:
        if total <= max_budget:
            break
        kept.remove(poi)
        total -= poi.estimated_cost

    if total > max_budget:
        logger.warning(
            "Trip still over budget after trimming non-meal POIs (%.2f > %.2f); "
            "meal stops alone exceed max_budget.",
            total,
            max_budget,
        )

    return kept