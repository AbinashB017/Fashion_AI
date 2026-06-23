"""
src/llm/gemini_client.py
-------------------------
Gemini integration for the Dare XAI Fashion AI system.

Gemini is used ONLY for:
  1. Intent extraction: NL query → structured JSON intent
  2. Explanation generation: outfit + RAG context → natural language reasoning

Gemini does NOT generate outfit recommendations directly.
All recommendations come from the retrieval + ranking pipeline.
This keeps the system deterministic, explainable, and reproducible.
"""

import os
import sys
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Any

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))


# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════

def _get_api_keys() -> Dict[str, str]:
    """Load API keys from .env file or environment."""
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    return {
        "gemini_1": os.getenv("GEMINI_API_KEY_1", ""),
        "gemini_2": os.getenv("GEMINI_API_KEY_2", ""),
        "groq_1": os.getenv("GROQ_API_KEY_1", ""),
        "groq_2": os.getenv("GROQ_API_KEY_2", ""),
    }


def _load_model() -> str:
    """
    Load the Gemini model name from .env → GEMINI_MODEL.
    Falls back to 'gemini-2.5-flash' only if the env var is completely absent.
    This ensures the model name is NEVER hardcoded in Python logic —
    it is always driven by configuration.
    """
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    return os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


# ══════════════════════════════════════════════════════════════════════════════
# Intent extraction
# ══════════════════════════════════════════════════════════════════════════════

INTENT_SCHEMA = {
    "occasion": "string — one of: office, party, casual, wedding, festive, sports, vacation, winter, or null",
    "gender": "string — one of: men, women, or null if not specified",
    "wear_type": "string — one of: western, ethnic, or null",
    "style": "string — e.g. formal, smart-casual, sporty, bohemian, or null",
    "season": "string — one of: summer, winter, monsoon, or null if not inferable",
    "budget": "string — one of: affordable (<₹2000), mid-range (₹2000-5000), premium (>₹5000), or null",
    "color_preference": "string — specific colour preference mentioned, or null",
    "item_type": "string — specific item type if user mentions one (e.g. 'dress', 'suit'), or null",
    "query_summary": "string — 1-sentence summary of what the user wants",
}

INTENT_PROMPT_TEMPLATE = """You are a fashion intent parser. Extract structured intent from the user's fashion query.

User Query: "{query}"

Extract a JSON object with these exact fields:
{schema}

Rules:
- Be conservative: only fill fields that are clearly indicated or strongly implied
- occasion: map to the closest standard value
- gender: if not mentioned, leave null (never assume)
- Return ONLY valid JSON, no markdown, no explanation
- All values must be strings or null (never arrays or booleans)

JSON:"""


import requests

def _call_groq(api_key: str, prompt: str, is_json: bool = False) -> str:
    """Call Groq REST API."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
    }
    if is_json:
        payload["response_format"] = {"type": "json_object"}
        
    response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload, timeout=20)
    
    if response.status_code != 200:
        raise RuntimeError(f"Groq API error {response.status_code}: {response.text}")
        
    return response.json()["choices"][0]["message"]["content"]


def _call_llm_cascade(prompt: str, model_name: str, is_json: bool = False) -> str:
    """
    Cascade through API keys if rate-limited or exhausted:
    1. Gemini Key 1
    2. Gemini Key 2
    3. Groq Key 1
    4. Groq Key 2
    """
    keys = _get_api_keys()
    
    # Sequence of tuples: (provider_name, api_key)
    cascade_sequence = [
        ("gemini", keys["gemini_1"]),
        ("gemini", keys["gemini_2"]),
        ("groq", keys["groq_1"]),
        ("groq", keys["groq_2"])
    ]
    
    for provider, key in cascade_sequence:
        if not key:
            continue
            
        try:
            if provider == "gemini":
                from google import genai
                client = genai.Client(api_key=key)
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                )
                return response.text.strip()
                
            elif provider == "groq":
                return _call_groq(key, prompt, is_json).strip()
                
        except Exception as e:
            err_str = str(e).lower()
            print(f"[LLM] {provider.upper()} API call failed with key ending in ...{key[-4:] if len(key)>4 else ''}. Error: {e}")
            if "resource_exhausted" in err_str or "429" in err_str or "rate_limit" in err_str or "quota" in err_str:
                print(f"[LLM] Exhausted/Rate Limited. Cascading to next provider/key...")
                continue
            else:
                # If it's a different error, we still try the next one to be safe
                print(f"[LLM] Error is not explicitly a rate limit, but cascading anyway.")
                continue
                
    raise RuntimeError("All LLM providers and keys exhausted or failed.")


def extract_intent(query: str, model_name: Optional[str] = None) -> Dict:
    """
    model_name: if None (default), reads GEMINI_MODEL from .env.
                Pass explicitly only in tests or when overriding.
    """
    model_name = model_name or _load_model()
    """
    Extract structured intent from a natural language fashion query.

    Returns a dict with fields from INTENT_SCHEMA.
    Falls back to heuristic extraction if Gemini fails.
    """
    try:
        schema_str = json.dumps(INTENT_SCHEMA, indent=2)
        prompt = INTENT_PROMPT_TEMPLATE.format(
            query=query,
            schema=schema_str,
        )

        raw = _call_llm_cascade(prompt, model_name, is_json=True)

        # Clean up any markdown fences
        raw = re.sub(r"```json\s*", "", raw)
        raw = re.sub(r"```\s*", "", raw)
        raw = raw.strip()

        intent = json.loads(raw)
        intent["_source"] = "gemini"
        intent["_original_query"] = query
        return intent

    except Exception as e:
        print(f"[Gemini] Intent extraction failed: {e} — using heuristic fallback")
        return _heuristic_intent(query)


def _heuristic_intent(query: str) -> Dict:
    """
    Rule-based fallback intent extraction when Gemini is unavailable.
    Covers the most common fashion request patterns.
    """
    q = query.lower()

    intent = {
        "occasion": None, "gender": None, "wear_type": None,
        "style": None, "season": None, "budget": None,
        "color_preference": None, "item_type": None,
        "query_summary": query,
        "_source": "heuristic",
        "_original_query": query,
    }

    # Occasion mapping
    # IMPORTANT: sports keywords are checked BEFORE office to prevent
    # "workout" matching "work" (office) and "gym" matching nothing.
    # Each keyword uses whole-phrase matching where ambiguity exists.
    import re as _re
    _word = lambda w: bool(_re.search(r'\b' + _re.escape(w) + r'\b', q))

    occasion_keywords = {
        "sports":   ["gym", "workout", "sport", "exercise", "athletic", "yoga",
                     "running", "jogging", "training"],
        "wedding":  ["wedding", "marriage", "shaadi", "reception", "baraat", "groom", "bride"],
        "festive":  ["festival", "festive", "puja", "diwali", "eid", "navratri", "durga",
                     "holi", "rakhi"],
        "office":   ["office", "meeting", "business", "corporate", "professional", "interview"],
        "party":    ["party", "night out", "club", "evening", "cocktail", "dinner", "gala"],
        "casual":   ["casual", "everyday", "daily", "weekend", "relaxed", "chill", "hangout",
                     "street"],
        "vacation": ["vacation", "beach", "holiday", "travel", "trip"],
        "winter":   ["winter", "cold", "snow", "cozy", "layering"],
    }
    # "formal" and "work" need word-boundary check to avoid false matches
    if _word("formal"):
        intent["occasion"] = "office"
    elif _word("work") and not _word("workout") and not _word("working out"):
        intent["occasion"] = "office"
    else:
        for occ, kws in occasion_keywords.items():
            if any(_word(kw) for kw in kws):
                intent["occasion"] = occ
                break

    # Gender — check women/female first to avoid "men" matching inside "women"
    if any(w in q for w in ["women", "female", "woman", "girl", "she", "her", "ladies"]):
        intent["gender"] = "women"
    elif any(w in q for w in [" men ", " man ", "male", "guy", " he ", "his", "boys",
                               "groom", "menstyle"]):
        intent["gender"] = "men"
    # Edge: query starts or ends with 'men'/'man'
    elif q.startswith("men ") or q.endswith(" men") or q == "men":
        intent["gender"] = "men"
    elif q.startswith("man ") or q.endswith(" man") or q == "man":
        intent["gender"] = "men"

    # Wear type
    if any(w in q for w in ["ethnic", "kurta", "saree", "sherwani", "salwar", "traditional", "indian"]):
        intent["wear_type"] = "ethnic"
    elif any(w in q for w in ["western", "jeans", "formal", "suit", "dress", "casual"]):
        intent["wear_type"] = "western"

    # Season
    if any(w in q for w in ["summer", "hot", "beach"]):
        intent["season"] = "summer"
    elif any(w in q for w in ["winter", "cold", "snow"]):
        intent["season"] = "winter"

    # Style
    if any(w in q for w in ["formal", "professional"]):
        intent["style"] = "formal"
    elif any(w in q for w in ["smart casual", "smart-casual"]):
        intent["style"] = "smart-casual"
    elif any(w in q for w in ["sporty", "athletic"]):
        intent["style"] = "sporty"

    # Colour
    colors = ["red", "blue", "black", "white", "navy", "green", "pink",
              "brown", "grey", "beige", "cream", "gold", "purple"]
    for c in colors:
        if c in q:
            intent["color_preference"] = c
            break

    # Item type
    items = ["dress", "suit", "shirt", "trousers", "jeans", "kurta",
             "saree", "sherwani", "blazer", "jacket", "top", "skirt"]
    for item in items:
        if item in q:
            intent["item_type"] = item
            break

    return intent


# ══════════════════════════════════════════════════════════════════════════════
# Explanation generation
# ══════════════════════════════════════════════════════════════════════════════

EXPLANATION_PROMPT_TEMPLATE = """You are a world-class personal fashion stylist writing for a premium AI fashion assistant.

User Request: "{user_query}"

Recommended Outfit {outfit_number} — "{outfit_theme}":
{outfit_description}

Compatibility Score: {score}%

{rag_context}

Write a compelling 3-4 sentence outfit explanation that:
1. Explains WHY these specific items work together (colour theory, occasion fit, silhouette balance)
2. References the stylist knowledge above where relevant
3. Highlights 1-2 standout style decisions
4. Ends with a confident styling tip

Tone: Expert but approachable. Not sales-y. Think Vogue meets personal stylist.
Length: 3-4 sentences maximum.
Do NOT mention compatibility scores, indices, or technical details.
Write directly about the clothes and the person wearing them.

Explanation:"""

OUTFIT_DESCRIPTION_TEMPLATE = """Items:
{items}

Occasion: {occasion} | Style: {wear_type} | Gender: {gender}
Price Range: ₹{min_price} – ₹{max_price}
Colour Palette: {palette}"""


def generate_explanation(
    user_query: str,
    outfit: Dict,
    rag_context: str = "",
    outfit_number: int = 1,
    outfit_theme: str = "Complete Look",
    compatibility_score: int = 0,
    model_name: Optional[str] = None,
) -> str:
    """
    Generate a natural language explanation for an outfit recommendation.

    Args:
        user_query          : Original user query
        outfit              : Dict with 'items' (list of product dicts) and 'score'
        rag_context         : Formatted stylist rationale context from FashionRAG
        outfit_number       : 1, 2, or 3 (for the three outfit options)
        outfit_theme        : e.g. "Classic Office", "Modern Party"
        compatibility_score : int 0-100
        model_name          : Gemini model. If None, reads GEMINI_MODEL from .env.

    Returns:
        Natural language explanation string.
    """
    model_name = model_name or _load_model()
    # Build outfit description
    items = outfit.get("items", [])
    item_lines = []
    prices = []
    for item in items:
        line = f"  - {item.get('name', 'Item')} ({item.get('category_label', '')}) by {item.get('brand', '')}"
        item_lines.append(line)
        p = item.get("price_inr", 0)
        if p:
            prices.append(float(p))

    outfit_desc = OUTFIT_DESCRIPTION_TEMPLATE.format(
        items="\n".join(item_lines),
        occasion=outfit.get("occasion", "any"),
        wear_type=outfit.get("wear_type", "western"),
        gender=outfit.get("gender", ""),
        min_price=int(min(prices)) if prices else 0,
        max_price=int(max(prices)) if prices else 0,
        palette=outfit.get("palette", "mixed"),
    )

    prompt = EXPLANATION_PROMPT_TEMPLATE.format(
        user_query=user_query,
        outfit_number=outfit_number,
        outfit_theme=outfit_theme,
        outfit_description=outfit_desc,
        score=compatibility_score,
        rag_context=rag_context if rag_context else "(No additional stylist context available)",
    )

    try:
        explanation = _call_llm_cascade(prompt, model_name, is_json=False)
        return explanation

    except Exception as e:
        print(f"[Gemini] Explanation generation failed: {e}")
        return _fallback_explanation(outfit, outfit_theme)


def _fallback_explanation(outfit: Dict, theme: str) -> str:
    """Simple fallback explanation when Gemini is unavailable."""
    items = outfit.get("items", [])
    names = [i.get("name", "") for i in items[:3]]
    return (
        f"This {theme} look pairs {', '.join(names[:2])} for a cohesive, "
        f"well-balanced ensemble. "
        f"The combination is crafted to complement the occasion while "
        f"maintaining a polished, put-together appearance."
    )


# ══════════════════════════════════════════════════════════════════════════════
# Conversation context
# ══════════════════════════════════════════════════════════════════════════════

def build_chat_context(user_preferences: Dict, conversation_history: List[Dict]) -> str:
    """
    Build a context string from user session preferences for Gemini.
    Used to personalise intent extraction over a conversation.
    """
    parts = []
    if user_preferences.get("gender"):
        parts.append(f"User gender preference: {user_preferences['gender']}")
    if user_preferences.get("preferred_occasions"):
        parts.append(f"Past occasions: {', '.join(user_preferences['preferred_occasions'])}")
    if user_preferences.get("wear_type"):
        parts.append(f"Prefers: {user_preferences['wear_type']} wear")

    if conversation_history:
        recent = conversation_history[-3:]
        history_str = "\n".join([
            f"  [{h.get('role', 'user')}]: {h.get('content', '')[:100]}"
            for h in recent
        ])
        parts.append(f"Recent conversation:\n{history_str}")

    return "\n".join(parts)


if __name__ == "__main__":
    # Test intent extraction
    test_queries = [
        "I need an outfit for a business meeting",
        "Something stylish for a summer beach vacation",
        "Wedding outfit for men, traditional",
        "Casual weekend look for women",
    ]
    print("=" * 60)
    print("  GEMINI INTENT EXTRACTION TEST")
    print("=" * 60)
    for q in test_queries:
        print(f"\nQuery: '{q}'")
        intent = extract_intent(q)
        print(f"  occasion={intent.get('occasion')} | "
              f"gender={intent.get('gender')} | "
              f"wear_type={intent.get('wear_type')} | "
              f"style={intent.get('style')} | "
              f"source={intent.get('_source')}")
