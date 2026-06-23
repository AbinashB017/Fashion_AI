import sys
import os
import io
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

# Ensure src is in path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.loader import load_products, load_outfits
from src.embeddings.fashion_clip import load_cached_embeddings
from src.retrieval.faiss_index import build_faiss_index
from src.graph.compatibility_graph import load_graph
from src.rag.fashion_rag import load_rag
from src.ranking.scorer import CompatibilityScorer
from src.engine.outfit_engine import OutfitEngine

# Global singleton for the engine
engine: OutfitEngine = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    print("Initializing Dare XAI Outfit Engine...")
    products = load_products(validate=False)
    outfits = load_outfits(validate=False)
    try:
        embeddings, metadata = load_cached_embeddings()
    except FileNotFoundError:
        print("[Setup] Embeddings not found. Generating them on the fly (this takes a minute)...")
        from src.embeddings.fashion_clip import generate_product_embeddings
        embeddings, metadata = generate_product_embeddings(products)
    faiss_idx = build_faiss_index(embeddings, metadata, products, save=False)
    
    try:
        graph = load_graph()
    except FileNotFoundError:
        print("[Setup] Graph not found. Building it on the fly...")
        from src.graph.compatibility_graph import build_graph
        graph = build_graph(products, outfits, save=False)
        
    rag = load_rag(outfits)
    scorer = CompatibilityScorer(graph=graph, embeddings=embeddings, metadata=metadata)
    
    engine = OutfitEngine(
        products_df=products,
        outfits_df=outfits,
        faiss_index=faiss_idx,
        graph=graph,
        scorer=scorer,
        rag=rag
    )
    print("Engine ready!")
    yield
    print("Shutting down engine...")

app = FastAPI(title="Dare XAI Fashion API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow Vite frontend
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the images directory so frontend can load images directly
img_dir = ROOT / "images"
if img_dir.exists():
    app.mount("/images", StaticFiles(directory=str(img_dir)), name="images")

# --- Schemas ---
class ChatRequest(BaseModel):
    query: str
    gender_pref: str = "Any"
    style_strategy: str = "Balanced" # classic, modern, premium mapping
    n_outfits: int = 3

class SwapRequest(BaseModel):
    current_outfit: List[Dict[str, Any]]
    slot_to_swap: str
    occasion: str = "casual"
    gender: Optional[str] = None
    query: Optional[str] = None

# --- Endpoints ---

import math

def sanitize_nans(obj):
    if isinstance(obj, dict):
        return {k: sanitize_nans(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_nans(v) for v in obj]
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    return obj

@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    intent_override = None
    if req.gender_pref != "Any":
        intent_override = {"gender": req.gender_pref.lower()}
        
    try:
        result = engine.generate_outfits(req.query, intent=intent_override, n_outfits=req.n_outfits)
        return sanitize_nans(result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/upload")
async def upload_endpoint(file: UploadFile = File(...)):
    """Handles image upload and returns complementary outfits."""
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
        
    # Save the uploaded file temporarily
    temp_path = ROOT / "temp_upload.jpg"
    try:
        content = await file.read()
        with open(temp_path, "wb") as f:
            f.write(content)
            
        # Use existing method in OutfitEngine
        result = engine.find_complements_for_image(str(temp_path))
        
        # Transform the output to look like a standard outfit result for the UI
        # By taking the top complement per slot
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
            
        source_item = result["source_item"]
        complements = result["complements"]
        
        # Build one basic outfit from the complements
        outfit_items = [source_item]
        for slot, items in complements.items():
            if items:
                outfit_items.append(items[0])
                
        # Fake a score and explanation for now, or route through scorer
        outfit = {
            "theme": "Visual Match",
            "items": outfit_items,
            "score": {"percentage": 85},
            "explanation": "This outfit was curated based on visual similarity to your uploaded image.",
            "total_price": sum(item.get("price_inr", 0) for item in outfit_items),
        }
        
        return sanitize_nans({
            "intent": {"occasion": "visual match"},
            "outfits": [outfit],
            "source_item": source_item
        })
    finally:
        if temp_path.exists():
            temp_path.unlink()


@app.post("/api/swap")
async def swap_endpoint(req: SwapRequest):
    """
    Given a current outfit and a slot to swap, finds an alternative item.
    """
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
        
    # We will use the intent and query to retrieve candidates, 
    # then pick a candidate for the specified slot that is NOT already in the outfit.
    intent = {
        "occasion": req.occasion,
        "gender": req.gender
    }
    
    # Retrieve candidates
    pool = engine._hybrid_retrieve(intent, req.query or "")
    
    candidates = pool.get(req.slot_to_swap, [])
    if not candidates:
        raise HTTPException(status_code=404, detail=f"No alternatives found for slot: {req.slot_to_swap}")
        
    current_ids = [item["id"] for item in req.current_outfit]
    
    # Find best alternative
    alt_item = None
    for cand in candidates:
        if cand["id"] not in current_ids:
            alt_item = cand
            break
            
    if not alt_item:
        raise HTTPException(status_code=404, detail="No new alternatives left to swap.")
        
    # Replace item in outfit
    new_outfit_items = []
    for item in req.current_outfit:
        from src.engine.outfit_engine import CAT_TO_SLOT
        item_slot = CAT_TO_SLOT.get(item.get("category", ""), "unknown")
        if item_slot == req.slot_to_swap:
            new_outfit_items.append(alt_item)
        else:
            new_outfit_items.append(item)
            
    new_score = {"percentage": 80}
    if engine.scorer:
        ranked = engine.scorer.rank_outfits([new_outfit_items])
        new_score = ranked[0]["score"]
        
    from src.llm.gemini_client import generate_explanation
    outfit_for_explain = {
        "items": new_outfit_items,
        "occasion": req.occasion,
        "wear_type": "western",
        "gender": req.gender or "",
    }
    try:
        explanation = generate_explanation(
            user_query=req.query or "swapped outfit",
            outfit=outfit_for_explain,
            compatibility_score=new_score["percentage"]
        )
    except:
        explanation = engine._fallback_explanation(new_outfit_items, "Updated Look")
        
    return sanitize_nans({
        "items": new_outfit_items,
        "score": new_score,
        "explanation": explanation,
        "total_price": sum(item.get("price_inr", 0) for item in new_outfit_items),
        "swapped_in": alt_item
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
