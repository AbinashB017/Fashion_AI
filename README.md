# DareXAI — AI Fashion Stylist 👗🤖

An intelligent, multi-modal fashion recommendation engine and chat-based stylist. 

DareXAI allows users to ask natural language questions (e.g., *"I need an outfit for a business meeting"* or *"Suggest something stylish for a summer beach vacation"*) or upload images to receive curated, personalized, and explainable fashion recommendations.

---

## 🌟 Key Features

1. **Multi-Modal Conversational Interface**
   - Natural Language Understanding (NLU) powered by cascaded LLMs (Gemini & Groq).
   - Drag-and-Drop Image Upload powered by **FashionCLIP** for visual similarity search.
   - Interactive React UI with hover-to-swap outfit functionality.

2. **High-Speed Hybrid Retrieval Engine**
   - Combines metadata filtering, FAISS semantic search, and Compatibility Graph edges.
   - Capable of assembling millions of potential outfit permutations in under 100ms.

3. **Intelligent Cascading Architecture**
   - Built for high availability. Automatically cascades API requests: `Gemini Key 1` → `Gemini Key 2` → `Groq (Llama-3.3-70b)` → `Heuristics` to completely eliminate rate-limit downtime.
   - LLM generation processes asynchronously via ThreadPool execution for blazing fast responses.

4. **Explainable AI (XAI)**
   - Every recommended outfit includes a dynamically generated stylist rationale explaining *why* the pieces work together based on color theory, occasion, and silhouette balance.

---

## 🧠 System Architecture

The architecture is divided into three core layers:

### 1. The Frontend (React + Vite)
- Built with modern React and TailwindCSS.
- Manages chat state, image uploading, and outfit rendering.
- Features dynamic carousels and smooth micro-animations.

### 2. The Retrieval & Recommendation Engine (Python)
- **Intent Extractor:** LLM-based parser that converts user chat into structured JSON (gender, occasion, style, budget).
- **FashionCLIP Embedder:** Uses `patrickjohncyh/fashion-clip` to map text and images into a shared 512-dimensional vector space. Runs as a Global Singleton to prevent memory bottlenecks.
- **FAISS Indexer:** Handles the semantic similarity matching of candidate items.
- **Outfit Assembler:** A graph-based algorithm that ensures outfits are structurally complete (Top + Bottom + Footwear).

### 3. The LLM Explanation Layer (FastAPI)
- Uses an Asynchronous ThreadPool to rapidly hit Groq/Gemini APIs in parallel.
- Analyzes the final curated outfit array and generates a personalized styling explanation for the user.

---

## 🚀 Setup Instructions

### Prerequisites
- Python 3.10+
- Node.js 18+
- API Keys: Google Gemini (Free Tier) and Groq (Free Tier)
- HuggingFace Access Token

### 1. Clone the Repository
```bash
git clone https://github.com/AbinashB017/Fashion_AI.git
cd Fashion_AI
```

### 2. Backend Setup
Create a virtual environment and install dependencies:
```bash
python -m venv venv

# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate

pip install -r requirements.txt
```

Create a `.env` file in the root directory:
```env
GEMINI_API_KEY_1=your_gemini_key
GEMINI_API_KEY_2=your_gemini_key_optional
GROQ_API_KEY_1=your_groq_key
GROQ_API_KEY_2=your_groq_key_optional
HF_TOKEN=your_huggingface_read_token
```

### 3. Generate the AI Embeddings
Before starting the server, you need to generate the local FAISS index and AI embeddings:
```bash
python src/embeddings/fashion_clip.py
python src/retrieval/faiss_index.py
```

### 4. Start the Application
**Terminal 1 (Backend):**
```bash
# Keep your virtual environment activated
python backend/main.py
```

**Terminal 2 (Frontend):**
```bash
cd frontend
npm install
npm run dev
```

Visit `http://localhost:5173` in your browser to start styling!
