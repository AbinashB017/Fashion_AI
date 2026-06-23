"""
src/ui/styles.py
----------------
Custom CSS to give the Streamlit app a premium, dark-mode, glassmorphism UI.
"""

def get_custom_css() -> str:
    return """
<style>
    /* Import modern Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

    /* Global Typography & Background */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif !important;
    }
    
    .stApp {
        background-color: #0d1117;
        color: #e6edf3;
    }

    /* Hide Streamlit default UI elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }

    /* Chat Bubbles */
    .stChatMessage {
        background-color: transparent !important;
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .stChatMessage[data-testid="chat-message-user"] {
        background-color: #1f2937 !important;
        border: 1px solid #374151;
    }
    .stChatMessage[data-testid="chat-message-assistant"] {
        background-color: rgba(99, 102, 241, 0.05) !important;
        border: 1px solid rgba(99, 102, 241, 0.2);
    }

    /* Premium Outfit Cards (Glassmorphism) */
    div.outfit-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    div.outfit-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5), 0 8px 10px -6px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(99, 102, 241, 0.5);
    }

    /* Outfit Headers */
    .outfit-theme {
        font-size: 1.4rem;
        font-weight: 600;
        color: #fff;
        margin-bottom: 0.2rem;
        background: -webkit-linear-gradient(45deg, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .outfit-price {
        font-size: 1.1rem;
        font-weight: 500;
        color: #10b981;
        margin-bottom: 1rem;
    }

    /* Metrics & Scores */
    .score-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 8px;
        margin-bottom: 10px;
    }
    .score-high { background-color: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); }
    .score-med  { background-color: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.3); }
    
    /* Product Items List */
    .item-list {
        list-style-type: none;
        padding-left: 0;
        margin-top: 1rem;
    }
    .item-list li {
        padding: 8px 0;
        border-bottom: 1px solid rgba(255,255,255,0.05);
        font-size: 0.95rem;
    }
    .item-list li:last-child {
        border-bottom: none;
    }
    .item-category {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #9ca3af;
        display: block;
    }

    /* Explanation Text */
    .stylist-note {
        margin-top: 1rem;
        padding: 1rem;
        background-color: rgba(0,0,0,0.2);
        border-left: 3px solid #818cf8;
        border-radius: 4px;
        font-size: 0.95rem;
        line-height: 1.5;
        color: #cbd5e1;
        font-style: italic;
    }

    /* Progress bar overrides */
    .stProgress > div > div > div > div {
        background-image: linear-gradient(to right, #6366f1, #a855f7);
    }
</style>
"""
