import streamlit as st


_THEME_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&display=swap');

    :root {
        --bg-top: #f4f8ff;
        --bg-bottom: #f8fcf7;
        --surface: rgba(255, 255, 255, 0.86);
        --border: #d5e3f6;
        --text: #13243a;
        --muted: #4f637d;
        --primary: #0077b6;
        --accent: #0f9d76;
    }

    html, body, [class*="css"] {
        font-family: 'Space Grotesk', sans-serif;
        color: var(--text);
    }

    .stApp {
        background:
            radial-gradient(1300px 500px at 8% -10%, #d7e8ff 0%, transparent 60%),
            radial-gradient(900px 500px at 95% -20%, #d6f4e3 0%, transparent 60%),
            linear-gradient(180deg, var(--bg-top), var(--bg-bottom));
    }

    .hero {
        background: linear-gradient(120deg, rgba(0,119,182,0.14), rgba(15,157,118,0.11));
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 18px 22px;
        margin: 6px 0 16px 0;
        box-shadow: 0 8px 22px rgba(15, 55, 90, 0.06);
    }

    .hero h1 {
        margin: 0;
        letter-spacing: 0.2px;
        font-weight: 700;
        font-size: 1.8rem;
    }

    .hero p {
        margin: 8px 0 0 0;
        color: var(--muted);
        font-size: 0.98rem;
    }

    div[data-testid="stMetric"] {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 10px 12px;
        box-shadow: 0 8px 18px rgba(18, 38, 66, 0.06);
    }

    .section-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 16px;
        box-shadow: 0 8px 18px rgba(18, 38, 66, 0.06);
    }

    .quick-start {
        margin: 0;
        padding-left: 18px;
        color: var(--text);
    }

    .quick-start li {
        margin-bottom: 6px;
    }
</style>
"""


border_hover = "#0077B6"  # Bordure néon pour le hover
THEME_SIDERBAR_CSS = """
<style>
    /* 1. Style de base pour chaque item de navigation */
    [data-testid="stSidebarNavItems"] li {
        margin-bottom: 8px;
        border-radius: 8px;
        transition: all 0.3s ease;
        border: 1px solid transparent;
    }

    /* 2. Style au survol (Hover) */
    [data-testid="stSidebarNavItems"] li:hover {
        background-color: rgba(0, 212, 255, 0.1); /* Fond bleu très léger */
        border: 1px solid #0077B6; /* Bordure néon */
        transform: translateX(5px); /* Petit décalage vers la droite */
    }

    /* 3. Style du texte à l'intérieur */
    [data-testid="stSidebarNavItems"] span {
        color: #000000;
        font-family: 'Inter', sans-serif;
        font-weight: 500;
    }

    /* 4. Effet de "Glow" sur le texte au survol */
    [data-testid="stSidebarNavItems"] li:hover span {
        color: #0077B6;
        text-shadow: 0 0 10px rgba(0, 212, 255, 0.5);
    }

    [data-testid="stSidebarNavLink"][aria-current="page"] {
    background: linear-gradient(90deg, rgba(0, 212, 255, 0.2) 0%, rgba(0, 212, 255, 0) 100%) !important;
    border-left: 5px solid #0077B6 !important; /* Ta barre néon à gauche */
    font-weight: bold !important;
    }

    /* 2. Style du texte de la page active pour qu'il brille */
    [data-testid="stSidebarNavLink"][aria-current="page"] span {
        color: #0077B6 !important;
        text-shadow: 0 0 10px rgba(0, 212, 255, 0.6);
    }

    /* 3. Petit bonus : on cache la puce par défaut de Streamlit si elle te gêne */
    [data-testid="stSidebarNavSeparator"] {
        display: none !important;
    }
    </style>
    """

def apply_theme() -> None:
    st.markdown(_THEME_CSS, unsafe_allow_html=True)
    st.markdown(THEME_SIDERBAR_CSS, unsafe_allow_html=True)

def render_hero(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="hero">
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
