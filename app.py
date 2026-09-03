import streamlit as st
import os

# Configuration de la page d'accueil
st.set_page_config(page_title="Accueil - Plateforme de Clustering", layout="centered")

# --- SYSTÈME VISUEL D'ARRIÈRE-PLAN ---
image_chargee = False

fichier_image = None
for ext in ["image_ecole.jpg", "image_ecole.jpeg", "image_ecole.png"]:
    if os.path.exists(ext):
        fichier_image = ext
        break

if fichier_image is not None:
    try:
        import base64
        type_mime = "image/png" if fichier_image.endswith(".png") else "image/jpeg"
        
        with open(fichier_image, 'rb') as f:
            data = f.read()
        img_base64 = base64.b64encode(data).decode()
        
        # NOTE : On passe l'opacité à 0.40 pour rendre l'image beaucoup plus nette et visible !
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: linear-gradient(rgba(255, 255, 255, 0.40), rgba(255, 255, 255, 0.40)), url("data:{type_mime};base64,{img_base64}");
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
            }}
            /* Petit bonus pour masquer la barre latérale uniquement sur l'accueil */
            [data-testid="stSidebar"] {{
                display: none;
            }}
            [data-testid="collapsedControl"] {{
                display: none;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
        image_chargee = True
    except:
        pass

if not image_chargee:
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 100%);
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# --- CONTENU DE LA PAGE D'ACCUEIL ---
st.write("")
st.write("")

# Bloc blanc modernisé contenant uniquement votre nouveau titre "CLUSTERING"
st.markdown(
    """
    <div style='background-color: rgba(255, 255, 255, 0.85); padding: 2.5rem; border-radius: 1rem; box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1); margin-bottom: 2rem;'>
        <h1 style='text-align: center; color: #1e3a8a; margin: 0; font-size: 3rem; font-weight: 800; letter-spacing: 2px;'>CLUSTERING</h1>
    </div>
    """, 
    unsafe_allow_html=True
)

# Création des colonnes pour aligner les boutons
col1, col2 = st.columns(2)

with col1:
    if st.button("Informations sur la plateforme", icon=":material/info:", use_container_width=True):
        st.info(
            "**À propos de cette application :**\n\n"
            "Ce plateforme est un outil d'analyse comportementale et de segmentation de données, "
            "capable de partitionner et segmenter des données comportementales (Réseaux sociaux, "
            "sessions Wi-Fi, transactions de supermarchés, etc.).\n\n"
            "**Spécifications techniques :** Python, Streamlit, (K-Means & DBSCAN)."
        )

with col2:
    st.markdown(
        """
        <a href="/Analyse" target="_self" style="text-decoration: none;">
            <button style="
                width: 100%;
                background-color: #1e3a8a;
                color: white;
                border: none;
                padding: 0.55rem;
                font-size: 1rem;
                border-radius: 0.375rem;
                cursor: pointer;
                font-weight: 600;
                display: flex;
                align-items: center;
                justify-content: center;
                gap: 0.5rem;
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
                height: 38px;">
                Accéder à l'analyse
            </button>
        </a>
        """,
        unsafe_allow_html=True
    )
