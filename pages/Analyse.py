import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler

# Configuration globale de la page
st.set_page_config(page_title="Framework Universel de Clustering", layout="wide")

st.title(":material/analytics: Outil d'Analyse et de Segmentation de Données Universel")
st.write("Ce système générique segmente n'importe quelle base de données via Machine Learning.")

# 1. Zone d'importation dynamique
st.sidebar.header(":material/folder_open: Importation de données")
fichier_importe = st.sidebar.file_uploader("Déposez votre fichier de données (.CSV) :", type=["csv"])

df = None

if fichier_importe is not None:
    try:
        df = pd.read_csv(fichier_importe)
        st.sidebar.success("Fichier chargé avec succès !")
    except Exception as e:
        st.sidebar.error("Erreur de lecture du fichier.")
else:
    # Fichiers par défaut
    try:
        df = pd.read_csv("donnees_utilisateur2.0.csv")
    except:
        try:
            df = pd.read_csv("donnees_utilisateurs.csv")
        except:
            st.sidebar.warning("En attente d'un fichier CSV...")

if df is not None:
    # Identification automatique des colonnes numériques et texte
    colonnes_totales = df.columns.tolist()
    colonnes_numeriques = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    colonnes_texte = df.select_dtypes(include=['object']).columns.tolist()
    
    # Recherche automatique de la colonne d'identifiant (ID, User_ID, Client_ID, etc.)
    col_id = None
    candidates_id = ['id', 'user_id', 'id_utilisateur', 'utilisateur', 'name', 'nom', 'client', 'client_id', 'appareil_id']
    for c in colonnes_totales:
        if c.lower() in candidates_id:
            col_id = c
            break
    if not col_id and colonnes_texte:
        col_id = colonnes_texte[0]
        
    # Sélection automatique des 2 premières colonnes numériques détectées dans le fichier
    if len(colonnes_numeriques) >= 2:
        col_x = colonnes_numeriques[0]
        col_y = colonnes_numeriques[1]
        
        X = df[[col_x, col_y]]
        
        # 2. Choix de l'algorithme d'IA
        st.sidebar.header(":material/psychology: Modèle d'Apprentissage")
        choix_algo = st.sidebar.selectbox("Algorithme :", ["K-Means (Algorithme par Centres)", "DBSCAN (Algorithme par Densité)"])
        
        # 3. Menu de recherche d'une cible
        st.sidebar.header(":material/track_changes: Ciblage")
        liste_ids = df[col_id].astype(str).tolist() if col_id else [f"Ligne {i}" for i in range(len(df))]
        cible = st.sidebar.selectbox("Mettre en valeur un profil :", ["Aucun"] + liste_ids)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # --- EXÉCUTION DU MODÈLE SÉLECTIONNÉ ---
        if choix_algo == "K-Means (Algorithme par Centres)":
            nb_clusters = st.sidebar.slider("Nombre de grappes (K) :", min_value=2, max_value=6, value=3)
            model = KMeans(n_clusters=nb_clusters, random_state=42)
            df['Cluster_ID'] = model.fit_predict(X)
            
            # Affichage graphique
            scatter = ax.scatter(df[col_x], df[col_y], c=df['Cluster_ID'], cmap='viridis', alpha=0.6, edgecolors='k')
            centres = model.cluster_centers_
            ax.scatter(centres[:, 0], centres[:, 1], c='red', s=200, marker='X', label='Centres de gravité')
            
            legend1 = ax.legend(*scatter.legend_elements(), title="Groupes (Couleurs)", loc="upper left")
            ax.add_artist(legend1)
            ax.legend(loc="upper right")
            st.subheader(f"Analyse structurelle K-Means sur '{col_x}' et '{col_y}'")
            
        else:
            # Traitement DBSCAN avec normalisation
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            model = DBSCAN(eps=0.35, min_samples=5)
            df['Cluster_ID'] = model.fit_predict(X_scaled)
            
            scatter = ax.scatter(df[col_x], df[col_y], c=df['Cluster_ID'], cmap='plasma', alpha=0.6, edgecolors='k')
            
            legend1 = ax.legend(*scatter.legend_elements(), title="Groupes (-1 = Bruit)", loc="upper left")
            ax.add_artist(legend1)
            st.subheader(f"Analyse par densité DBSCAN sur '{col_x}' et '{col_y}'")
            
        # --- MISE EN AVANT D'UNE CIBLE SPÉCIFIQUE ---
        if cible != "Aucun":
            if col_id:
                idx_cible = df[df[col_id].astype(str) == str(cible)].index
            else:
                idx_cible = int(cible.split(" "))
                
            ligne_cible = df.loc[idx_cible].iloc[0] if hasattr(df.loc[idx_cible], 'iloc') else df.loc[idx_cible]
            ax.scatter(ligne_cible[col_x], ligne_cible[col_y], c='red', s=250, edgecolors='white', linewidth=3, label=f"Cible : {cible}")
            ax.legend(loc="upper right")
            
            st.markdown(f"**Profil ciblé : {cible}** | {col_x} : `{ligne_cible[col_x]}` | {col_y} : `{ligne_cible[col_y]}` | **Segment attribué : {ligne_cible['Cluster_ID']}**")

        # Remplacement dynamique des étiquettes des axes sur le graphique
        ax.set_xlabel(col_x.replace('_', ' '))
        ax.set_ylabel(col_y.replace('_', ' '))
        ax.grid(True, linestyle='--', alpha=0.3)
        st.pyplot(fig)
        
        # ==========================================
        # 📋 STATISTIQUES RÉELLES ET GÉNÉRIQUES
        # ==========================================
        st.write("---")
        st.subheader("Analyse statistique des segments détectés")
        
        groupes_uniques = sorted(df['Cluster_ID'].unique())
        cols_streamlit = st.columns(len(groupes_uniques))
        
        for idx, g in enumerate(groupes_uniques):
            with cols_streamlit[idx]:
                df_g = df[df['Cluster_ID'] == g]
                nb_lignes = len(df_g)
                moyen_x = df_g[col_x].mean()
                moyen_y = df_g[col_y].mean()
                
                nom_groupe = f"Groupe {g}" if g != -1 else "Données atypiques (Bruit)"
                
                if g == -1:
                    description = "Enregistrements isolés qui s'écartent du comportement général."
                elif idx == 0:
                    description = f"Segment avec des valeurs faibles sur les indicateurs '{col_x}' et '{col_y}'."
                elif idx == len(groupes_uniques) - 1:
                    description = f"Segment regroupant les valeurs les plus élevées de la base de données."
                else:
                    description = "Segment au comportement intermédiaire et modéré."
                
                st.markdown(f"### :material/groups: {nom_groupe}")
                st.markdown(f"**Effectif :** `{nb_lignes} lignes`")
                st.write(description)
                st.caption(f":material/trending_up: Moyennes : {moyen_x:.1f} | {moyen_y:.1f}")
        st.write("---")
        
        # 4. Section de téléchargement du fichier final
        st.subheader(":material/database: Base de données enrichie et exportable")
        csv_exportable = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Exporter le jeu de données segmenté (.CSV)",
            data=csv_exportable,
            file_name="data_segmentation_output.csv",
            mime="text/csv",
            icon=":material/download:"
        )
        st.dataframe(df)
        
    else:
        st.error("Le fichier injecté ne contient pas assez d'attributs numériques pour générer un partitionnement mathématique.")
