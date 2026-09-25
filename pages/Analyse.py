import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler

# Configuration globale de la page
st.set_page_config(page_title="Analyse du comportement utilisateur", layout="wide")

st.title(":material/analytics: Outils de regroupement par comportement")
st.write("Segmentez les utilisateurs de votre plateforme selon leur comportement (interactions, activité, engagement...) grâce au Machine Learning.")

# --- Échelle de comportement utilisée pour nommer les groupes (du moins au plus actif) ---
ECHELLE_COMPORTEMENT = ["Très peu actif", "Peu actif", "Modéré", "Actif", "Très actif", "Extrêmement actif"]

def etiquettes_pour_n_groupes(n):
    """Choisit n libellés de comportement répartis sur l'échelle, du moins au plus actif."""
    if n <= 1:
        return [ECHELLE_COMPORTEMENT[len(ECHELLE_COMPORTEMENT) // 2]]
    if n <= len(ECHELLE_COMPORTEMENT):
        return [ECHELLE_COMPORTEMENT[round(i * (len(ECHELLE_COMPORTEMENT) - 1) / (n - 1))] for i in range(n)]
    # Cas rare (plus de groupes que d'échelons prévus, ex. DBSCAN) : on numérote au-delà de l'échelle
    return ECHELLE_COMPORTEMENT + [f"Groupe {i+1}" for i in range(len(ECHELLE_COMPORTEMENT), n)]

def construire_etiquettes_clusters(df, col_x, col_y):
    """Classe les clusters (hors bruit) du moins au plus actif et leur attribue un libellé."""
    groupes = sorted(g for g in df['Cluster_ID'].unique() if g != -1)
    scores = {g: df.loc[df['Cluster_ID'] == g, [col_x, col_y]].mean().mean() for g in groupes}
    ordre_par_score = sorted(groupes, key=scores.get)  # du moins au plus actif
    libelles = etiquettes_pour_n_groupes(len(ordre_par_score))
    etiquette_par_cluster = {g: libelles[rang] for rang, g in enumerate(ordre_par_score)}
    if -1 in df['Cluster_ID'].unique():
        etiquette_par_cluster[-1] = "Atypique (Bruit)"
    return etiquette_par_cluster, ordre_par_score

# 1. Zone d'importation dynamique
st.sidebar.header(":material/folder_open: Données utilisateurs")
fichier_importe = st.sidebar.file_uploader("Déposez l'export de vos données utilisateurs (.CSV) :", type=["csv"])

df = None

if fichier_importe is not None:
    try:
        df = pd.read_csv(fichier_importe)
        st.sidebar.success("Fichier chargé avec succès !")
    except Exception as e:
        st.sidebar.error("Erreur de lecture du fichier.")
else:
    # Fichier par défaut (ne contient aucune donnée d'achat)
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
        
        # --- EXÉCUTION DU MODÈLE SÉLECTIONNÉ ---
        fig = go.Figure()

        if choix_algo == "K-Means (Algorithme par Centres)":
            nb_clusters = st.sidebar.slider("Nombre de grappes (K) :", min_value=2, max_value=6, value=3)
            model = KMeans(n_clusters=nb_clusters, random_state=42)
            df['Cluster_ID'] = model.fit_predict(X)

            etiquette_par_cluster, ordre_par_score = construire_etiquettes_clusters(df, col_x, col_y)

            # Un point par utilisateur, coloré et étiqueté par profil de comportement
            for g in sorted(df['Cluster_ID'].unique()):
                sous_df = df[df['Cluster_ID'] == g]
                identifiants = sous_df[col_id].astype(str) if col_id else sous_df.index.astype(str)
                fig.add_trace(go.Scatter(
                    x=sous_df[col_x], y=sous_df[col_y], mode='markers',
                    name=etiquette_par_cluster[g],
                    marker=dict(size=9, line=dict(width=1, color='DarkSlateGrey')),
                    text=identifiants,
                    customdata=[etiquette_par_cluster[g]] * len(sous_df),
                    hovertemplate="<b>%{text}</b><br>" + f"{col_x}" + ": %{x}<br>" + f"{col_y}" + ": %{y}<br>Profil : %{customdata}<extra></extra>"
                ))

            centres = model.cluster_centers_
            fig.add_trace(go.Scatter(
                x=centres[:, 0], y=centres[:, 1], mode='markers', name='Centres de gravité',
                marker=dict(symbol='x', size=14, color='red', line=dict(width=2)),
                hoverinfo='skip'
            ))
            st.subheader(f"Analyse structurelle K-Means sur '{col_x}' et '{col_y}'")

        else:
            # Traitement DBSCAN avec normalisation
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            model = DBSCAN(eps=0.35, min_samples=5)
            df['Cluster_ID'] = model.fit_predict(X_scaled)

            etiquette_par_cluster, ordre_par_score = construire_etiquettes_clusters(df, col_x, col_y)

            for g in sorted(df['Cluster_ID'].unique()):
                sous_df = df[df['Cluster_ID'] == g]
                identifiants = sous_df[col_id].astype(str) if col_id else sous_df.index.astype(str)
                fig.add_trace(go.Scatter(
                    x=sous_df[col_x], y=sous_df[col_y], mode='markers',
                    name=etiquette_par_cluster[g],
                    marker=dict(size=9, line=dict(width=1, color='DarkSlateGrey')),
                    text=identifiants,
                    customdata=[etiquette_par_cluster[g]] * len(sous_df),
                    hovertemplate="<b>%{text}</b><br>" + f"{col_x}" + ": %{x}<br>" + f"{col_y}" + ": %{y}<br>Profil : %{customdata}<extra></extra>"
                ))
            st.subheader(f"Analyse par densité DBSCAN sur '{col_x}' et '{col_y}'")

        # --- MISE EN AVANT D'UNE CIBLE SPÉCIFIQUE ---
        if cible != "Aucun":
            if col_id:
                idx_cible = df[df[col_id].astype(str) == str(cible)].index
            else:
                idx_cible = int(cible.split(" "))
                
            ligne_cible = df.loc[idx_cible].iloc[0] if hasattr(df.loc[idx_cible], 'iloc') else df.loc[idx_cible]
            fig.add_trace(go.Scatter(
                x=[ligne_cible[col_x]], y=[ligne_cible[col_y]], mode='markers',
                name=f"Cible : {cible}",
                marker=dict(size=16, color='red', line=dict(width=3, color='white')),
                text=[str(cible)],
                customdata=[etiquette_par_cluster[ligne_cible['Cluster_ID']]],
                hovertemplate="<b>%{text}</b><br>" + f"{col_x}" + ": %{x}<br>" + f"{col_y}" + ": %{y}<br>Profil : %{customdata}<extra></extra>"
            ))
            
            st.markdown(f"**Profil ciblé : {cible}** | {col_x} : `{ligne_cible[col_x]}` | {col_y} : `{ligne_cible[col_y]}` | **Segment attribué : {etiquette_par_cluster[ligne_cible['Cluster_ID']]}**")

        # Remplacement dynamique des étiquettes des axes, graphique zoomable/déplaçable
        fig.update_layout(
            xaxis_title=col_x.replace('_', ' '),
            yaxis_title=col_y.replace('_', ' '),
            hovermode='closest',
            dragmode='zoom',
            height=480,
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(
                orientation="h",
                yanchor="bottom", y=-0.35,
                xanchor="center", x=0.5,
                title=None,
            ),
        )
        st.plotly_chart(fig, use_container_width=True)
        
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
                
                nom_groupe = etiquette_par_cluster[g] if g != -1 else "Atypique (Bruit)"
                
                if g == -1:
                    description = "Enregistrements isolés qui s'écartent du comportement général."
                elif g == ordre_par_score[0]:
                    description = f"Segment avec des valeurs faibles sur les indicateurs '{col_x}' et '{col_y}'."
                elif g == ordre_par_score[-1]:
                    description = f"Segment regroupant les valeurs les plus élevées de la base de données."
                else:
                    description = "Segment au comportement intermédiaire et modéré."
                
                st.markdown(f"### :material/groups: {nom_groupe}")
                st.markdown(f"**Effectif :** `{nb_lignes} lignes`")
                st.write(description)
                st.caption(f":material/trending_up: Moyennes : {moyen_x:.1f} | {moyen_y:.1f}")
        st.write("---")
        
        # 4. Section de téléchargement du fichier final
        st.subheader(":material/database: Utilisateurs segmentés")
        csv_exportable = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Exporter les utilisateurs segmentés (.CSV)",
            data=csv_exportable,
            file_name="data_segmentation_output.csv",
            mime="text/csv",
            icon=":material/download:"
        )
        st.dataframe(df)
        
    else:
        st.error("Le fichier importé ne contient pas assez d'indicateurs numériques (ex : interactions, activité) pour segmenter les utilisateurs.")
        
