import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# 1. Charger les données
df = pd.read_csv("donnees_utilisateurs.csv")
X = df[['Publications_Par_Mois', 'Temps_Connexion_Minutes']]

# 2. DBSCAN est sensible aux échelles, on normalise les données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Initialiser et entraîner l'algorithme DBSCAN
# eps = distance max entre 2 points pour être voisins, min_samples = nb de points min pour faire un groupe
dbscan = DBSCAN(eps=0.35, min_samples=8)
df['Groupe_DBSCAN'] = dbscan.fit_predict(X_scaled)

# Note : Les points marqués "-1" sont considérés comme du bruit (outliers) par DBSCAN
print("\n--- Répartition DBSCAN (le groupe -1 est le 'bruit') ---")
print(df['Groupe_DBSCAN'].value_counts())

# 4. Créer le graphique
plt.figure(figsize=(10, 6))

# Dessiner les points (colorés selon le groupe DBSCAN)
scatter = plt.scatter(df['Publications_Par_Mois'], df['Temps_Connexion_Minutes'], 
                      c=df['Groupe_DBSCAN'], cmap='plasma', alpha=0.7, edgecolors='b')

plt.title("Segmentation des utilisateurs avec DBSCAN (Densité)")
plt.xlabel("Nombre de publications par mois")
plt.ylabel("Temps de connexion quotidien (minutes)")
plt.grid(True, linestyle='--', alpha=0.5)

# Afficher la fenêtre graphique
plt.show()
