import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# 1. Charger les données générées à l'étape 1
df = pd.read_csv("donnees_utilisateurs.csv")

# 2. Extraire les colonnes numériques pour l'IA
X = df[['Publications_Par_Mois', 'Temps_Connexion_Minutes']]

# 3. Initialiser et entraîner l'algorithme K-Means (on veut 3 groupes)
kmeans = KMeans(n_clusters=3, random_state=42)
df['Groupe_KMeans'] = kmeans.fit_predict(X)

# 4. Afficher les résultats dans la console
print("\n--- Répartition des utilisateurs par groupe ---")
print(df['Groupe_KMeans'].value_counts())

# 5. Créer le graphique visuel
plt.figure(figsize=(10, 6))

# Dessiner les utilisateurs (colorés selon leur groupe)
scatter = plt.scatter(df['Publications_Par_Mois'], df['Temps_Connexion_Minutes'], 
                      c=df['Groupe_KMeans'], cmap='viridis', alpha=0.7, edgecolors='b')

# Dessiner le centre de chaque groupe (les gros points rouges)
centres = kmeans.cluster_centers_
plt.scatter(centres[:, 0], centres[:, 1], c='red', s=200, marker='X', label='Centres de groupes')

# Habillage du graphique
plt.title("Segmentation des utilisateurs avec K-Means")
plt.xlabel("Nombre de publications par mois")
plt.ylabel("Temps de connexion quotidien (minutes)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

# Afficher la fenêtre graphique
plt.show()
