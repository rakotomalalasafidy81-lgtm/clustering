import pandas as pd
from sklearn.datasets import make_blobs

# On génère 300 faux utilisateurs regroupés en 3 comportements types
X, y = make_blobs(n_samples=300, centers=3, cluster_std=6.0, center_box=(20, 80), random_state=42)

# On organise les données dans un tableau propre
df = pd.DataFrame(X, columns=['Publications_Par_Mois', 'Temps_Connexion_Minutes'])

# On s'assure que les valeurs restent positives et réalistes
df['Publications_Par_Mois'] = df['Publications_Par_Mois'].clip(lower=0).astype(int)
df['Temps_Connexion_Minutes'] = df['Temps_Connexion_Minutes'].clip(lower=5).astype(int)

# On ajoute un identifiant unique pour chaque utilisateur
df['User_ID'] = [f"User_{i}" for i in range(1, 301)]

# On sauvegarde le tout dans un fichier CSV
df.to_csv("donnees_utilisateurs.csv", index=False)
print("Succès : Le fichier 'donnees_utilisateurs.csv' a été généré !")
