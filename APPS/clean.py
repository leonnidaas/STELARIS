"""
Petit scrip générté par Gemini pour faire le ménage dans les tuiles LiDAR téléchargées, en comparant avec une liste d'URLs IGN.
Il supprime les fichiers qui ne sont pas dans la liste, en gardant uniquement ceux qui sont nécessaires pour la trajectoire.
Il n'est pas sencé etre utilisé dans le pipeline de labellisation, mais peut être utile pour faire du ménage dans les dossiers de tuiles avant de lancer le pipeline.
"""


from pathlib import Path

def clean_stelar_tiles(txt_urls_path, folder_path):
    """
    Compare les fichiers locaux avec une liste d'URLs IGN et supprime les surplus.
    """
    # 1. Extraire les noms de fichiers depuis les URLs
    # Path(url).name récupère "LHD_FXX_...copc.laz" à partir de l'URL complète
    with open(txt_urls_path, 'r') as f:
        to_keep = {Path(line.strip()).name for line in f if line.strip()}

    folder = Path(folder_path)
    if not folder.exists():
        print(f"❌ Erreur : Le dossier {folder} n'existe pas.")
        return

    # 2. Lister les fichiers actuellement dans le dossier
    current_files = {f.name for f in folder.iterdir() if f.is_file()}

    # 3. Calculer les fichiers à supprimer (ceux qui sont là mais pas dans la liste)
    extras = current_files - to_keep

    if not extras:
        print("✅ Le dossier est déjà propre (tous les fichiers correspondent à la liste).")
        return

    print(f"📦 Analyse terminée : {len(to_keep)} fichiers à garder, {len(extras)} à supprimer.")

    # 4. Suppression
    for file_name in extras:
        file_path = folder / file_name
        try:
            # Sécurité : on ne supprime que les fichiers .laz ou .copc.laz
            if file_path.suffix in ['.laz', '.copc']:
                file_path.unlink() # Suppression réelle
                print(f"🗑️ Supprimé : {file_name}")
        except Exception as e:
            print(f"❌ Impossible de supprimer {file_name} : {e}")

if __name__ == "__main__":
    # Exemple d'utilisation
    txt_list_path = "/media/leon_peltzer/DATA/leon/projet_cara/DATA/01_INTERIM/BORDEAUX_COUTRAS/tiles_laz/urls.txt"
    folder_to_clean = "/media/leon_peltzer/DATA/leon/projet_cara/DATA/01_INTERIM/BORDEAUX_COUTRAS/tiles_laz"
    clean_stelar_tiles(txt_list_path, folder_to_clean)