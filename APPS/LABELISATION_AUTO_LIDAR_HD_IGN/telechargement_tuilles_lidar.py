import os
import sys
import argparse
import requests
from requests.adapters import HTTPAdapter
import laspy
import time
import threading
import json
from tqdm import tqdm
from pathlib import Path

# Session HTTP réutilisable par thread (connection pooling, keep-alive)
_thread_local = threading.local()

def _get_session(pool_connections=4, pool_maxsize=16):
    """Retourne une session requests persistante pour le thread courant."""
    session = getattr(_thread_local, 'session', None)
    if session is None:
        session = requests.Session()
        adapter = HTTPAdapter(
            pool_connections=pool_connections,
            pool_maxsize=pool_maxsize,
            max_retries=0,  # On gère les retries manuellement
        )
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        _thread_local.session = session
    return session


class RateLimiter:
    def __init__(self, requests_per_second):
        self.requests_per_second = max(float(requests_per_second), 1.0)
        self.min_interval = 1.0 / self.requests_per_second
        self.lock = threading.Lock()
        self.last_call = 0.0

    def wait(self):
        with self.lock:
            now = time.monotonic()
            elapsed = now - self.last_call
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
            self.last_call = time.monotonic()


def get_remote_file_size(url, timeout=50, rate_limiter=None):
    """Récupère la taille du fichier distant via HEAD request."""
    try:
        if rate_limiter is not None:
            rate_limiter.wait()
        session = _get_session()
        response = session.head(url, timeout=timeout, allow_redirects=True)
        if response.status_code == 200:
            content_length = response.headers.get('Content-Length')
            if content_length:
                return int(content_length)
    except Exception:
        pass
    return None


def load_size_cache(cache_file):
    if not cache_file or not os.path.exists(cache_file):
        return {}
    try:
        with open(cache_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


def save_size_cache(cache_file, cache_data):
    if not cache_file:
        return
    try:
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def build_remote_size_map(urls, rate_limiter, cache_file=None, refresh=False, verbose=True):
    """Construit une carte des tailles distantes pour une liste d'URLs, en utilisant un cache local.

    """
    cache = {} if refresh else load_size_cache(cache_file)
    size_map = {}

    urls_to_query = []
    for url in urls:
        cached_size = cache.get(url)
        if isinstance(cached_size, int) and cached_size > 0:
            size_map[url] = cached_size
        else:
            urls_to_query.append(url)

    if urls_to_query:
        iterator = tqdm(urls_to_query, desc="Préchargement tailles", unit="file", disable=not verbose)
        for url in iterator:
            size = get_remote_file_size(url, rate_limiter=rate_limiter)
            if isinstance(size, int) and size > 0:
                size_map[url] = size
                cache[url] = size
            else:
                # Garder une trace explicite des URLs testées sans taille exploitable
                # (HEAD sans Content-Length, accès refusé, timeout, etc.).
                # Cela rend le JSON exhaustif même si certaines tailles restent inconnues.
                if url not in cache:
                    cache[url] = 0

    save_size_cache(cache_file, cache)
    return size_map


def load_cached_size_map(urls, cache_file=None):
    cache = load_size_cache(cache_file)
    size_map = {}
    for url in urls:
        cached_size = cache.get(url)
        if isinstance(cached_size, int) and cached_size > 0:
            size_map[url] = cached_size
    return size_map

def verify_laz_integrity(filepath):
    """Vérifie l'intégrité d'un fichier LAZ en tentant de l'ouvrir avec laspy."""
    try:
        with laspy.open(filepath) as laz_file:
            # Simple lecture de l'en-tête pour vérifier que le fichier est valide
            header = laz_file.header
            return header.point_count > 0
    except Exception:
        return False

def is_file_valid(filepath, expected_size=None, verify_integrity=False):
    """
    Vérifie si un fichier est valide.
    
    Args:
        filepath: Chemin du fichier
        expected_size: Taille attendue en octets (optionnel)
        verify_integrity: Si True, vérifie l'intégrité du fichier LAZ
    
    Returns:
        bool: True si le fichier est valide
    """
    if not os.path.exists(filepath):
        return False
    file_size = os.path.getsize(filepath)
    # Si le fichier est vide ou trop petit, il est invalide
    if file_size < 1024:  # Moins de 1 Ko
        return False
    # Si on a une taille attendue, on la compare
    if expected_size is not None:
        if file_size != expected_size:
            return False
    # Vérification optionnelle de l'intégrité
    if verify_integrity and filepath.lower().endswith(('.laz', '.las')):
        return verify_laz_integrity(filepath)
    
    return True


def link_tiles_to_trajectory(traj_tiles_dir: Path, url_list: list, pool_dir: Path):
    """Vérifie l'intégrité des liens et les renouvelle si nécessaire."""
    
    for entry in url_list:
        filename = entry.split('/')[-1] if '://' in entry else entry
        source = pool_dir / filename
        target = traj_tiles_dir / filename
        
        # 1. Sécurité : Si le fichier source a disparu du pool, on ne peut pas lier
        if not source.exists():
            print(f"Source absente du pool : {filename}")
            continue

        # 2. Détection et nettoyage des liens morts
        # is_symlink() est True si le raccourci existe (même mort)
        # .exists() est False si le lien pointe vers un fichier inexistant
        if target.is_symlink() and not target.exists():
            print(f"Renouvellement du lien mort : {filename}")
            target.unlink()  # On supprime le raccourci cassé avant de le recréer

        # 3. Création du lien s'il n'existe pas (ou s'il vient d'être supprimé)
        # follow_symlinks=False est vital pour ne pas suivre le lien vers la source
        if not target.exists(follow_symlinks=False):
            os.symlink(source.absolute(), target)


def check_and_download(url, output_folder, chunk_size=10*1024*1024, verify_integrity=False,
    expected_size=None, max_retries=3, verbose_file=False):
    """
    Télécharge un fichier si nécessaire.
    
    Args:
        url: URL du fichier
        output_folder: Dossier de destination
        chunk_size: Taille des chunks de téléchargement
        verify_integrity: Vérifier l'intégrité du fichier LAZ
        expected_size: Taille attendue du fichier
        verbose_file: Afficher une barre de progression pour ce fichier
    
    Returns:
        tuple: (status, size_downloaded) où status in ("EXIST", "DONE", "FAIL") et size_downloaded en octets
    """
    url = url.strip()
    if not url:
        return "SKIP", 0
    
    filename = url.split('/')[-1]
    filepath = os.path.join(output_folder, filename)
    
    # La taille distante est préchargée une seule fois en amont

    # Vérifier si le fichier existe déjà et est valide
    if is_file_valid(filepath, expected_size, verify_integrity):
        return "EXIST", expected_size if expected_size else 0

    # Si le fichier existe mais est invalide, le supprimer
    if os.path.exists(filepath):
        try:
            os.remove(filepath)
        except Exception:
            pass

    session = _get_session()
    for attempt in range(max_retries):
        temp_filepath = filepath + ".tmp"
        try:
            with session.get(url, stream=True, timeout=120) as r:
                if r.status_code == 429:
                    retry_after = r.headers.get("Retry-After")
                    wait_seconds = float(retry_after) if retry_after else (2 ** attempt)
                    time.sleep(wait_seconds)
                    continue

                r.raise_for_status()

                total = expected_size
                if total is None:
                    content_length = r.headers.get("Content-Length")
                    if content_length and content_length.isdigit():
                        total = int(content_length)

                pbar = None
                if verbose_file:
                    pbar = tqdm(
                        total=total,
                        unit='B',
                        unit_scale=True,
                        unit_divisor=1024,
                        leave=False,
                        mininterval=0.2,
                        desc=filename[:40],
                    )

                with open(temp_filepath, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            if pbar:
                                pbar.update(len(chunk))

                if pbar:
                    pbar.close()

            if is_file_valid(temp_filepath, expected_size, verify_integrity):
                os.rename(temp_filepath, filepath)
                actual_size = os.path.getsize(filepath)
                return "DONE", actual_size

            os.remove(temp_filepath)
            return "FAIL", 0

        except Exception as e:
            if attempt == max_retries - 1:
                print(f"\n❌ Erreur sur {filename} : {type(e).__name__} - {e}")
            else:
                time.sleep(2 ** attempt)
        finally:
            if os.path.exists(temp_filepath):
                try:
                    os.remove(temp_filepath)
                except Exception:
                    pass

    return "FAIL", 0


def download_tiles(traj_tiles_dir, file_list, output_folder, max_workers=10, chunk_size=10*1024*1024,
                   verify_integrity=False, verbose=True, rate_limit=10,
                   size_cache_file=None, refresh_sizes=False, prefetch_sizes=False):
    """
    Télécharge les tuiles LiDAR depuis une liste d'URLs.
    
    Args:
        traj_tiles_dir: Dossier contenant les tuiles pour le trajet
        file_list: Chemin du fichier contenant les URLs
        output_folder: Dossier de destination
        max_workers: Conservé pour compatibilité, ignoré (mode séquentiel forcé à 1)
        chunk_size: Taille (octets) de chaque bloc lu depuis HTTP puis écrit sur disque
        verify_integrity: Vérifier l'intégrité des fichiers LAZ
        verbose: Afficher les messages de progression
        rate_limit: Limite globale de requêtes API par seconde
        size_cache_file: Fichier JSON de cache des tailles distantes (Content-Length)
        refresh_sizes: Si True, ignore le cache et recharge les tailles
        prefetch_sizes: Paramètre conservé pour compatibilité.
            Les tailles sont désormais calculées automatiquement si le cache est absent
            ou incomplet pour les URLs à télécharger.

    Détails des arguments liés aux tailles:
        - chunk_size:
            Impacte le compromis CPU/système d'I/O. Plus grand => moins d'appels Python, mais
            moins de réactivité de la barre par fichier. Plus petit => updates plus fluides,
            mais overhead plus important.
        - size_cache_file:
            Stocke les tailles distantes pour éviter de refaire des HEAD à chaque exécution.
            Utilisé pour estimer le volume total et valider la taille des fichiers.
        - refresh_sizes:
            Force la reconstruction des tailles distantes depuis l'API et écrase le cache.
        - prefetch_sizes:
            Conservé pour compatibilité. En pratique, le cache est désormais complété
            automatiquement dès qu'il manque des tailles pour les URLs à télécharger.
    
    Returns:
        dict: Statistiques avec les clés 'exist', 'done', 'fail'
    
    Raises:
        FileNotFoundError: Si le fichier de liste n'existe pas
        ValueError: Si le fichier de liste est vide
    """
    # Créer le dossier de sortie si nécessaire
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if traj_tiles_dir is not None:
        traj_tiles_dir = Path(traj_tiles_dir)
    
    # Lire la liste des URLs
    if not os.path.exists(file_list):
        raise FileNotFoundError(f"Le fichier {file_list} n'existe pas.")
    
    with open(file_list, 'r') as f:
        all_urls = [line.strip() for line in f if line.strip()]
    
    if not all_urls:
        raise ValueError(f"Aucune URL trouvée dans {file_list}")
    
    # Analyse des fichiers existants
    if verbose:
        print(f"Analyse de {len(all_urls)} fichiers...")
    
    urls_to_download = []
    for url in all_urls:
        filename = url.split('/')[-1]
        filepath = os.path.join(output_folder, filename)
        
        # Vérification rapide sans télécharger la taille distante
        if not is_file_valid(filepath, expected_size=None, verify_integrity=False):
            urls_to_download.append(url)
    
    if verbose:
        print(f"Fichiers déjà présents (non vérifiés) : {len(all_urls) - len(urls_to_download)}")
        print(f"Fichiers à télécharger ou vérifier : {len(urls_to_download)}")
    
    if not urls_to_download:
        if traj_tiles_dir is not None:
            link_tiles_to_trajectory(traj_tiles_dir, all_urls, Path(output_folder)) # /!\ output_folder doit être le pool de tuiles, pas le dossier de la trajectoire
        if verbose:
            print("Tous les fichiers sont déjà présents")
        return {"exist": len(all_urls), "done": 0, "fail": 0, "total_size_bytes": 0}
    
    stats = {"exist": 0, "done": 0, "fail": 0, "total_size_bytes": 0, 
             "exist_size_bytes": 0, "done_size_bytes": 0}
    rate_limiter = RateLimiter(rate_limit)

    if size_cache_file is None:
        cache_base_dir = traj_tiles_dir if traj_tiles_dir is not None else Path(output_folder)
        size_cache_file = str(cache_base_dir / "remote_sizes_cache.json")

    cache_preview = load_cached_size_map(urls_to_download, size_cache_file)
    cache_covers_all_urls = len(cache_preview) == len(urls_to_download)
    cache_file_exists = os.path.exists(size_cache_file)

    # Règle auto-cache:
    # - si le fichier cache n'existe pas -> calculer les tailles automatiquement
    # - si le cache est incomplet pour les URLs à télécharger -> compléter automatiquement
    # - si refresh_sizes=True -> forcer le recalcul complet
    should_compute_sizes = (
        refresh_sizes
        or (not cache_file_exists)
        or (not cache_covers_all_urls)
    )

    if should_compute_sizes:
        if verbose:
            missing_count = len(urls_to_download) - len(cache_preview)
            eta_s = int(max(missing_count, 0) / max(rate_limit, 1))
            if not cache_file_exists:
                print(f"Cache tailles absent: calcul automatique des tailles distantes (~{eta_s}s)")
            elif refresh_sizes:
                print(f"Rafraîchissement forcé du cache des tailles (~{eta_s}s)")
            else:
                print(f"Cache tailles incomplet: calcul des {missing_count} tailles manquantes (~{eta_s}s)")

        remote_sizes = build_remote_size_map(
            urls_to_download,
            rate_limiter=rate_limiter,
            cache_file=size_cache_file,
            refresh=refresh_sizes,
            verbose=verbose,
        )
    else:
        remote_sizes = cache_preview
        if verbose:
            print("Cache tailles complet détecté: pas de recalcul des tailles distantes")
    
    # Calculer la taille totale
    total_size_bytes = sum(remote_sizes.get(url, 0) for url in urls_to_download)
    stats["total_size_bytes"] = total_size_bytes
    
    if verbose:
        total_size_gb = total_size_bytes / (1024 * 1024 * 1024)
        total_size_mb = total_size_bytes / (1024 * 1024)
        print(f"Taille totale à télécharger : {total_size_gb:.2f} Go ({total_size_mb:.0f} Mo)")
    
    # Mode volontairement simple et stable: un seul worker (compatibilité API conservée)
    if max_workers != 1 and verbose:
        print(f"Mode simplifié actif: max_workers={max_workers} ignoré, exécution en 1 worker")
    max_workers = 1

    # Barre de progression globale en fichiers
    cache_data = load_size_cache(size_cache_file)
    with tqdm(total=len(urls_to_download), unit='file', 
              desc="Fichiers", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
              disable=not verbose, mininterval=1.0) as pbar_files:
        for url in urls_to_download:
            result, size_bytes = check_and_download(
                url,
                output_folder,
                chunk_size,
                verify_integrity,
                remote_sizes.get(url),
                verbose_file=verbose,
            )
            if result == "EXIST":
                stats["exist"] += 1
                stats["exist_size_bytes"] += size_bytes
                pbar_files.update(1)
                pbar_files.set_description(f"Fichiers (+{size_bytes / (1024**2):.0f}MB)")
            elif result == "DONE":
                stats["done"] += 1
                stats["done_size_bytes"] += size_bytes
                if isinstance(size_bytes, int) and size_bytes > 0:
                    remote_sizes[url] = size_bytes
                    cache_data[url] = size_bytes
                pbar_files.update(1)
                pbar_files.set_description(f"Fichiers (+{size_bytes / (1024**2):.0f}MB)")
            elif result == "FAIL":
                stats["fail"] += 1
                if url not in cache_data:
                    cache_data[url] = 0
                pbar_files.update(1)
                pbar_files.set_description("Fichiers (❌ ÉCHEC)")

    # Persister les tailles connues/observées afin d'enrichir le cache au fil des runs
    save_size_cache(size_cache_file, cache_data)
    
    # Creation des racoucis vers les tuiles dans le dossier de la trajectoire
    if traj_tiles_dir is not None:
        link_tiles_to_trajectory(traj_tiles_dir, all_urls, Path(output_folder)) # /!\ output_folder doit être le pool de tuiles, pas le dossier de la trajectoire
    
    # Afficher les statistiques
    if verbose:
        print(f"\n=== RÉSUMÉ ===")
        print(f"Fichiers déjà présents : {stats['exist']} ({stats['exist_size_bytes'] / (1024**2):.0f} Mo)")
        print(f"Fichiers téléchargés : {stats['done']} ({stats['done_size_bytes'] / (1024**2):.0f} Mo)")
        print(f"Échecs : {stats['fail']}")
        print(f"Total : {(stats['exist_size_bytes'] + stats['done_size_bytes']) / (1024**2):.0f} Mo téléchargés/disponibles")
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Télécharge des tuiles LiDAR depuis une liste d'URLs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation :
    %(prog)s -i urls.txt -o lidar_data/
    %(prog)s -i urls_a.txt urls_b.txt -o lidar_data/
    %(prog)s --input-dir listes_urls/ -o lidar_data/
    %(prog)s -i urls.txt -o lidar_data/ --traj-tiles-dir dossier_trajet/tiles_laz
        """
    )
    
    parser.add_argument('-i', '--input', nargs='+', default=None,
                        help='Un ou plusieurs fichiers de listes d\'URLs (une URL par ligne)')
    parser.add_argument('--input-dir', default=None,
                        help='Dossier contenant plusieurs listes (glob: *.txt)')
    parser.add_argument('-o', '--output', required=True,
                        help='Dossier de destination pour les fichiers')
    parser.add_argument('--traj-tiles-dir', default=None,
                        help='Dossier de la trajectoire pour créer des liens symboliques (optionnel)')
    parser.add_argument('-w', '--workers', type=int, default=1,
                        help='Conservé pour compatibilité, mode simplifié forcé à 1')
    parser.add_argument('-c', '--chunk-size', default='10M',
                        help='Taille des chunks de téléchargement (ex: 10M, 5M) (défaut: 10M)')
    parser.add_argument('--verify', action='store_true',
                        help='Vérifier l\'intégrité des fichiers LAZ (plus lent)')
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='Mode silencieux (pas de messages)')
    parser.add_argument('--rate-limit', type=float, default=10,
                        help='Limite globale de requêtes API par seconde (défaut: 10)')
    parser.add_argument('--size-cache-file', default=None,
                        help='Fichier JSON pour cache des tailles distantes (défaut: <output>/remote_sizes_cache.json)')
    parser.add_argument('--refresh-sizes', action='store_true',
                        help='Ignore le cache local et recharge toutes les tailles distantes')
    parser.add_argument('--no-prefetch-sizes', action='store_true',
                        help='Démarre immédiatement sans requêtes HEAD en amont (utilise seulement le cache existant)')
    
    args = parser.parse_args()

    input_files = []
    if args.input:
        input_files.extend(args.input)
    if args.input_dir:
        input_dir = Path(args.input_dir).expanduser().resolve()
        if not input_dir.exists() or not input_dir.is_dir():
            raise FileNotFoundError(f"Dossier input-dir invalide: {input_dir}")
        input_files.extend(str(p) for p in sorted(input_dir.glob("*.txt")))

    if not input_files:
        raise ValueError("Aucune liste fournie. Utilise --input ou --input-dir.")

    # Déduplication en conservant l'ordre.
    dedup_input_files = list(dict.fromkeys(input_files))
    
    # Parser la taille des chunks
    chunk_size_str = args.chunk_size.upper()
    if chunk_size_str.endswith('M'):
        chunk_size = int(chunk_size_str[:-1]) * 1024 * 1024
    elif chunk_size_str.endswith('K'):
        chunk_size = int(chunk_size_str[:-1]) * 1024
    else:
        chunk_size = int(chunk_size_str)
    
    # Lancer le téléchargement
    try:
        aggregate = {
            "exist": 0,
            "done": 0,
            "fail": 0,
            "total_size_bytes": 0,
            "exist_size_bytes": 0,
            "done_size_bytes": 0,
        }

        for file_list in dedup_input_files:
            if not args.quiet:
                print(f"\n=== Liste: {file_list} ===")
            stats = download_tiles(
                traj_tiles_dir=args.traj_tiles_dir,
                file_list=file_list,
                output_folder=args.output,
                max_workers=args.workers,
                chunk_size=chunk_size,
                verify_integrity=args.verify,
                verbose=not args.quiet,
                rate_limit=args.rate_limit,
                size_cache_file=args.size_cache_file,
                refresh_sizes=args.refresh_sizes,
                prefetch_sizes=not args.no_prefetch_sizes,
            )
            for k in aggregate:
                aggregate[k] += stats.get(k, 0)

        if not args.quiet and len(dedup_input_files) > 1:
            print("\n=== RÉSUMÉ GLOBAL ===")
            print(f"Listes traitées : {len(dedup_input_files)}")
            print(f"Fichiers déjà présents : {aggregate['exist']}")
            print(f"Fichiers téléchargés : {aggregate['done']}")
            print(f"Échecs : {aggregate['fail']}")
            total_mb = (aggregate['exist_size_bytes'] + aggregate['done_size_bytes']) / (1024**2)
            print(f"Total disponible/téléchargé : {total_mb:.0f} Mo")
    except (FileNotFoundError, ValueError) as e:
        print(f"ERREUR : {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nInterruption par l'utilisateur.")
        sys.exit(1)
    except Exception as e:
        print(f"ERREUR inattendue : {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()