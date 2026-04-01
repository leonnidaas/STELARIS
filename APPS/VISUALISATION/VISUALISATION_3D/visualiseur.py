import laspy
import open3d as o3d
import numpy as np
import os
import time
from collections import OrderedDict
from scipy.spatial import cKDTree
import pandas as pd
import matplotlib.pyplot as plt
from termcolor import colored
from utils import get_traj_paths

class LidarVisualizer:
    def __init__(self, lidar_folder, df):
        """
        Initialise le visualiseur avec le dossier des dalles LiDAR et la trajectoire.
        """
        self.lidar_folder = lidar_folder
        self.df = df.copy()
        self.df['time_utc'] = pd.to_datetime(self.df['time_utc'])  
        self.point_size = 1.0    
        self.line_size = 10.0
        # Couleurs officielles IGN pour la classification
        self.default_colors = {
            2: [0.5, 0.3, 0.1],   # Sol (Brun)
            6: [1, 0, 0],         # Bâtiments (Rouge)
            17:[1, 0.5, 0],      # Tablier de Ponts (Orange)
            3: [0, 1, 0],         # Végétation basse (Vert)
            4: [0, 0.7, 0],       # Végétation moyenne
            5: [0, 0.4, 0],       # Végétation haute
            0: [0.5, 0.5, 0.5],   # Non classé (Gris)
            1: [0.5, 0.5, 0.5],   # Non classé (Gris)
            9: [0, 0.5, 1],       # eau bleu
            64:[1, 0, 1],         # poteaux (ROSES)
            67:[1, 1, 0],         # autre batis (Jaune)
        }
        self.traj3d_color = [0, 0, 0]  # noir pour la trajectoire

        # États pour l'interactivité
        self.current_color_mode = 'class'
        self.pts_cache = None
        self.cls_cache = None
        self.tile_cache = OrderedDict()
        self.tile_cache_limit = 8
        self.flythrough_running = False
        self.last_flythrough_trigger = 0.0
        self.current_color_mode = 'class'
        self.stop_flythrough_requested = False

        # Configuration viewer/fly-through factorisée.
        self.camera_near_clip = 0.001
        self.camera_far_clip = 100000.0
        self.flythrough_frame_delay = 0.05
        self.camera_anchor_mode = 'offset'
        self.camera_view_mode = 'forward'
        self.camera_fov_deg = 90
        self.camera_pitch_deg = -20.0
        self.step_size = 0.04
        self.lookahead_points = 12
        self.lookahead_distance = 5.0
        self.pause_focus_distance = 0.001
        self.camera_height = 2.2
        self.camera_radius = 150.0
        self.refresh_distance = 50.0
        self.flythrough_show_time = True
        self.flythrough_time_refresh_s = 0.10

        # État de scène actif pour alléger les signatures internes.
        self.active_pcd = None
        self.active_scene_offset = None
        self.active_dynamic_loader = None
        self.active_traj_times = None
        self.active_progress_callback = None

    def _normalize_class_factors(self, class_decimation_factors):
        """Normalise la config de décimation par classe vers {int: int>=1}."""
        if not class_decimation_factors:
            return {}

        out = {}
        for k, v in class_decimation_factors.items():
            try:
                cls_id = int(k)
                fac = max(1, int(v))
                out[cls_id] = fac
            except Exception:
                continue
        return out

    def _apply_adaptive_decimation(
        self,
        points,
        classes,
        dists_to_traj,
        base_factor=1,
        adaptive_by_distance=True,
        near_distance=15.0,
        far_distance=50.0,
        mid_factor=2,
        far_factor=4,
        class_decimation_factors=None,
    ):
        """Décimation déterministe par distance à la trajectoire + surcharge optionnelle par classe."""
        if points is None or len(points) == 0:
            return points, classes

        n = len(points)
        base_factor = max(1, int(base_factor))
        factors = np.full(n, base_factor, dtype=np.int32)

        if adaptive_by_distance and dists_to_traj is not None and len(dists_to_traj) == n:
            d = np.asarray(dists_to_traj, dtype=float)
            near_distance = float(max(0.0, near_distance))
            far_distance = float(max(near_distance, far_distance))
            mid_factor = max(1, int(mid_factor))
            far_factor = max(mid_factor, int(far_factor))

            factors = np.where(d > near_distance, base_factor * mid_factor, factors)
            factors = np.where(d > far_distance, base_factor * far_factor, factors)

        class_factors = self._normalize_class_factors(class_decimation_factors)
        if class_factors:
            cls = np.asarray(classes, dtype=np.int32)
            for cls_id, fac in class_factors.items():
                factors = np.where(cls == cls_id, np.maximum(factors, fac), factors)

        idx = np.arange(n, dtype=np.int64)
        keep = (idx % np.maximum(factors, 1)) == 0
        return points[keep], classes[keep]

    def configure_viewer(self, **overrides):
        """Met à jour les paramètres persistants du visualiseur."""
        allowed_keys = {
            'camera_near_clip',
            'camera_far_clip',
            'flythrough_frame_delay',
            'camera_anchor_mode',
            'camera_view_mode',
            'camera_fov_deg',
            'camera_pitch_deg',
            'step_size',
            'lookahead_points',
            'lookahead_distance',
            'pause_focus_distance',
            'camera_height',
            'camera_radius',
            'refresh_distance',
            'flythrough_show_time',
            'flythrough_time_refresh_s',
            'point_size',
        }
        unknown_keys = sorted(set(overrides) - allowed_keys)
        if unknown_keys:
            raise ValueError(f"Paramètres viewer inconnus: {', '.join(unknown_keys)}")

        for key, value in overrides.items():
            if value is not None:
                setattr(self, key, value)

    def _set_active_scene(self, pcd=None, scene_offset=None, dynamic_loader=None):
        """Stocke l'état de scène actif utilisé par le fly-through."""
        self.active_pcd = pcd
        self.active_scene_offset = scene_offset
        self.active_dynamic_loader = dynamic_loader

    def _set_camera_pose(self, ctr, eye, lookat, up_vec):
        """Applique une pose caméra robuste à partir de eye/lookat/up."""
        z = eye - lookat
        z /= np.linalg.norm(z) + 1e-12

        upn = up_vec / (np.linalg.norm(up_vec) + 1e-12)
        if abs(np.dot(upn, z)) > 0.98:
            upn = np.array([0.0, 1.0, 0.0], dtype=float)
            if abs(np.dot(upn, z)) > 0.98:
                upn = np.array([1.0, 0.0, 0.0], dtype=float)

        x = np.cross(upn, z)
        x /= np.linalg.norm(x) + 1e-12
        y = np.cross(z, x)

        extrinsic = np.eye(4, dtype=float)
        extrinsic[0, :3] = x
        extrinsic[1, :3] = y
        extrinsic[2, :3] = z
        extrinsic[0, 3] = -np.dot(x, eye)
        extrinsic[1, 3] = -np.dot(y, eye)
        extrinsic[2, 3] = -np.dot(z, eye)

        params = ctr.convert_to_pinhole_camera_parameters()
        params.extrinsic = extrinsic
        try:
            ctr.convert_from_pinhole_camera_parameters(params, allow_arbitrary=True)
        except TypeError:
            ctr.convert_from_pinhole_camera_parameters(params)

    def _focus_lookat_near_current_camera(self, ctr):
        """Rapproche le lookat de la caméra courante sans changer brutalement la position."""
        try:
            params = ctr.convert_to_pinhole_camera_parameters()
        except Exception:
            return

        extrinsic = params.extrinsic
        rot = extrinsic[:3, :3]
        trans = extrinsic[:3, 3]

        eye = -rot.T @ trans
        front = None
        try:
            if hasattr(ctr, 'get_front'):
                front = np.asarray(ctr.get_front(), dtype=float)
        except Exception:
            front = None

        if front is None or np.linalg.norm(front) < 1e-9:
            back = rot[2, :]
            front = -back

        front_norm = np.linalg.norm(front)
        if front_norm < 1e-9:
            front = np.array([1.0, 0.0, 0.0], dtype=float)
        else:
            front = front / front_norm

        up = None
        try:
            if hasattr(ctr, 'get_up'):
                up = np.asarray(ctr.get_up(), dtype=float)
        except Exception:
            up = None

        if up is None or np.linalg.norm(up) < 1e-9:
            up = rot[1, :]
        up = up / (np.linalg.norm(up) + 1e-12)

        # Garde un focus très proche, mais compatible avec le near clip.
        focus_dist = float(max(0.0005, self.pause_focus_distance, self.camera_near_clip * 1.2))
        lookat = eye + front * focus_dist

        current_zoom = None
        try:
            if hasattr(ctr, 'get_zoom'):
                current_zoom = float(ctr.get_zoom())
        except Exception:
            current_zoom = None

        # 1) Pose exacte pour conserver eye/front/up stables.
        self._set_camera_pose(ctr, eye, lookat, up)

        # 2) Pivot explicite pour l'orbite autour du lookat local.
        try:
            if hasattr(ctr, 'set_lookat'):
                ctr.set_lookat(lookat)
            # Preserve user zoom level instead of forcing a reset.
            if current_zoom is not None and hasattr(ctr, 'set_zoom'):
                ctr.set_zoom(float(np.clip(current_zoom, 0.0005, 5.0)))
        except Exception:
            pass

    def _get_traj3d(self, df_seg, gnss_offset=(0, 0, 4.137)):
        """Extrait la trajectoire et applique l'offset antenne en metres."""
        # TODO : l'offset de l'antenne depend du train et devrait être paramétrable, mais pour l'instant on applique un offset vertical de 4.137m qui correspond à la hauteur de l'antenne sur le toit du train par rapport au point de référence au sol.
        traj = df_seg[['x_gt', 'y_gt', 'z_gt_ign69']].values.copy()
          # Inversion de l'offset vertical pour correspondre au système de coordonnées (z vers le haut)
        traj[:, 2] -= gnss_offset[2]
        return traj

    def _interp_time_at_position(self, pos: float, max_idx: int):
        """Interpole le timestamp UTC pour une position fractionnaire sur la trajectoire."""
        times = self.active_traj_times
        if times is None or len(times) == 0:
            return None

        if len(times) == 1 or max_idx <= 0:
            return pd.to_datetime(times[0])

        if pos <= 0.0:
            return pd.to_datetime(times[0])
        if pos >= float(max_idx):
            return pd.to_datetime(times[max_idx])

        i0 = int(np.floor(pos))
        i1 = min(i0 + 1, max_idx)
        t = pos - i0

        t0 = pd.to_datetime(times[i0])
        t1 = pd.to_datetime(times[i1])
        dt_ns = int((t1.value - t0.value) * t)
        return pd.to_datetime(t0.value + dt_ns)

    def _get_segment(self, start=None, end=None, mode='percent'):
        """Découpe la trajectoire par pourcentage ou par temps UTC."""
        temp_df = self.df.sort_values('time_utc').reset_index(drop=True)
        if mode == 'percent':
            s_idx, e_idx = int(len(temp_df) * (start or 0)), int(len(temp_df) * (end or 1))
            return temp_df.iloc[s_idx:e_idx]
        elif mode == 'time':
            start_dt = pd.to_datetime(start)
            end_dt = pd.to_datetime(end)

            # Si une date est fournie, on filtre en datetime complet (plus fiable).
            has_explicit_date = any(ch in str(start) for ch in ['-', '/']) or any(ch in str(end) for ch in ['-', '/'])
            if has_explicit_date:
                mask = (temp_df['time_utc'] >= start_dt) & (temp_df['time_utc'] <= end_dt)
                res = temp_df[mask]
            else:
                # Sinon on filtre par heure et on conserve le plus long bloc contigu
                # pour éviter de mélanger plusieurs passages sur des jours différents.
                t_start, t_end = start_dt.time(), end_dt.time()
                mask = (temp_df['time_utc'].dt.time >= t_start) & (temp_df['time_utc'].dt.time <= t_end)
                res = temp_df[mask]

                if len(res) > 0:
                    # Découpe par rupture temporelle pour isoler un seul passage.
                    dt = res['time_utc'].diff().dt.total_seconds().fillna(0)
                    block_id = (dt > 10.0).cumsum()
                    counts = block_id.value_counts()
                    best_id = counts.index[0]
                    res = res[block_id == best_id]

            if len(res) == 0:
                raise ValueError(f"Aucun point trouvé entre {start} et {end}")
            return res
        
    def _get_color(self, pts, classes, mode='class'):
        """Génère les couleurs RGB selon la classe ou l'altitude."""
        if mode == 'class':
            c_map = np.full((len(classes), 3), [0.5, 0.5, 0.5])
            for cid, col in self.default_colors.items():
                c_map[classes == cid] = col
            return c_map
        elif mode == 'altitude':
            z = pts[:, 2]
            z_n = (z - z.min()) / (z.max() - z.min() + 1e-6)
            return plt.get_cmap('terrain')(z_n)[:, :3]
        return np.full((len(classes), 3), [0.5, 0.5, 0.5])
        
    def _sampling(self, las, factor=1):
        """Sous-échantillonnage pour la performance."""
        p = np.vstack((las.x[::factor], las.y[::factor], las.z[::factor])).T
        cl = las.classification[::factor]
        return p, cl

    def _get_candidate_tiles(self, traj_3d, margin):
        """Retourne les dalles LiDAR qui intersectent l'emprise XY de la trajectoire."""
        min_x, min_y = traj_3d[:, :2].min(axis=0) - margin
        max_x, max_y = traj_3d[:, :2].max(axis=0) + margin

        candidates = []
        files = [f for f in os.listdir(self.lidar_folder) if f.endswith('.laz')]
        for file in files:
            path = os.path.join(self.lidar_folder, file)
            with laspy.open(path) as fh:
                h = fh.header
                if h.min[0] > max_x or h.max[0] < min_x or h.min[1] > max_y or h.max[1] < min_y:
                    continue
                candidates.append(
                    {
                        'path': path,
                        'bbox': (h.min[0], h.min[1], h.max[0], h.max[1]),
                    }
                )
        return candidates

    def _get_tile_points(self, tile_path, factor=1, hide_wires=False):
        """Charge une dalle et la garde en cache pour éviter les relectures disque."""
        cache_key = (tile_path, factor, hide_wires)
        if cache_key in self.tile_cache:
            self.tile_cache.move_to_end(cache_key)
            return self.tile_cache[cache_key]

        with laspy.open(tile_path) as fh:
            las = fh.read()
            p, cl = self._sampling(las, factor=factor)

        if hide_wires:
            mask = ~np.isin(cl, [0, 1])
            p, cl = p[mask], cl[mask]

        self.tile_cache[cache_key] = (p, cl)
        self.tile_cache.move_to_end(cache_key)
        while len(self.tile_cache) > self.tile_cache_limit:
            self.tile_cache.popitem(last=False)
        return p, cl

    def _load_points_near_position(self, position_xy, candidate_tiles, radius, factor=1, hide_wires=False):
        """Charge uniquement les points à proximité immédiate de la caméra."""
        radius_sq = radius * radius
        x_min = position_xy[0] - radius
        x_max = position_xy[0] + radius
        y_min = position_xy[1] - radius
        y_max = position_xy[1] + radius

        all_points, all_classes = [], []
        for tile in candidate_tiles:
            min_x, min_y, max_x, max_y = tile['bbox']
            if min_x > x_max or max_x < x_min or min_y > y_max or max_y < y_min:
                continue

            p, cl = self._get_tile_points(tile['path'], factor=factor, hide_wires=hide_wires)
            d2 = (p[:, 0] - position_xy[0]) ** 2 + (p[:, 1] - position_xy[1]) ** 2
            mask = d2 <= radius_sq
            if np.any(mask):
                all_points.append(p[mask])
                all_classes.append(cl[mask])

        if not all_points:
            return np.empty((0, 3)), np.empty((0,), dtype=int)

        return np.concatenate(all_points), np.concatenate(all_classes)
        
    def _print_help(self):
        """Affiche les commandes clavier dans le terminal."""
        os.system('cls' if os.name == 'nt' else 'clear')
        print("="*60)
        print("         STELARIS LiDAR VISUALIZER - COMMANDES")
        print("="*60)
        print("\n[NAVIGATION]")
        print("  - Z (W) / S : Avancer / Reculer")
        print("  - Q (A) / D : Gauche / Droite")
        print("  - R / F     : Monter / Descendre")
        print("  - M         : Reset la vue")
        print("  - L         : Arrêt du fly-through")
        print("\n[MODES]")
        print("  - C         : Switch Couleur (Classe / Altitude)")
        print("  - T         : Lancer un fly-through le long de la trajectoire")
        print("  - 5 / 6     : Near clip proche / loin")
        print("  - , / .     : Fly-through plus rapide / plus lent")
        print("  - o / i       : Augmenter / Diminuer la taille des points")
        print("  - V         : Toggle visée (avant / ciel)")
        print("  - B         : Toggle ancrage (offset / antenne)")
        print("  - H         : Ré-afficher ce guide")
        print("  - Echap     : Quitter")
        print("="*60 + "\n")

    def _play_camera_along_traj(self, vis, ctr, traj_local):
        """Anime la caméra en suivant la trajectoire locale (déjà centrée).
        vis : instance de Visualizer
        ctr : instance de ViewControl pour manipuler la caméra
        traj_local : trajectoire centrée à suivre (Nx3)
        """
        if traj_local is None or len(traj_local) < 2:
            print("Fly-through indisponible : trajectoire trop courte.")
            return
        if self.flythrough_running:
            return

        self.stop_flythrough_requested = False
        self.flythrough_running = True

        try:
            camera_up = np.array([0.0, 0.0, -1.0], dtype=float)
            max_idx = len(traj_local) - 1
            last_loaded_position = None
            last_time_print = 0.0
            print("Fly-through en cours... (appuyez sur Echap pour fermer la vue)")

            step_size = float(self.step_size)
            if step_size <= 0:
                step_size = 1.0

            # Positions fractionnaires pour permettre un déplacement plus fin que 1 point.
            positions = list(np.arange(0.0, float(max_idx) + 1e-9, step_size))
            if positions[-1] < float(max_idx):
                positions.append(float(max_idx))

            def _interp_point(pos):
                if pos <= 0.0:
                    return traj_local[0]
                if pos >= float(max_idx):
                    return traj_local[max_idx]
                i0 = int(np.floor(pos))
                i1 = min(i0 + 1, max_idx)
                t = pos - i0
                return traj_local[i0] * (1.0 - t) + traj_local[i1] * t

            def _set_camera_pose(eye, lookat, up_vec):
                self._set_camera_pose(ctr, eye, lookat, up_vec)

            for pos in positions:
                if self.stop_flythrough_requested:
                    break

                target_pos = min(pos + float(self.lookahead_points), float(max_idx))
                p = _interp_point(pos)
                target = _interp_point(target_pos)
                front = target - p
                norm = np.linalg.norm(front)
                if norm < 1e-6:
                    continue
                front /= norm

                # Pitch: inclinaison verticale de la vue (negatif = regarde un peu vers le sol).
                pitch_rad = np.deg2rad(self.camera_pitch_deg)
                front = front * np.cos(pitch_rad) + camera_up * np.sin(pitch_rad)
                front /= np.linalg.norm(front) + 1e-12

                if self.active_dynamic_loader is not None and self.active_pcd is not None and self.active_scene_offset is not None:
                    if last_loaded_position is None or np.linalg.norm(p - last_loaded_position) >= self.refresh_distance:
                        pts, cls = self.active_dynamic_loader(p + self.active_scene_offset)
                        self.pts_cache = pts
                        self.cls_cache = cls
                        self.active_pcd.points = o3d.utility.Vector3dVector(pts - self.active_scene_offset)
                        self.active_pcd.colors = o3d.utility.Vector3dVector(self._get_color(pts, cls, self.current_color_mode))
                        vis.update_geometry(self.active_pcd)
                        last_loaded_position = p.copy()

                if self.camera_anchor_mode == 'antenna':
                    eye = p.copy()
                else:
                    eye = p + camera_up * self.camera_height

                if self.camera_view_mode == 'sky':
                    # sky_dir = camera_up * 0.85 + front * 0.35
                    sky_dir = camera_up
                    sky_dir /= np.linalg.norm(sky_dir) + 1e-12
                    lookat = eye + sky_dir * self.lookahead_distance
                    pose_up = front
                else:
                    lookat = eye - front * self.lookahead_distance
                    pose_up = camera_up

                _set_camera_pose(eye, lookat, pose_up)

                # Affichage léger de l'heure UTC dynamique (throttlé pour limiter l'I/O terminal).
                if self.flythrough_show_time:
                    now = time.time()
                    if (now - last_time_print) >= float(self.flythrough_time_refresh_s):
                        t_utc = self._interp_time_at_position(pos, max_idx)
                        if t_utc is not None:
                            time_label = t_utc.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                            print(f"\rUTC: {time_label} | progression: {pos:.2f}/{max_idx}", end="", flush=True)
                            if self.active_progress_callback is not None:
                                try:
                                    self.active_progress_callback(
                                        {
                                            "time_utc": time_label,
                                            "progress": float(pos / max(1, max_idx)),
                                            "position": float(pos),
                                            "max_index": int(max_idx),
                                        }
                                    )
                                except Exception:
                                    pass
                            last_time_print = now

                vis.poll_events()
                vis.update_renderer()
                time.sleep(self.flythrough_frame_delay)
        finally:
            self.stop_flythrough_requested = False
            self.flythrough_running = False
            if self.flythrough_show_time:
                print()

    def _render(self, points, classes, traj_3d=None, window_name="LiDAR", auto_flythrough=False, dynamic_loader=None):
        """Moteur de rendu Open3D avec callbacks clavier.
        points : Nx3 numpy array des points à afficher
        classes : N array des classes pour le coloriage
        traj_3d : trajectoire 3D à afficher (optionnel)
        window_name : titre de la fenêtre
        auto_flythrough : lancer automatiquement le fly-through à l'ouverture
        dynamic_loader : fonction de chargement dynamique des points autour de la caméra (optionnel)
        refresh_distance : distance à parcourir avant de rafraîchir les points dynamiquement"""
        if len(points) == 0 and traj_3d is None:
            print("Aucun point à afficher.")
            return
        
        self._print_help()
        self.pts_cache = points
        self.cls_cache = classes
        if traj_3d is not None:
            offset = np.mean(traj_3d, axis=0)
        elif len(points) > 0:
            offset = np.mean(points, axis=0)
        else:
            offset = np.zeros(3)

        pcd = o3d.geometry.PointCloud()
        if len(points) > 0:
            pcd.points = o3d.utility.Vector3dVector(points - offset)
            pcd.colors = o3d.utility.Vector3dVector(self._get_color(points, classes, self.current_color_mode))
        else:
            pcd.points = o3d.utility.Vector3dVector(np.empty((0, 3)))
            pcd.colors = o3d.utility.Vector3dVector(np.empty((0, 3)))

        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window(window_name=window_name, width=1280, height=720)
        opt = vis.get_render_option()
        opt.point_size = self.point_size
        # La bbox de reference doit venir du nuage de points, pas de la trajectoire complete.
        try:
            vis.add_geometry(pcd, reset_bounding_box=True)
        except TypeError:
            vis.add_geometry(pcd)

        if traj_3d is not None:
            ls = o3d.geometry.LineSet()
            ls.points = o3d.utility.Vector3dVector(traj_3d - offset)
            ls.lines = o3d.utility.Vector2iVector([[i, i+1] for i in range(len(traj_3d)-1)])
            ls.paint_uniform_color(self.traj3d_color)
            # on change la taille de la ligne de trajectoire
            opt.line_width = self.line_size
            # Ne pas recalculer la bbox sur la trajectoire longue: sinon zoom-in molette limite.
            try:
                vis.add_geometry(ls, reset_bounding_box=False)
            except TypeError:
                vis.add_geometry(ls)

        traj_local = (traj_3d - offset) if traj_3d is not None else None

        ctr = vis.get_view_control()
        self._set_active_scene(pcd=pcd, scene_offset=offset, dynamic_loader=dynamic_loader)
        current_fov = ctr.get_field_of_view()
        ctr.change_field_of_view(float(self.camera_fov_deg) - current_fov)
        self.flythrough_frame_delay = float(np.clip(self.flythrough_frame_delay, 0.01, 1.0))

        
        def set_near_clip(target_near):
            self.camera_near_clip = float(max(0.0001, target_near))
            try:
                if hasattr(ctr, "set_constant_z_near"):
                    ctr.set_constant_z_near(self.camera_near_clip)
                print(f"Near clip: {self.camera_near_clip:.3f} m")
            except Exception:
                pass

        def set_far_clip(target_far):
            self.camera_far_clip = float(max(10.0, target_far))
            try:
                if hasattr(ctr, "set_constant_z_far"):
                    ctr.set_constant_z_far(self.camera_far_clip)
            except Exception:
                pass

        def set_frame_delay(target_delay):
            self.flythrough_frame_delay = float(np.clip(target_delay, 0.0005, 1.0))
            fps = 1.0 / self.flythrough_frame_delay
            print(f"Frame delay: {self.flythrough_frame_delay:.4f} s ({fps:.1f} FPS)")
        
        def change_color_mode(vis):
            self.current_color_mode = 'altitude' if self.current_color_mode == 'class' else 'class'
            if self.pts_cache is not None and self.cls_cache is not None and len(self.pts_cache) > 0:
                pcd.colors = o3d.utility.Vector3dVector(
                    self._get_color(self.pts_cache, self.cls_cache, self.current_color_mode)
                )
            vis.update_geometry(pcd)
            print(f"Mode couleur : {self.current_color_mode}")
            return True

        def start_flythrough(vis):
            now = time.time()
            if self.flythrough_running or (now - self.last_flythrough_trigger) < 0.5:
                return True

            self.last_flythrough_trigger = now
            self._play_camera_along_traj(vis, ctr, traj_local)
            return True
        
        def decrease_near_clip(vis):
            set_near_clip(self.camera_near_clip - 0.05)
            return True

        def increase_near_clip(vis):
            set_near_clip(self.camera_near_clip + 0.05)
            return True

        def faster_flythrough(vis):
            set_frame_delay(self.flythrough_frame_delay - 0.005)
            return True

        def slower_flythrough(vis):
            set_frame_delay(self.flythrough_frame_delay + 0.005)
            return True

        def toggle_camera_view_mode(vis):
            self.camera_view_mode = 'sky' if self.camera_view_mode == 'forward' else 'forward'
            print(f"Mode visée: {self.camera_view_mode}")
            return True

        def toggle_camera_anchor_mode(vis):
            self.camera_anchor_mode = 'antenna' if self.camera_anchor_mode == 'offset' else 'offset'
            print(f"Ancrage caméra: {self.camera_anchor_mode}")
            return True

        def stop_flythrough(vis):
            self.stop_flythrough_requested = True
            self._focus_lookat_near_current_camera(ctr)
            vis.update_renderer()
            return True
        def reset_view_point(vis):
            self.stop_flythrough_requested = True
            vis.reset_view_point(True)
            return True
        def exit_viewer(vis):
            self.stop_flythrough_requested = True
            vis.close()
            print("Visualisation fermée.")
            return True
        
        def print_view_status(vis):
            ctr = vis.get_view_control()
            # On récupère les paramètres de caméra
            cam_param = ctr.convert_to_pinhole_camera_parameters()
            print("--- État de la Caméra ---")
            print(f"Matrice Extrinsèque :\n{cam_param.extrinsic}")
            print(f'lookat : {cam_param.extrinsic[:3, 3] - cam_param.extrinsic[2, :3] * self.lookahead_distance}')
            return False
        def increase_point_size(vis):
            opt = vis.get_render_option()
            new_size = opt.point_size + 0.5
            if new_size > 10.0:
                new_size = 1.0
            opt.point_size = new_size
            print(f"Taille des points : {opt.point_size:.1f}")
            return True
        def decrease_point_size(vis):
            opt = vis.get_render_option()
            new_size = opt.point_size - 0.5
            if new_size < 0.5:
                new_size = 10.0
            opt.point_size = new_size
            print(f"Taille des points : {opt.point_size:.1f}")
            return True
        # Mapping des touches
        step = 7.5
        vis.register_key_callback(256, exit_viewer)
        vis.register_key_callback(ord("C"), change_color_mode)
        vis.register_key_callback(ord("H"), lambda v: self._print_help())
        vis.register_key_callback(ord("T"), start_flythrough)
        vis.register_key_callback(ord("5"), decrease_near_clip)
        vis.register_key_callback(ord("6"), increase_near_clip)
        vis.register_key_callback(ord("M"), faster_flythrough)
        vis.register_key_callback(ord(","), slower_flythrough)
        vis.register_key_callback(ord("V"), toggle_camera_view_mode)
        vis.register_key_callback(ord("B"), toggle_camera_anchor_mode)
        vis.register_key_callback(ord("R"), lambda v: ctr.camera_local_translate(0, 0, step))
        vis.register_key_callback(ord("F"), lambda v: ctr.camera_local_translate(0, 0, -step))
        vis.register_key_callback(ord("S"), lambda v: ctr.camera_local_translate(-step, 0, 0))
        vis.register_key_callback(ord("W"), lambda v: ctr.camera_local_translate(step, 0, 0)) # Z sur clavier AZERTY
        vis.register_key_callback(ord("Z"), lambda v: ctr.camera_local_translate(step, 0, 0)) # Avancer AZERTY
        vis.register_key_callback(ord("D"), lambda v: ctr.camera_local_translate(0, step, 0))
        vis.register_key_callback(ord("A"), lambda v: ctr.camera_local_translate(0, -step, 0)) # Q sur clavier AZERTY
        vis.register_key_callback(ord("Q"), lambda v: ctr.camera_local_translate(0, -step, 0)) # Gauche AZERTY
        vis.register_key_callback(ord(";"), reset_view_point) # M sur clavier AZERTY
        vis.register_key_callback(ord("L"), stop_flythrough)
        vis.register_key_callback(ord("P"), print_view_status)
        vis.register_key_callback(ord("O"), increase_point_size)
        vis.register_key_callback(ord("I"), decrease_point_size)


        set_near_clip(self.camera_near_clip)
        set_far_clip(self.camera_far_clip)
        set_frame_delay(self.flythrough_frame_delay)


        if auto_flythrough and traj_local is not None:
            self._play_camera_along_traj(vis, ctr, traj_local)
        
        vis.run()
        vis.destroy_window()
        

    def show_corridor(
        self,
        start=0,
        end=1,
        mode='percent',
        width=40,
        factor=1,
        color_mode='class',
        hide_wires=False,
        gnss_offset=(0, 0, 4.137),
        adaptive_decimation=False,
        near_distance=15.0,
        far_distance=50.0,
        mid_factor=2,
        far_factor=4,
        class_decimation_factors=None,
        **viewer_options,
    ):
        """Affiche les points LiDAR dans un rayon de 'width' mètres autour du train."""
        self.current_color_mode = color_mode
        df_seg = self._get_segment(start, end, mode)
        traj_3d = self._get_traj3d(df_seg, gnss_offset=gnss_offset)
        traj_tree = cKDTree(traj_3d[:, :2]) # Indexation spatiale pour filtrage rapide
        all_p, all_cl = [], []
        candidate_tiles = self._get_candidate_tiles(traj_3d, width if width is not None else 100)

        for tile in candidate_tiles:
            p, cl = self._get_tile_points(tile['path'], factor=factor, hide_wires=hide_wires)
            dists, _ = traj_tree.query(p[:, :2], k=1)
            mask = dists <= width if width is not None else np.ones(len(p), dtype=bool)
            p, cl = p[mask], cl[mask]
            d_local = dists[mask]

            if adaptive_decimation and len(p) > 0:
                p, cl = self._apply_adaptive_decimation(
                    p,
                    cl,
                    d_local,
                    base_factor=1,
                    adaptive_by_distance=True,
                    near_distance=near_distance,
                    far_distance=far_distance,
                    mid_factor=mid_factor,
                    far_factor=far_factor,
                    class_decimation_factors=class_decimation_factors,
                )

            if len(p) > 0:
                all_p.append(p)
                all_cl.append(cl)
        
        if all_p:
            pts, cls = np.concatenate(all_p), np.concatenate(all_cl)
            self._render(pts, cls, traj_3d=traj_3d, window_name=f"Corridor {width}m")

    def show_corridor_flythrough(
        self,
        start=0,
        end=1,
        mode='percent',
        width=40,
        factor=1,
        color_mode='class',
        hide_wires=False,
        gnss_offset=(0, 0, 4.137),
        progress_callback=None,
        adaptive_decimation=False,
        near_distance=15.0,
        far_distance=50.0,
        mid_factor=2,
        far_factor=4,
        class_decimation_factors=None,
        **viewer_options,
    ):
        """Fly-through optimisé: ne charge que les points proches de la caméra."""
        # Compatibilite defensive: si progress_callback arrive par kwargs, on l'extrait.
        if progress_callback is None and "progress_callback" in viewer_options:
            progress_callback = viewer_options.pop("progress_callback")
        else:
            viewer_options.pop("progress_callback", None)

        self.configure_viewer(**viewer_options)
        self.current_color_mode = color_mode
        df_seg = self._get_segment(start, end, mode)
        traj_3d = self._get_traj3d(df_seg, gnss_offset=gnss_offset)
        traj_tree = cKDTree(traj_3d[:, :2])
        self.active_traj_times = df_seg['time_utc'].to_numpy()
        self.active_progress_callback = progress_callback
        candidate_tiles = self._get_candidate_tiles(traj_3d, max(width if width is not None else 100, self.camera_radius))

        def dynamic_loader(camera_position):
            pts, cls = self._load_points_near_position(
                camera_position[:2],
                candidate_tiles,
                radius=self.camera_radius,
                factor=factor,
                hide_wires=hide_wires,
            )

            if adaptive_decimation and len(pts) > 0:
                dists, _ = traj_tree.query(pts[:, :2], k=1)
                pts, cls = self._apply_adaptive_decimation(
                    pts,
                    cls,
                    dists,
                    base_factor=1,
                    adaptive_by_distance=True,
                    near_distance=near_distance,
                    far_distance=far_distance,
                    mid_factor=mid_factor,
                    far_factor=far_factor,
                    class_decimation_factors=class_decimation_factors,
                )

            return pts, cls

        initial_points, initial_classes = dynamic_loader(traj_3d[0])
        self._render(
            initial_points,
            initial_classes,
            traj_3d=traj_3d,
            window_name=f"Corridor Fly-Through {self.camera_radius}m",
            auto_flythrough=True,
            dynamic_loader=dynamic_loader,
        )
        self.active_progress_callback = None


if __name__ == "__main__":
    traj_id = "BORDEAUX_COUTRAS"
    config = get_traj_paths(traj_id)
    CSV_FILE = config['sync_csv']
    LIDAR_DIR = config['lidar_tiles']
    gnss_offset = config['gnss_offset']

    if os.path.exists(CSV_FILE) and os.path.exists(LIDAR_DIR):
        matched_df = pd.read_csv(CSV_FILE)
        viz = LidarVisualizer(LIDAR_DIR, matched_df)
        
        viz.show_corridor_flythrough(
            start=0, 
            end=0.1,
            mode='percent', 
            width=50, 
            factor=1, 
            hide_wires=False,
            gnss_offset=gnss_offset,
        )
    
    else:
        print(f"Erreur : Vérifiez la présence de {CSV_FILE} et du dossier {LIDAR_DIR}")