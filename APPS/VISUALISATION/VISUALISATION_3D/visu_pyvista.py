"""
visualiseur_pyvista.py
──────────────────────────────────────────────────────────────────────────────
Remplacement 1-pour-1 de visualiseur.py (Open3D) par PyVista.
Ajout principal : l'heure UTC est affichée directement dans la fenêtre 3D,
mise à jour en temps réel pendant le fly-through.

Dépendances :
    pip install pyvista laspy[lazrs] scipy pandas numpy

Usage identique à l'original :
    viz = LidarVisualizer(LIDAR_DIR, matched_df)
    viz.show_corridor_flythrough(start=0, end=1, mode='percent', width=50)
──────────────────────────────────────────────────────────────────────────────
"""

import os
import time
from collections import OrderedDict

import laspy
import numpy as np
import pandas as pd
import pyvista as pv
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

try:
    from utils import get_traj_paths
except ImportError:
    pass


# ── Couleurs IGN par classe (RGB 0-255 pour PyVista) ──────────────────────────

CLS_COLORS_RGB = {
    0:  (128, 128, 128),   # Non classé        → Gris
    1:  (128, 128, 128),   # Jamais classé     → Gris
    2:  (128,  77,  25),   # Sol               → Brun
    3:  (  0, 255,   0),   # Végétation basse  → Vert vif
    4:  (  0, 179,   0),   # Végétation moy.   → Vert moyen
    5:  (  0, 102,   0),   # Végétation haute  → Vert foncé
    6:  (255,   0,   0),   # Bâtiment          → Rouge
    17: (255, 128,   0),   # Pont / Structure  → Orange
}
DEFAULT_COLOR_RGB = (128, 128, 128)


class LidarVisualizer:
    """
    Visualiseur LiDAR PyVista — API identique à la version Open3D.

    Nouveautés vs Open3D :
      • L'heure UTC est affichée en overlay dans la fenêtre 3D (coin sup. gauche).
      • Pas de limitation sur le near-clip natif.
      • Rendu hardware accéléré via VTK/PyVista.
    """

    def __init__(self, lidar_folder, df):
        self.lidar_folder = lidar_folder
        self.df = df.copy()
        self.df["time_utc"] = pd.to_datetime(self.df["time_utc"])

        # ── Cache et état ───────────────────────────────────────────────────
        self.tile_cache          = OrderedDict()
        self.tile_cache_limit    = 8
        self.pts_cache           = None
        self.cls_cache           = None
        self.active_traj_times   = None
        self.active_progress_callback = None

        # ── Paramètres caméra / fly-through ────────────────────────────────
        self.current_color_mode      = "class"
        self.flythrough_running      = False
        self.stop_flythrough_requested = False
        self.last_flythrough_trigger = 0.0

        self.camera_near_clip        = 0.05
        self.flythrough_frame_delay  = 0.05
        self.camera_anchor_mode      = "offset"   # "offset" | "antenna"
        self.camera_view_mode        = "forward"  # "forward" | "sky"
        self.camera_fov_deg          = 90
        self.camera_pitch_deg        = -20.0
        self.step_size               = 0.04
        self.lookahead_points        = 12
        self.lookahead_distance      = 5.0
        self.pause_focus_distance    = 0.02
        self.camera_height           = 2.2
        self.camera_radius           = 150.0
        self.refresh_distance        = 50.0
        self.dynamic_loading         = True
        self.flythrough_show_time    = True
        self.flythrough_time_refresh_s = 0.10

        # ── Couleur de la trajectoire (blanc cassé) ─────────────────────────
        self.traj_color = "white"

    # ────────────────────────────────────────────────────────────────────────
    # API publique
    # ────────────────────────────────────────────────────────────────────────

    def configure_viewer(self, **overrides):
        """Met à jour les paramètres persistants du visualiseur."""
        allowed_keys = {
            "camera_near_clip", "flythrough_frame_delay", "camera_anchor_mode",
            "camera_view_mode", "camera_fov_deg", "camera_pitch_deg",
            "step_size", "lookahead_points", "lookahead_distance",
            "pause_focus_distance", "camera_height", "camera_radius",
            "refresh_distance", "dynamic_loading",
            "flythrough_show_time", "flythrough_time_refresh_s",
        }
        unknown = sorted(set(overrides) - allowed_keys)
        if unknown:
            raise ValueError(f"Paramètres viewer inconnus : {', '.join(unknown)}")
        for key, value in overrides.items():
            if value is not None:
                setattr(self, key, value)

    def show_corridor(
        self,
        start=0, end=1, mode="percent",
        width=40, factor=1, color_mode="class",
        hide_wires=False, gnss_offset=(0, 0, 4.137),
    ):
        """Affiche les points LiDAR dans un couloir autour du train."""
        self.current_color_mode = color_mode
        df_seg  = self._get_segment(start, end, mode)
        traj_3d = self._get_traj3d(df_seg, gnss_offset=gnss_offset)
        traj_tree = cKDTree(traj_3d[:, :2])
        candidates = self._get_candidate_tiles(traj_3d, width or 100)

        all_p, all_cl = [], []
        for tile in candidates:
            p, cl = self._get_tile_points(tile["path"], factor=factor, hide_wires=hide_wires)
            dists, _ = traj_tree.query(p[:, :2], k=1)
            mask = dists <= width if width is not None else np.ones(len(p), bool)
            if mask.any():
                all_p.append(p[mask]); all_cl.append(cl[mask])

        if all_p:
            pts, cls = np.concatenate(all_p), np.concatenate(all_cl)
            self._render(pts, cls, traj_3d=traj_3d,
                         window_name=f"Corridor {width}m")

    def show_corridor_flythrough(
        self,
        start=0, end=1, mode="percent",
        width=40, factor=1, color_mode="class",
        hide_wires=False, gnss_offset=(0, 0, 4.137),
        progress_callback=None,
        **viewer_options,
    ):
        """Fly-through optimisé : charge dynamiquement les dalles proches."""
        if progress_callback is None and "progress_callback" in viewer_options:
            progress_callback = viewer_options.pop("progress_callback")
        else:
            viewer_options.pop("progress_callback", None)

        self.configure_viewer(**viewer_options)
        self.current_color_mode   = color_mode
        df_seg  = self._get_segment(start, end, mode)
        traj_3d = self._get_traj3d(df_seg, gnss_offset=gnss_offset)
        self.active_traj_times    = df_seg["time_utc"].to_numpy()
        self.active_progress_callback = progress_callback

        candidates = self._get_candidate_tiles(traj_3d, width or 100)
        dynamic_loader = None

        if self.dynamic_loading:
            candidates = self._get_candidate_tiles(
                traj_3d, max(width or 100, self.camera_radius)
            )

            def dynamic_loader(camera_position):
                return self._load_points_near_position(
                    camera_position[:2], candidates,
                    radius=self.camera_radius, factor=factor, hide_wires=hide_wires,
                )

            initial_pts, initial_cls = dynamic_loader(traj_3d[0])
        else:
            initial_pts, initial_cls = self._load_corridor_points(
                traj_3d=traj_3d,
                candidates=candidates,
                width=width,
                factor=factor,
                hide_wires=hide_wires,
            )
        self._render(
            initial_pts, initial_cls,
            traj_3d=traj_3d,
            window_name=f"Corridor Fly-Through {self.camera_radius}m",
            auto_flythrough=True,
            dynamic_loader=dynamic_loader,
        )
        self.active_progress_callback = None

    # ────────────────────────────────────────────────────────────────────────
    # Rendu PyVista
    # ────────────────────────────────────────────────────────────────────────

    def _render(
        self,
        points, classes,
        traj_3d=None,
        window_name="LiDAR",
        auto_flythrough=False,
        dynamic_loader=None,
    ):
        if len(points) == 0 and traj_3d is None:
            print("Aucun point à afficher.")
            return

        self._print_help()
        self.pts_cache = points
        self.cls_cache = classes

        # Centre de la scène (pour la stabilité numérique)
        if traj_3d is not None:
            offset = np.mean(traj_3d, axis=0)
        elif len(points) > 0:
            offset = np.mean(points, axis=0)
        else:
            offset = np.zeros(3)

        # ── Plotter ─────────────────────────────────────────────────────────
        pl = pv.Plotter(window_size=[1280, 720], title=window_name)
        pl.set_background("#1a1a2e")

        # ── Nuage de points ─────────────────────────────────────────────────
        pts_local = points - offset if len(points) > 0 else np.empty((0, 3))
        cloud = pv.PolyData(pts_local)
        colors_rgb = self._get_color_rgb(points, classes, self.current_color_mode)
        cloud["colors"] = colors_rgb
        pts_actor = pl.add_points(
            cloud, scalars="colors", rgb=True,
            point_size=2, render_points_as_spheres=False,
        )

        # ── Trajectoire ─────────────────────────────────────────────────────
        traj_local = None
        if traj_3d is not None:
            traj_local = traj_3d - offset
            spline = pv.Spline(traj_local, n_points=len(traj_local))
            pl.add_mesh(spline, color=self.traj_color, line_width=2)

        # ── Texte UTC dans la fenêtre ────────────────────────────────────────
        # On crée l'acteur texte une seule fois ;
        # pendant le fly-through on met à jour son contenu via .SetInput().
        time_actor = pl.add_text(
            "UTC : --:--:--",
            position="upper_left",
            font_size=14,
            color="yellow",
            font="courier",
            name="utc_overlay",
        )

        info_actor = pl.add_text(
            f"[T] Fly-through  [C] Couleur  [L] Stop  [Echap] Quitter\n"
            f"Z/S Av/Rec  Q/D G/D  R/F Haut/Bas",
            position="lower_left",
            font_size=9,
            color="#aaaaaa",
            font="courier",
            name="help_overlay",
        )

        def _set_text_actor(actor, text, corner_index=None):
            """Compat update for both vtkTextActor (SetInput) and CornerAnnotation (SetText)."""
            if hasattr(actor, "SetInput"):
                actor.SetInput(text)
                return
            if hasattr(actor, "SetText"):
                # CornerAnnotation expects a corner index.
                if corner_index is not None:
                    actor.SetText(corner_index, text)
                else:
                    try:
                        actor.SetText(text)
                    except TypeError:
                        pass

        # ── Paramètres caméra initiaux ──────────────────────────────────────
        pl.camera.view_angle = float(self.camera_fov_deg)
        if traj_local is not None and len(traj_local) > 0:
            eye = traj_local[0] + np.array([0, 0, self.camera_height])
            fwd = traj_local[min(self.lookahead_points, len(traj_local) - 1)] - traj_local[0]
            fwd = fwd / (np.linalg.norm(fwd) + 1e-12)
            pl.camera.position = eye.tolist()
            pl.camera.focal_point = (eye + fwd * self.lookahead_distance).tolist()
            pl.camera.up = (0, 0, 1)
        else:
            pl.reset_camera()

        def _apply_camera_clipping():
            near = max(1e-4, float(self.camera_near_clip))
            pl.camera.clipping_range = (near, 100000.0)

        _apply_camera_clipping()

        # ── État mutable partagé entre callbacks ────────────────────────────
        state = {
            "flythrough_running":  False,
            "stop_requested":      False,
            "color_mode":         self.current_color_mode,
            "last_loaded_pos":     None,
        }

        camera_up = np.array([0.0, 0.0, 1.0])

        # ────────────────────────────────────────────────────────────────────
        # Fly-through
        # ────────────────────────────────────────────────────────────────────

        def run_flythrough():
            if traj_local is None or len(traj_local) < 2:
                print("Fly-through indisponible : trajectoire trop courte.")
                return
            if state["flythrough_running"]:
                return

            state["flythrough_running"] = True
            state["stop_requested"]     = False
            max_idx = len(traj_local) - 1
            step    = max(float(self.step_size), 1e-3)
            positions = list(np.arange(0.0, float(max_idx) + 1e-9, step))
            if positions[-1] < float(max_idx):
                positions.append(float(max_idx))

            last_time_print   = 0.0
            last_loaded_pos   = None

            def interp_pt(pos):
                pos = float(np.clip(pos, 0, max_idx))
                i0  = int(np.floor(pos))
                i1  = min(i0 + 1, max_idx)
                t   = pos - i0
                return traj_local[i0] * (1 - t) + traj_local[i1] * t

            def process_ui_events():
                """Keep VTK interactor responsive so keyboard callbacks fire during fly-through."""
                try:
                    if pl.iren is not None:
                        pl.iren.process_events()
                except Exception:
                    pass

            def responsive_sleep(delay_s):
                remaining = max(0.0, float(delay_s))
                while remaining > 0.0:
                    dt = min(0.01, remaining)
                    process_ui_events()
                    time.sleep(dt)
                    remaining -= dt

            print("Fly-through en cours… (appuyez sur L pour stopper, Echap pour fermer)")

            for pos in positions:
                process_ui_events()
                if state["stop_requested"]:
                    break

                # ── Direction de visée ──────────────────────────────────────
                target_pos = min(pos + float(self.lookahead_points), float(max_idx))
                p      = interp_pt(pos)
                target = interp_pt(target_pos)
                fwd    = target - p
                norm   = np.linalg.norm(fwd)
                if norm < 1e-6:
                    continue
                fwd /= norm

                # Pitch
                pitch_rad = np.deg2rad(self.camera_pitch_deg)
                fwd = fwd * np.cos(pitch_rad) + camera_up * np.sin(pitch_rad)
                fwd /= np.linalg.norm(fwd) + 1e-12

                # ── Rechargement dynamique des points ───────────────────────
                if dynamic_loader is not None:
                    if (last_loaded_pos is None or
                            np.linalg.norm(p - last_loaded_pos) >= self.refresh_distance):
                        new_pts, new_cls = dynamic_loader(p + offset)
                        self.pts_cache = new_pts
                        self.cls_cache = new_cls
                        pts_local_new = new_pts - offset
                        cloud.points  = pts_local_new
                        cloud["colors"] = self._get_color_rgb(
                            new_pts, new_cls, state["color_mode"]
                        )
                        pts_actor.mapper.SetInputData(cloud)
                        last_loaded_pos = p.copy()

                # ── Pose caméra ─────────────────────────────────────────────
                if self.camera_anchor_mode == "antenna":
                    eye = p.copy()
                else:
                    eye = p + camera_up * self.camera_height

                if self.camera_view_mode == "sky":
                    look_dir = camera_up
                    look_dir /= np.linalg.norm(look_dir) + 1e-12
                    focal = eye + look_dir * self.lookahead_distance
                    up_vec = fwd
                else:
                    focal  = eye + fwd * self.lookahead_distance
                    up_vec = camera_up

                pl.camera.position    = eye.tolist()
                pl.camera.focal_point = focal.tolist()
                pl.camera.up          = up_vec.tolist()
                _apply_camera_clipping()

                # ── Mise à jour de l'heure UTC dans la fenêtre ──────────────
                now = time.time()
                if self.flythrough_show_time and (now - last_time_print) >= self.flythrough_time_refresh_s:
                    t_utc = self._interp_time_at_position(pos, max_idx)
                    if t_utc is not None:
                        label = t_utc.strftime("%Y-%m-%d  %H:%M:%S.%f")[:-3]
                        # Mise à jour du texte directement dans la fenêtre VTK
                        _set_text_actor(time_actor, f"UTC : {label}", corner_index=2)
                        prog_pct = pos / max(1, max_idx) * 100
                        _set_text_actor(
                            info_actor,
                            f"[T] Fly-through  [C] Couleur  [L] Stop  [Echap] Quitter\n"
                            f"Z/S Av/Rec  Q/D G/D  R/F Haut/Bas      "
                            f"{prog_pct:5.1f}%",
                            corner_index=0,
                        )
                        print(
                            f"\rUTC: {label}  |  {prog_pct:5.1f}%",
                            end="", flush=True,
                        )
                        if self.active_progress_callback is not None:
                            try:
                                self.active_progress_callback({
                                    "time_utc":  label,
                                    "progress":  float(pos / max(1, max_idx)),
                                    "position":  float(pos),
                                    "max_index": int(max_idx),
                                })
                            except Exception:
                                pass
                        last_time_print = now

                pl.render()
                responsive_sleep(self.flythrough_frame_delay)

            print()  # saut de ligne après le \r
            state["flythrough_running"] = False
            _set_text_actor(time_actor, "UTC : terminé", corner_index=2)
            pl.render()

        # ────────────────────────────────────────────────────────────────────
        # Callbacks clavier
        # ────────────────────────────────────────────────────────────────────

        def on_t():
            now = time.time()
            if state["flythrough_running"] or (now - self.last_flythrough_trigger) < 0.5:
                return
            self.last_flythrough_trigger = now
            run_flythrough()

        def on_l():
            state["stop_requested"] = True
            print("\nFly-through arrêté.")

        def on_c():
            state["color_mode"] = (
                "altitude" if state["color_mode"] == "class" else "class"
            )
            self.current_color_mode = state["color_mode"]
            if self.pts_cache is not None and len(self.pts_cache) > 0:
                cloud["colors"] = self._get_color_rgb(
                    self.pts_cache, self.cls_cache, state["color_mode"]
                )
                pts_actor.mapper.SetInputData(cloud)
            pl.render()
            print(f"Mode couleur : {state['color_mode']}")

        MOVE_STEP = 7.5

        def on_z():   # Avancer (AZERTY)
            _translate_camera(pl, MOVE_STEP, 0, 0, near_clip=self.camera_near_clip)
        def on_s():   # Reculer
            _translate_camera(pl, -MOVE_STEP, 0, 0, near_clip=self.camera_near_clip)
        def on_q():   # Gauche (AZERTY)
            _translate_camera(pl, 0, -MOVE_STEP, 0, near_clip=self.camera_near_clip)
        def on_d():   # Droite
            _translate_camera(pl, 0, MOVE_STEP, 0, near_clip=self.camera_near_clip)
        def on_r():   # Monter
            _translate_camera(pl, 0, 0, MOVE_STEP, near_clip=self.camera_near_clip)
        def on_f():   # Descendre
            _translate_camera(pl, 0, 0, -MOVE_STEP, near_clip=self.camera_near_clip)

        def on_comma():   # Fly-through plus rapide
            self.flythrough_frame_delay = max(0.001, self.flythrough_frame_delay - 0.005)
            print(f"Délai : {self.flythrough_frame_delay:.4f}s ({1/self.flythrough_frame_delay:.1f} FPS)")

        def on_period():  # Fly-through plus lent
            self.flythrough_frame_delay = min(1.0, self.flythrough_frame_delay + 0.005)
            print(f"Délai : {self.flythrough_frame_delay:.4f}s ({1/self.flythrough_frame_delay:.1f} FPS)")

        def on_v():   # Toggle mode visée
            self.camera_view_mode = "sky" if self.camera_view_mode == "forward" else "forward"
            print(f"Mode visée : {self.camera_view_mode}")

        def on_b():   # Toggle ancrage caméra
            self.camera_anchor_mode = "antenna" if self.camera_anchor_mode == "offset" else "offset"
            print(f"Ancrage caméra : {self.camera_anchor_mode}")

        def on_h():
            self._print_help()

        key_handlers = {
            "t": on_t,
            "l": on_l,
            "c": on_c,
            "z": on_z,
            "s": on_s,
            "q": on_q,
            "d": on_d,
            "r": on_r,
            "f": on_f,
            "comma": on_comma,
            "period": on_period,
            "v": on_v,
            "b": on_b,
            "h": on_h,
        }

        def dispatch_key(keysym):
            if not keysym:
                return
            key = str(keysym).lower()
            # Compat QWERTY/AZERTY
            if key == "w":
                key = "z"
            elif key == "a":
                key = "q"
            elif key == ",":
                key = "comma"
            elif key == ".":
                key = "period"
            cb = key_handlers.get(key)
            if cb is not None:
                cb()

        def _get_keysym_from_iren():
            iren = pl.iren
            if iren is None:
                return None
            if hasattr(iren, "GetKeySym"):
                return iren.GetKeySym()
            interactor = getattr(iren, "interactor", None)
            if interactor is not None and hasattr(interactor, "GetKeySym"):
                return interactor.GetKeySym()
            return None

        def on_keypress_vtk(*_args):
            dispatch_key(_get_keysym_from_iren())

        # High-level bindings (PyVista) + low-level VTK observer for robustness.
        for key, cb in key_handlers.items():
            pl.add_key_event(key, cb)
            if len(key) == 1 and key.isalpha():
                pl.add_key_event(key.upper(), cb)

        try:
            if pl.iren is not None and hasattr(pl.iren, "add_observer"):
                pl.iren.add_observer("KeyPressEvent", on_keypress_vtk)
        except Exception:
            pass

        # Some backends expose punctuation as literal keys.
        pl.add_key_event(",", on_comma)
        pl.add_key_event(".", on_period)

        # ── Lancement ───────────────────────────────────────────────────────
        if auto_flythrough and traj_local is not None:
            # On ouvre la fenêtre sans bloquer, puis on joue le fly-through
            pl.show(auto_close=False, interactive_update=True)
            run_flythrough()
            pl.close()
        else:
            pl.show()

    # ────────────────────────────────────────────────────────────────────────
    # Helpers internes
    # ────────────────────────────────────────────────────────────────────────

    def _get_traj3d(self, df_seg, gnss_offset=(0, 0, 4.137)):
        traj = df_seg[["x_gt", "y_gt", "z_gt_ign69"]].values.copy()
        traj[:, 2] -= gnss_offset[2]
        return traj

    def _get_segment(self, start=None, end=None, mode="percent"):
        temp_df = self.df.sort_values("time_utc").reset_index(drop=True)
        if mode == "percent":
            s = int(len(temp_df) * (start or 0))
            e = int(len(temp_df) * (end   or 1))
            return temp_df.iloc[s:e]
        elif mode == "time":
            start_dt = pd.to_datetime(start)
            end_dt   = pd.to_datetime(end)
            has_date = any(c in str(start) for c in ["-", "/"]) or \
                       any(c in str(end)   for c in ["-", "/"])
            if has_date:
                mask = (temp_df["time_utc"] >= start_dt) & (temp_df["time_utc"] <= end_dt)
                res  = temp_df[mask]
            else:
                t0, t1 = start_dt.time(), end_dt.time()
                mask = (temp_df["time_utc"].dt.time >= t0) & \
                       (temp_df["time_utc"].dt.time <= t1)
                res  = temp_df[mask]
                if len(res) > 0:
                    dt = res["time_utc"].diff().dt.total_seconds().fillna(0)
                    bid = (dt > 10.0).cumsum()
                    res = res[bid == bid.value_counts().index[0]]
            if len(res) == 0:
                raise ValueError(f"Aucun point entre {start} et {end}")
            return res

    def _interp_time_at_position(self, pos, max_idx):
        times = self.active_traj_times
        if times is None or len(times) == 0:
            return None
        if len(times) == 1 or max_idx <= 0:
            return pd.to_datetime(times[0])
        pos = float(np.clip(pos, 0, max_idx))
        i0  = int(np.floor(pos))
        i1  = min(i0 + 1, max_idx)
        t   = pos - i0
        t0  = pd.to_datetime(times[i0])
        t1  = pd.to_datetime(times[i1])
        dt_ns = int((t1.value - t0.value) * t)
        return pd.to_datetime(t0.value + dt_ns)

    def _get_candidate_tiles(self, traj_3d, margin):
        mn = traj_3d[:, :2].min(0) - margin
        mx = traj_3d[:, :2].max(0) + margin
        candidates = []
        for f in os.listdir(self.lidar_folder):
            if not f.endswith(".laz"):
                continue
            path = os.path.join(self.lidar_folder, f)
            with laspy.open(path) as fh:
                h = fh.header
                if h.min[0] > mx[0] or h.max[0] < mn[0] or \
                   h.min[1] > mx[1] or h.max[1] < mn[1]:
                    continue
                candidates.append({
                    "path": path,
                    "bbox": (h.min[0], h.min[1], h.max[0], h.max[1]),
                })
        return candidates

    def _get_tile_points(self, path, factor=1, hide_wires=False):
        key = (path, factor, hide_wires)
        if key in self.tile_cache:
            self.tile_cache.move_to_end(key)
            return self.tile_cache[key]
        with laspy.open(path) as fh:
            las = fh.read()
        p  = np.vstack((las.x[::factor], las.y[::factor], las.z[::factor])).T
        cl = las.classification[::factor]
        if hide_wires:
            mask = ~np.isin(cl, [0, 1])
            p, cl = p[mask], cl[mask]
        self.tile_cache[key] = (p, cl)
        self.tile_cache.move_to_end(key)
        while len(self.tile_cache) > self.tile_cache_limit:
            self.tile_cache.popitem(last=False)
        return p, cl

    def _load_points_near_position(self, pos_xy, candidates, radius, factor=1, hide_wires=False):
        r2 = radius ** 2
        xl, xr = pos_xy[0] - radius, pos_xy[0] + radius
        yl, yr = pos_xy[1] - radius, pos_xy[1] + radius
        all_p, all_cl = [], []
        for tile in candidates:
            mn_x, mn_y, mx_x, mx_y = tile["bbox"]
            if mn_x > xr or mx_x < xl or mn_y > yr or mx_y < yl:
                continue
            p, cl = self._get_tile_points(tile["path"], factor=factor, hide_wires=hide_wires)
            d2 = (p[:, 0] - pos_xy[0]) ** 2 + (p[:, 1] - pos_xy[1]) ** 2
            mask = d2 <= r2
            if mask.any():
                all_p.append(p[mask]); all_cl.append(cl[mask])
        if not all_p:
            return np.empty((0, 3)), np.empty((0,), dtype=int)
        return np.concatenate(all_p), np.concatenate(all_cl)

    def _load_corridor_points(self, traj_3d, candidates, width, factor=1, hide_wires=False):
        """Charge tous les points du corridor pour éviter les trous pendant le fly-through."""
        traj_tree = cKDTree(traj_3d[:, :2])
        all_p, all_cl = [], []
        for tile in candidates:
            p, cl = self._get_tile_points(tile["path"], factor=factor, hide_wires=hide_wires)
            if width is None:
                mask = np.ones(len(p), dtype=bool)
            else:
                dists, _ = traj_tree.query(p[:, :2], k=1)
                mask = dists <= width
            if mask.any():
                all_p.append(p[mask]); all_cl.append(cl[mask])
        if not all_p:
            return np.empty((0, 3)), np.empty((0,), dtype=int)
        return np.concatenate(all_p), np.concatenate(all_cl)

    def _get_color_rgb(self, pts, classes, mode="class"):
        """Retourne un tableau (N, 3) uint8 de couleurs RGB."""
        n = len(classes)
        if mode == "class":
            out = np.full((n, 3), DEFAULT_COLOR_RGB, dtype=np.uint8)
            for cid, col in CLS_COLORS_RGB.items():
                out[classes == cid] = col
            return out
        elif mode == "altitude":
            z   = pts[:, 2]
            zn  = (z - z.min()) / (z.ptp() + 1e-6)
            cmap = plt.get_cmap("terrain")
            return (cmap(zn)[:, :3] * 255).astype(np.uint8)
        return np.full((n, 3), DEFAULT_COLOR_RGB, dtype=np.uint8)

    def _print_help(self):
        os.system("cls" if os.name == "nt" else "clear")
        print("=" * 60)
        print("      STELARIS LiDAR VISUALIZER (PyVista) — COMMANDES")
        print("=" * 60)
        print("\n[NAVIGATION]")
        print("  Z / S     : Avancer / Reculer")
        print("  Q / D     : Gauche / Droite")
        print("  R / F     : Monter / Descendre")
        print("\n[FLY-THROUGH]")
        print("  T         : Lancer le fly-through")
        print("  L         : Stopper le fly-through")
        print("  , / .     : Plus rapide / plus lent")
        print("  V         : Toggle visée (avant / ciel)")
        print("  B         : Toggle ancrage (offset / antenne)")
        print("\n[AFFICHAGE]")
        print("  C         : Switch couleur (classe / altitude)")
        print("  H         : Ré-afficher ce guide")
        print("  Echap     : Quitter")
        print("=" * 60 + "\n")
        print("  L'heure UTC est affichée en direct dans la fenêtre 3D.\n")


# ────────────────────────────────────────────────────────────────────────────
# Utilitaire : translation caméra dans le repère local
# ────────────────────────────────────────────────────────────────────────────

def _translate_camera(pl, dx, dy, dz, near_clip=None):
    """Translate la caméra dans son repère local (avant, droite, haut)."""
    pos   = np.array(pl.camera.position,    dtype=float)
    focal = np.array(pl.camera.focal_point, dtype=float)
    up    = np.array(pl.camera.up,          dtype=float)

    fwd   = focal - pos
    fwd  /= np.linalg.norm(fwd)   + 1e-12
    right = np.cross(fwd, up)
    right /= np.linalg.norm(right) + 1e-12
    up_   = np.cross(right, fwd)

    delta = fwd * dx + right * dy + up_ * dz
    pl.camera.position    = (pos   + delta).tolist()
    pl.camera.focal_point = (focal + delta).tolist()
    if near_clip is not None:
        pl.camera.clipping_range = (max(1e-4, float(near_clip)), 100000.0)
    pl.render()


# ────────────────────────────────────────────────────────────────────────────
# Point d'entrée
# ────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    traj_id = "BORDEAUX_COUTRAS"
    config  = get_traj_paths(traj_id)

    CSV_FILE  = config["sync_csv"]
    LIDAR_DIR = config["lidar_tiles"]
    gnss_offset = config.get("gnss_offset", (0, 0, 4.137))

    if os.path.exists(CSV_FILE) and os.path.exists(LIDAR_DIR):
        df  = pd.read_csv(CSV_FILE)
        viz = LidarVisualizer(LIDAR_DIR, df)
        viz.show_corridor(
            start=0, end=0.01,
            mode="percent",
            width=50,
            factor=1,
            hide_wires=False,
            gnss_offset=gnss_offset,
        )
    else:
        print(f"Erreur : vérifiez {CSV_FILE} et le dossier {LIDAR_DIR}", file=sys.stderr)