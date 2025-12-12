"""
agents_and_tasks.py
-------------------

Defines:
  - Task
  - EKFLocalizer
  - Robot

UPDATED:
  * LiDAR-based EKF update is now STABLE:
      - uses only top-3 informative beams
      - rejects near-zero / max-range beams
      - larger R for LiDAR
      - damped Kalman gain
      - applied only every 4th step
"""

import numpy as np


# ============================================================
#                     TASK CLASS
# ============================================================

class Task:
    """Represents a warehouse task with timing/utility info."""
    def __init__(self, name, location, release_time, deadline, duration, utility):
        self.name = name
        self.location = tuple(location)
        self.release_time = float(release_time)
        self.deadline = float(deadline)
        self.duration = float(duration)
        self.utility = float(utility)

    def __repr__(self):
        return (f"Task({self.name}, loc={self.location}, "
                f"window=[{self.release_time}, {self.deadline}], "
                f"duration={self.duration}, util={self.utility})")


# ============================================================
#                EKF LOCALIZER
# ============================================================

class EKFLocalizer:
    def __init__(self, init_pose=None):
        # state: [x, y, theta]
        self.x = np.zeros(3)
        self.P = np.eye(3) * 0.1

        if init_pose is not None:
            self.x = np.array(init_pose, dtype=float)

        # Process noise on control [v, w]
        self.Q = np.diag([0.01, 0.01])

        # GPS-like measurement noise (x, y)
        self.R = np.diag([0.01, 0.01])

        # H mapping state -> GPS measurement
        self.H = np.array([
            [1, 0, 0],
            [0, 1, 0],
        ])

        # LiDAR parameters
        self.lidar_sigma = 0.05
        self.max_range = 8.0

    def get_estimate(self):
        return np.array(self.x)

    # ----------------------------------------------------------
    # PREDICT STEP
    # ----------------------------------------------------------
    def predict(self, u, dt):
        v, w = u
        x, y, th = self.x

        xp = x + v * np.cos(th) * dt
        yp = y + v * np.sin(th) * dt
        thp = th + w * dt

        self.x = np.array([xp, yp, thp])

        Fx = np.array([
            [1, 0, -v * np.sin(th) * dt],
            [0, 1,  v * np.cos(th) * dt],
            [0, 0,  1]
        ])

        Fu = np.array([
            [np.cos(th) * dt, 0],
            [np.sin(th) * dt, 0],
            [0, dt],
        ])

        self.P = Fx @ self.P @ Fx.T + Fu @ self.Q @ Fu.T

    # ----------------------------------------------------------
    # GPS UPDATE
    # ----------------------------------------------------------
    def update(self, z):
        z = np.array(z)
        y = z - self.H @ self.x           # innovation
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = (np.eye(3) - K @ self.H) @ self.P

    # ----------------------------------------------------------
    # STABLE LIDAR EKF UPDATE (range-only, top-3 beams)
    # ----------------------------------------------------------
    def update_lidar(self, measured, expected):
        """
        Range-only LiDAR update that is *deliberately conservative*:

        - Uses ONLY the 2–3 beams with largest |z_i - h_i|
        - Rejects beams:
            * too close (< 0.1 m)
            * near max range (no hit)
            * whose expected distance is degenerate
        - Uses larger R to avoid over-trusting LiDAR
        - Damps Kalman gain to avoid wild corrections
        """
        z = np.asarray(measured, dtype=float)
        h = np.asarray(expected, dtype=float)
        N_total = len(z)
        if N_total == 0:
            return

        # 1) Innovation per beam
        err = np.abs(z - h)

        # 2) Sort beams by informativeness (largest innovation first)
        idx_sorted = np.argsort(err)[::-1]

        # 3) Select up to 3 valid beams
        selected = []
        for idx in idx_sorted:
            if len(selected) >= 3:
                break

            zi = z[idx]
            hi = h[idx]

            # discard invalid / uninformative beams
            if zi < 0.1:                          # too close / noisy
                continue
            if zi > self.max_range - 0.3:         # likely "no return"
                continue
            if hi < 0.05 or hi > self.max_range - 0.1:
                continue

            selected.append(idx)

        if len(selected) == 0:
            return

        selected = np.array(selected, dtype=int)
        N = len(selected)

        # 4) Innovation vector for selected beams
        y = (z - h)[selected]

        # 5) Build Jacobian H_sel (N x 3) for selected beams only
        H_sel = np.zeros((N, 3))
        eps = 1e-3
        x0 = self.x.copy()

        # angles for all beams
        all_angles = x0[2] + np.linspace(-np.pi, np.pi, N_total)

        for row, j in enumerate(selected):
            dist = h[j]
            ang = all_angles[j]

            if dist < 1e-3:
                continue

            # hit point in world
            x_hit = x0[0] + dist * np.cos(ang)
            y_hit = x0[1] + dist * np.sin(ang)

            dx = x_hit - x0[0]
            dy = y_hit - x0[1]

            # ∂range / ∂x, ∂range / ∂y
            H_sel[row, 0] = -dx / dist
            H_sel[row, 1] = -dy / dist

            # finite-difference for ∂range / ∂theta
            x_pert = x0.copy()
            x_pert[2] += eps
            dxp = x_hit - x_pert[0]
            dyp = y_hit - x_pert[1]
            dist_p = np.hypot(dxp, dyp)
            H_sel[row, 2] = (dist_p - dist) / eps

        # 6) Measurement noise (larger -> more conservative fusion)
        #    sigma_r = 0.5 m  -> variance = 0.25
        R_lidar = np.eye(N) * 0.25

        # 7) EKF update with damped Kalman gain
        S = H_sel @ self.P @ H_sel.T + R_lidar + 1e-6 * np.eye(N)
        K = self.P @ H_sel.T @ np.linalg.inv(S)

        # DAMPING to avoid aggressive snapping
        K *= 0.4

        dx_update = K @ y
        self.x = self.x + dx_update

        # normalize heading
        self.x[2] = np.arctan2(np.sin(self.x[2]), np.cos(self.x[2]))

        self.P = (np.eye(3) - K @ H_sel) @ self.P
        # enforce symmetry numerically
        self.P = 0.5 * (self.P + self.P.T)


# ============================================================
#                        ROBOT CLASS
# ============================================================

class Robot:
    """
    Simulated unicycle robot with:
      - true state
      - EKF localizer (GPS + LiDAR)
      - realistic motion noise
      - waypoint following
    """

    def __init__(self, robot_id, init_pose, v_max=0.8, omega_max=1.2):
        self.id = robot_id

        # True state
        self.x_true = np.array(init_pose, dtype=float).reshape(3,)

        # EKF
        self.ekf = EKFLocalizer(init_pose)

        # Limits
        self.v_max = v_max
        self.omega_max = omega_max

        # Path tracking
        self.path = []
        self.current_wp_idx = 0

        # Motion noise params (as you had them)
        self.odom_bias_v = 0.01
        self.odom_bias_w = 0.01
        self.motion_noise_base = 0.01
        self.motion_noise_gain = 0.01
        self.theta_noise_base = 0.01
        self.theta_noise_gain = 0.01

        # GPS sensor params
        self.gps_noise_std = 0.25
        self.measurement_dropout_prob = 0.15

        # LiDAR EKF fusion cadence (every 4th step)
        self.lidar_update_counter = 0

    # ------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------
    def set_path(self, path_cells):
        self.path = [tuple(p) for p in path_cells]
        self.current_wp_idx = 0

    def has_finished_path(self):
        return self.current_wp_idx >= len(self.path)

    def get_current_waypoint(self):
        if self.has_finished_path():
            return None
        return self.path[self.current_wp_idx]

    # ------------------------------------------------------
    # Getters (REQUIRED by main_sim)
    # ------------------------------------------------------
    def get_true_pose(self):
        return tuple(self.x_true)

    def get_est_pose(self):
        return tuple(self.ekf.get_estimate())

    # ------------------------------------------------------
    # Controller
    # ------------------------------------------------------
    def compute_control(self, waypoint):
        if waypoint is None:
            return 0.0, 0.0

        x, y, th = self.x_true
        wx, wy = waypoint

        dx = wx - x
        dy = wy - y
        dist = np.hypot(dx, dy)

        target_angle = np.arctan2(dy, dx)
        angle_err = np.arctan2(
            np.sin(target_angle - th),
            np.cos(target_angle - th)
        )

        ANGLE_THRESH = np.deg2rad(30)

        if abs(angle_err) > ANGLE_THRESH:
            v = 0.0
            w = 1.2 * angle_err
            w = np.clip(w, -self.omega_max, self.omega_max)
            return float(v), float(w)

        k_v = 0.9
        k_w = 1.2
        v = np.clip(k_v * dist, 0, self.v_max)
        w = np.clip(k_w * angle_err, -self.omega_max, self.omega_max)

        return float(v), float(w)

    # ------------------------------------------------------
    # Expected lidar (geometric, from TRUE pose)
    # ------------------------------------------------------
    def _expected_lidar_ranges(self, grid_map, max_range=8.0, num_rays=60):
        h, w = grid_map.shape
        x, y, th = self.x_true

        angles = th + np.linspace(-np.pi, np.pi, num_rays)

        expected = np.zeros(num_rays)
        step = 0.05

        for i, ang in enumerate(angles):
            d = 0.0
            while d < max_range:
                rx = x + d*np.cos(ang)
                ry = y + d*np.sin(ang)
                gx, gy = int(rx), int(ry)

                if gx < 0 or gy < 0 or gx >= w or gy >= h:
                    break

                if grid_map[gy, gx] == 1:
                    break

                d += step

            expected[i] = min(d, max_range)

        return expected

    # ------------------------------------------------------
    # MAIN STEP
    # ------------------------------------------------------
    def step(self, dt, grid_map=None, sensor_noise_std=0.05):
        wp = self.get_current_waypoint()
        v_cmd, w_cmd = self.compute_control(wp)

        prev_state = self.x_true.copy()
        prev_est = self.ekf.x.copy()

        # ---------------------------
        # true noisy motion
        # ---------------------------
        x, y, th = self.x_true

        self.odom_bias_v += np.random.normal(0.0, 0.0001)
        self.odom_bias_w += np.random.normal(0.0, 0.0001)

        v_eff = v_cmd * (1.0 + self.odom_bias_v)
        w_eff = w_cmd * (1.0 + self.odom_bias_w)

        sigma_xy = self.motion_noise_base + self.motion_noise_gain * abs(v_eff)
        sigma_th = self.theta_noise_base + self.theta_noise_gain * abs(w_eff)

        nx = np.random.normal(0.0, sigma_xy)
        ny = np.random.normal(0.0, sigma_xy)
        nth = np.random.normal(0.0, sigma_th)

        x_new = x + v_eff * np.cos(th) * dt + nx
        y_new = y + v_eff * np.sin(th) * dt + ny
        th_new = th + w_eff * dt + nth
        th_new = np.arctan2(np.sin(th_new), np.cos(th_new))

        self.x_true[:] = [x_new, y_new, th_new]

        # ---------------------------
        # collision check
        # ---------------------------
        collided = False
        if grid_map is not None:
            cx = int(self.x_true[0])
            cy = int(self.x_true[1])

            if (cx < 0 or cy < 0 or
                cy >= grid_map.shape[0] or cx >= grid_map.shape[1] or
                grid_map[cy, cx] == 1):

                collided = True
                self.x_true = prev_state.copy()
                self.ekf.x = prev_est.copy()

                self.x_true[0] -= 0.15 * np.cos(th)
                self.x_true[1] -= 0.15 * np.sin(th)

        # ---------------------------
        # waypoint check
        # ---------------------------
        if wp is not None:
            if np.hypot(self.x_true[0] - wp[0], self.x_true[1] - wp[1]) < 0.25:
                self.current_wp_idx += 1

        # ---------------------------
        # EKF PREDICT
        # ---------------------------
        if self.has_finished_path() and v_cmd == 0 and w_cmd == 0:
            # Idle robot: estimate = true pose
            self.ekf.x[:] = self.x_true
            self.ekf.P[:] = np.eye(3) * 0.05
        else:
            self.ekf.predict((v_eff, w_eff), dt)

        # ---------------------------
        # STABLE LIDAR UPDATE (every 4th step)
        # ---------------------------
        if grid_map is not None:
            self.lidar_update_counter = (self.lidar_update_counter + 1) % 4
            if self.lidar_update_counter == 0:
                exp_ranges = self._expected_lidar_ranges(grid_map, num_rays=60)
                meas_ranges = self.lidar_scan(grid_map, num_rays=60)
                self.ekf.update_lidar(meas_ranges, exp_ranges)

        # ---------------------------
        # GPS UPDATE
        # ---------------------------
        if np.random.rand() > self.measurement_dropout_prob:
            mx = self.x_true[0] + np.random.normal(0.0, self.gps_noise_std)
            my = self.x_true[1] + np.random.normal(0.0, self.gps_noise_std)
            self.ekf.update([mx, my])

        return collided

    # ------------------------------------------------------
    # FULL LIDAR BEAM MODEL (unchanged, forward sensor)
    # ------------------------------------------------------
    def lidar_scan(self, grid_map, max_range=8.0, num_rays=60, fov=np.pi * 2.0):
        if grid_map is None:
            return np.zeros(num_rays, dtype=float)

        h, w = grid_map.shape
        x, y, th = self.x_true

        w_hit   = 0.70
        w_short = 0.15
        w_max   = 0.10
        w_rand  = 0.05

        sigma_hit = 0.01
        lambda_short = 3.0

        half_fov = 0.5 * fov
        angles = th + np.linspace(-half_fov, half_fov, num_rays)

        expected = np.zeros(num_rays)
        measured = np.zeros(num_rays)

        step = 0.05

        # geometric expected
        for i, ang in enumerate(angles):
            d = 0.0
            while d < max_range:
                rx = x + d*np.cos(ang)
                ry = y + d*np.sin(ang)
                gx = int(rx)
                gy = int(ry)

                if gx < 0 or gy < 0 or gx >= w or gy >= h:
                    break
                if grid_map[gy, gx] == 1:
                    break

                d += step

            expected[i] = min(d, max_range)

        # noisy beam model
        for i, zexp in enumerate(expected):
            r = np.random.rand()

            if r < w_hit:
                z = np.random.normal(zexp, sigma_hit)
                z = np.clip(z, 0, max_range)

            elif r < w_hit + w_short:
                if zexp < 1e-6:
                    z = 0.0
                else:
                    z = np.random.exponential(1.0 / lambda_short)
                    z = np.clip(z, 0, zexp)

            elif r < w_hit + w_short + w_max:
                z = max_range

            else:
                z = np.random.uniform(0.0, max_range)

            measured[i] = z

        return measured


# END OF FILE
