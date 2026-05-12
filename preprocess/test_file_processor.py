"""
test_file_processor_modified.py  (drop-in replacement for spa/test_file_processor.py)
======================================================================================
Changes from original:

1. __check_point:
   Reads 4-element points [lat, lng, minute_in_day, second_of_minute] from CSV.

2. __grid_traj_point:
   Uses index access (traj_point[2]) for time instead of destructuring,
   so 4-element points work correctly.

3. __split_id_and_traj  →  __split_id_and_intervals:
   Instead of returning (L, 1) cell-ID-only arrays, returns (L, 3) arrays
   [cell_id, Δt_norm, Δd_norm] using the same normalisation stats as training.

4. __init__:
   Accepts stats_path (path to interval_stats.npy saved by traj_processor).
   The stats ensure test data is normalised identically to training data.

Everything else is identical to the original test_file_processor.py.
"""

import ast
import json
import copy
import math
import numpy as np
import pathlib
import random
from datetime import datetime
from shapely.geometry import Point, Polygon


class TestFileProcessor:

    def __init__(self, input_file_path, line_start, bbox_coords, all_grids,
                 key_lookup_dict, stats_path=None):
        """
        Args:
            input_file_path: path to chengdu_full.csv
            line_start:      number of lines already used by train+val
            bbox_coords:     [min_lat, min_lng, max_lat, max_lng]
            all_grids:       numpy array of all ST cells
            key_lookup_dict: mapping original cell_id → hot-cell int id
            stats_path:      path to interval_stats.npy produced by
                             traj_processor.process_training_data()
                             If None, intervals are left un-normalised (not recommended).
        """
        self.file = open(input_file_path)
        for self.num_line, line in enumerate(self.file):
            if self.num_line == line_start:
                break

        [self.min_lat, self.min_lng,
         self.max_lat, self.max_lng] = bbox_coords

        # Fast rectangle test — avoids Shapely Point/Polygon overhead per point.
        # All bbox checks in this class call self.__in_bbox(lat, lng) instead.
        # The Shapely Polygon is NOT constructed here.

        self.all_grids       = all_grids
        self.key_lookup_dict = key_lookup_dict

        # Precompute grid axis ranges ONCE — avoids rebuilding them on every
        # trajectory point, which caused extreme slowdown in __grid_traj_point.
        self._lat_ranges  = [all_grids[i][0][0]["lat_range"]
                             for i in range(all_grids.shape[0])]
        self._lng_ranges  = [all_grids[0][j][0]["lng_range"]
                             for j in range(all_grids.shape[1])]
        self._time_ranges = [all_grids[0][0][k]["timestamp_range"]
                             for k in range(all_grids.shape[2])]
        # Precompute cell_id lookup — plain Python list is ~4x faster than
        # indexing into a numpy object array and doing a dict lookup per point.
        self._cell_ids = [[[all_grids[i][j][k]["cell_id"]
                            for k in range(all_grids.shape[2])]
                           for j in range(all_grids.shape[1])]
                          for i in range(all_grids.shape[0])]

        self.__MINUTES_IN_DAY         = 1440
        self.__SECONDS_IN_DAY         = 86400
        self.__R_EARTH                = 6378137
        self.__PORTO_SECOND_INCREMENT = 15
        self.__SECONDS_IN_MINUTE      = 60

        # <<< Load interval normalisation stats from training
        if stats_path is not None and pathlib.Path(stats_path).exists():
            self._interval_stats = np.load(stats_path, allow_pickle=True).item()
            print(f"Loaded interval_stats from {stats_path}")
        else:
            # Fallback: no normalisation (identity stats)
            print("WARNING: interval_stats.npy not found. Intervals will not be normalised.")
            self._interval_stats = {
                'dt_mean': 0.0, 'dt_std': 1.0,
                'dd_mean': 0.0, 'dd_std': 1.0,
            }

    def process_data(self, num_data, dataset_mode, min_traj_len, max_traj_len,
                     drop_rate, spatial_distortion, temporal_distortion, start_ID):
        """Unchanged except __split_id_and_traj → __split_id_and_intervals."""
        list_traj_1      = []
        list_traj_2      = []
        list_traj_1_raw  = []
        list_traj_2_raw  = []

        while len(list_traj_1) < num_data:
            line = self.file.readline()
            self.num_line += 1
            if not line:
                break

            if dataset_mode.lower() == 'porto':
                new_traj = self.__process_csv_porto(
                    line, min_traj_len, max_traj_len)
            elif dataset_mode.lower() == 'didi':
                new_traj = self.__process_csv_didi(
                    line, min_traj_len, max_traj_len)
            else:
                assert False, "NOT IMPLEMENTED"

            if new_traj is not None:
                traj_1 = new_traj[0::2]
                traj_2 = new_traj[1::2]

                if drop_rate > 0:
                    traj_1 = self.__downsample_trajectory_random(traj_1, [drop_rate])[0]
                    traj_2 = self.__downsample_trajectory_random(traj_2, [drop_rate])[0]

                if spatial_distortion > 0 or temporal_distortion > 0:
                    traj_1 = self.__distort_spatiotemporal_traj(
                        traj_1, spatial_distortion, temporal_distortion)
                    traj_2 = self.__distort_spatiotemporal_traj(
                        traj_2, spatial_distortion, temporal_distortion)

                traj_1 = self.__grid_trajectory(traj_1)
                traj_1 = self.__remove_non_hot_cells(traj_1)
                traj_2 = self.__grid_trajectory(traj_2)
                traj_2 = self.__remove_non_hot_cells(traj_2)

                if drop_rate > 0:
                    t1_len = int(len(traj_1) / (1 - drop_rate)) if (1 - drop_rate) > 0 else 0
                    t2_len = int(len(traj_2) / (1 - drop_rate)) if (1 - drop_rate) > 0 else 0
                else:
                    t1_len, t2_len = len(traj_1), len(traj_2)

                if t1_len >= min_traj_len // 2 and t2_len >= min_traj_len // 2:
                    # <<< FIX: returns (L, 3) instead of (L, 1)
                    [t1_intv, raw_1] = self.__split_id_and_intervals(traj_1)
                    [t2_intv, raw_2] = self.__split_id_and_intervals(traj_2)

                    list_traj_1.append(np.array([start_ID, t1_intv], dtype=object))
                    list_traj_1_raw.append(np.array([start_ID, raw_1],  dtype=object))
                    list_traj_2.append(np.array([start_ID, t2_intv], dtype=object))
                    list_traj_2_raw.append(np.array([start_ID, raw_2],  dtype=object))
                    start_ID += 1

            print("Line no. %d. Trajectory read: %d out of %d" %
                  (self.num_line, len(list_traj_1), num_data))

        return [
            np.array(list_traj_1,     dtype=object),
            np.array(list_traj_2,     dtype=object),
            np.array(list_traj_1_raw, dtype=object),
            np.array(list_traj_2_raw, dtype=object),
        ]

    def close_file(self):
        if hasattr(self, 'file') and self.file:
            self.file.close()

    # ─── Changed private methods ──────────────────────────────────────────────

    def __check_point(self, trajectory):
        """
        CHANGED: handles 4-element points [lat, lng, minute, second].
        """
        new_trajectory = []
        for point in trajectory:
            if len(point) == 4:
                lat, lng, t_min, t_sec = point
            elif len(point) == 3:
                lat, lng, t_min = point
                t_sec = 0
            else:
                continue

            if self.__in_bbox(lat, lng):
                new_trajectory.append([lat, lng, t_min, t_sec])

        return new_trajectory

    def __grid_traj_point(self, traj_point):
        """
        CHANGED: uses index access so 4-element points work.
        Grid uses element [2] = minute_in_day.
        """
        lat  = traj_point[0]
        lng  = traj_point[1]
        time = traj_point[2]   # minute_in_day

        lat_cur  = self.__find_axis_index(self._lat_ranges,  lat)
        lng_cur  = self.__find_axis_index(self._lng_ranges,  lng)
        time_cur = self.__find_axis_index(self._time_ranges, time)

        cell_id = self._cell_ids[lat_cur][lng_cur][time_cur]
        return [[lat_cur, lng_cur, time_cur], cell_id]

    def __split_id_and_intervals(self, trajectory):
        """
        CHANGED: replaces __split_id_and_traj.
        Returns [(L, 3) float32 array of [cell_id, Δt_norm, Δd_norm],
                 (L, 4) raw trajectory].

        Uses self._interval_stats for z-score normalisation — identical
        normalisation to what training data received.
        """
        dt_list, dd_list = self.__compute_intervals(trajectory)
        stats = self._interval_stats

        id_intv_rows = []
        raw_rows     = []

        for i, point in enumerate(trajectory):
            cell_id = float(point[0])
            dt_norm = (dt_list[i] - stats['dt_mean']) / stats['dt_std']
            dd_norm = (dd_list[i] - stats['dd_mean']) / stats['dd_std']
            id_intv_rows.append([cell_id, dt_norm, dd_norm])
            raw_rows.append(point[1])   # raw_point = [lat, lng, minute, second]

        return [np.array(id_intv_rows, dtype=np.float32),   # (L, 3)
                np.array(raw_rows)]                          # (L, 4)

    def __compute_intervals(self, trajectory):
        """
        trajectory: list of [cell_id, raw_point, grid_data]
                    raw_point = [lat, lng, minute_in_day, second_of_minute]
        Returns (Δt_seconds_list, Δd_metres_list), both length L, first = 0.
        """
        dt_list = [0.0]
        dd_list = [0.0]

        for i in range(1, len(trajectory)):
            prev_raw = trajectory[i - 1][1]
            cur_raw  = trajectory[i][1]

            prev_sec = int(prev_raw[2]) * 60 + int(prev_raw[3])
            cur_sec  = int(cur_raw[2])  * 60 + int(cur_raw[3])

            dt = float(cur_sec - prev_sec)
            if dt < 0:
                dt += self.__SECONDS_IN_DAY

            dd = self.__haversine_metres(prev_raw[0], prev_raw[1],
                                         cur_raw[0],  cur_raw[1])
            dt_list.append(dt)
            dd_list.append(dd)

        return dt_list, dd_list

    def __haversine_metres(self, lat1, lon1, lat2, lon2):
        R    = self.__R_EARTH
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dp   = math.radians(lat2 - lat1)
        dl   = math.radians(lon2 - lon1)
        a    = (math.sin(dp/2)**2 +
                math.cos(phi1) * math.cos(phi2) * math.sin(dl/2)**2)
        return 2 * R * math.asin(min(1.0, math.sqrt(a)))

    # ─── Unchanged private methods ────────────────────────────────────────────

    def __process_csv_porto(self, line, min_trajectory_length, max_trajectory_length):
        trajectory = json.loads(line.split('","')[-1].replace('"', '').strip())
        if (len(trajectory) <= max_trajectory_length and
                len(trajectory) >= min_trajectory_length):
            start_dtime = datetime.fromtimestamp(int(line.split('","')[5]))
            start_second = (start_dtime.hour * 3600 +
                            start_dtime.minute * 60 +
                            start_dtime.second)
            new_traj = self.__check_point_and_add_timestamp_porto(
                trajectory, start_second)
            if len(new_traj) >= min_trajectory_length:
                return new_traj
        return None

    def __process_csv_didi(self, line, min_trajectory_length, max_trajectory_length):
        raw = line.split('","')[-1].replace('"', '').strip()
        # Fast length pre-filter using bracket count — confirmed safe for this
        # file (chengdu_full.csv uses list format throughout; 1 parse error in
        # 1.9M lines, zero tuples).  Skips json.loads for ~60% of lines that
        # fail the length check, avoiding the expensive hot-cell processing path.
        est_len = raw.count('[') - 1
        if est_len < min_trajectory_length or est_len > max_trajectory_length:
            return None
        trajectory = json.loads(raw)
        if (len(trajectory) <= max_trajectory_length and
                len(trajectory) >= min_trajectory_length):
            new_traj = self.__check_point(trajectory)
            if len(new_traj) >= min_trajectory_length:
                return new_traj
        return None

    def __check_point_and_add_timestamp_porto(self, trajectory, start_second):
        cur_second = start_second
        new_trajectory = []
        for point in trajectory:
            if cur_second >= self.__SECONDS_IN_DAY:
                cur_second -= self.__SECONDS_IN_DAY
            if self.__in_bbox(point[0], point[1]):
                cur_minute = int(cur_second / self.__SECONDS_IN_MINUTE)
                new_trajectory.append([point[1], point[0], cur_minute])
            cur_second += self.__PORTO_SECOND_INCREMENT
        return new_trajectory

    def __grid_trajectory(self, trajectory):
        traj_data = []
        for traj_point in trajectory:
            [grid_data, cell_id] = self.__grid_traj_point(traj_point)
            new_traj_point = [cell_id, list(traj_point), grid_data]
            traj_data.append(new_traj_point)
        return traj_data


    def __in_bbox(self, lat, lng):
        """Rectangle containment — O(1), no Shapely overhead."""
        return (self.min_lat <= lat <= self.max_lat and
                self.min_lng <= lng <= self.max_lng)

    def __find_axis_index(self, ranges, value):
        if len(ranges) == 0:
            raise ValueError("Empty ranges")
        if value <= ranges[0][0]:
            return 0
        if value >= ranges[-1][1]:
            return len(ranges) - 1
        lo, hi = 0, len(ranges) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            low, high = ranges[mid]
            if low <= value <= high:
                return mid
            elif value < low:
                hi = mid - 1
            else:
                lo = mid + 1
        if lo >= len(ranges):
            return len(ranges) - 1
        if hi < 0:
            return 0
        return hi if abs(value - ranges[hi][1]) <= abs(ranges[lo][0] - value) else lo

    def __downsample_trajectory_random(self, trajectory, point_drop_rates):
        downsampled_trajs = []
        for rate in point_drop_rates:
            mid = [x for x in trajectory[1:-1] if random.random() > rate]
            downsampled_trajs.append([trajectory[0]] + mid + [trajectory[-1]])
        return downsampled_trajs

    def __distort_spatiotemporal_traj(self, traj, s_dist_rate, t_dist):
        traj_ = [list(p) for p in traj]
        if s_dist_rate != 0:
            for point in traj_:
                if random.random() < s_dist_rate:
                    self.__distort_spatial_fix(point)
        if t_dist != 0:
            self.__distort_temporal_traj(traj_, t_dist)
        return traj_

    def __distort_spatial_fix(self, traj_point):
        old_lat, old_lng = traj_point[0], traj_point[1]
        lat1 = math.radians(old_lat)
        lng1 = math.radians(old_lng)
        bearing = math.radians(random.randint(1, 360))
        d_r = random.randint(0, 30) / self.__R_EARTH
        lat2 = math.asin(math.sin(lat1) * math.cos(d_r) +
                         math.cos(lat1) * math.sin(d_r) * math.cos(bearing))
        lng2 = lng1 + math.atan2(
            math.sin(bearing) * math.sin(d_r) * math.cos(lat1),
            math.cos(d_r) - math.sin(lat1) * math.sin(lat2))
        lat2, lng2 = math.degrees(lat2), math.degrees(lng2)
        if self.__in_bbox(lat2, lng2):
            traj_point[0] = lat2
            traj_point[1] = lng2

    def __distort_temporal_traj(self, trajectory, max_temporal_distortion):
        """Same as traj_processor: shift in seconds, keep minute+second consistent."""
        max_seconds = max_temporal_distortion * 60
        t_distort_sec = random.randint(-max_seconds, max_seconds)
        for point in trajectory:
            abs_sec = int(point[2]) * 60 + int(point[3])
            abs_sec = (abs_sec + t_distort_sec) % self.__SECONDS_IN_DAY
            point[2] = abs_sec // 60
            point[3] = abs_sec % 60

    def __remove_non_hot_cells(self, trajectory):
        new_trajectory = []
        for point in trajectory:
            if point[0] in self.key_lookup_dict:
                point_ = list(point)
                point_[0] = self.key_lookup_dict[point_[0]]
                new_trajectory.append(point_)
        return new_trajectory
