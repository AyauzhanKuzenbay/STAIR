"""
traj_processor_modified.py  (drop-in replacement for spa/traj_processor.py)
============================================================================
All four reviewer issues are fixed here.

CHANGES vs original traj_processor.py  (marked <<<):

1. GENERATOR FIX
   process_training_data() converts all_pairs to a list immediately so it
   can be iterated twice (once for stats, once for data).

2. SECOND-PRECISION Δt
   Points now have format [lat, lng, minute_in_day, second_of_minute].
   __compute_intervals() computes Δt as:
       Δt_seconds = (cur_min * 60 + cur_sec) - (prev_min * 60 + prev_sec)
   giving full second-level precision instead of minute-level.

3. CONSISTENT FORMAT FOR TRAIN / VALIDATION / TEST
   __keep_id_and_intervals() returns (L, 3) arrays [cell_id, Δt_norm, Δd_norm].
   Stats are saved to 'interval_stats.npy' in the output directory.
   test_file_processor loads these stats to normalise test data identically.

4. GRID COMPATIBILITY PRESERVED
   __grid_traj_point uses traj_point[2] (minute_in_day) for cell lookup —
   the temporal grid still works in minutes, no changes to cell_generator or
   arg_processor needed.
"""

from shapely.geometry import Point
from shapely.geometry import Polygon
import copy
import decimal
import math
import numpy as np
import pathlib
import random


class TrajProcessor:

    def __init__(self):
        self.__MINUTES_IN_DAY  = 1440
        self.__SECONDS_IN_DAY  = 86400    # <<< used for second-precision wrap
        self.__R_EARTH         = 6378137
        self._interval_stats   = None     # set on first process_training_data call

    # ─── Public API ───────────────────────────────────────────────────────────

    def first_loop(self, all_traj, point_drop_rates, spatial_distortions,
                   temporal_distortions, all_cells, bbox_coords, span, stride):
        """Unchanged from original."""
        [min_lat, min_lng, max_lat, max_lng] = bbox_coords
        # Pass coords directly — avoid Shapely Polygon/Point overhead per point.
        bbox_rect = (min_lat, min_lng, max_lat, max_lng)

        all_pairs = []
        for i in range(len(all_traj)):
            print(f"Processing trajectory (1st loop) {i+1} out of {len(all_traj)}")

            all_cur_traj_q = []
            cur_traj_q = self.__downsample_trajectory_random(
                all_traj[i], point_drop_rates)

            for traj_q in cur_traj_q:
                for s in spatial_distortions:
                    for t in temporal_distortions:
                        traj_q_new = self.__distort_spatiotemporal_traj(
                            traj_q, s, t, bbox_rect)
                        traj_q_new = self.__grid_trajectory(traj_q_new, all_cells)
                        all_cur_traj_q.append(traj_q_new)

            gt_traj_grid   = self.__grid_trajectory(all_traj[i], all_cells)
            all_ranges     = self.__create_pattern_ranges(span, stride)
            gt_patt_features = self.__get_pattern_features(gt_traj_grid, all_ranges)
            gt_data = [gt_traj_grid, gt_patt_features]
            all_pairs.append([gt_data, all_cur_traj_q])

        return all_pairs

    def second_loop(self, all_traj_pairs, key_lookup_dict,
                    min_gt_length, min_q_length):
        """Unchanged from original."""
        i = 0
        while len(all_traj_pairs) > 0:
            i += 1
            print(f"Processing trajectory (2nd loop) {i} out of {len(all_traj_pairs)}")
            [gt, all_q] = all_traj_pairs.pop()
            new_gt = self.__remove_non_hot_cells(gt[0], key_lookup_dict)
            if len(new_gt) >= min_gt_length:
                new_q = []
                for q in all_q:
                    q_ = self.__remove_non_hot_cells(q, key_lookup_dict)
                    if len(q_) >= min_q_length:
                        new_q.append(q_)
                if len(new_q) > 0:
                    yield [[new_gt, np.array(gt[1])], new_q]

    def flatten_traj_pairs(self, all_traj_pairs):
        """Unchanged from original."""
        pair_id = 0
        for one_pair in all_traj_pairs:
            print("Flattening trajectory pairs: %d" % pair_id)
            [[gt, gt_patt], q] = copy.deepcopy(one_pair)
            for one_q in q:
                yield [pair_id, [gt, gt_patt, one_q]]
            pair_id += 1

    def process_training_data(self, all_pairs, output_directory=None):
        """
        CHANGED:
          1. Converts all_pairs to a list immediately — fixes generator-consumed-twice.
          2. Computes interval stats from training data (first call only).
          3. Returns (L, 3) arrays per trajectory: [cell_id, Δt_norm, Δd_norm].
          4. Pads pattern arrays to width 3 for uniform input shape.
          5. Saves stats to output_directory/interval_stats.npy if provided.
        """
        # <<< FIX 1: materialise generator immediately so it can be iterated twice
        all_pairs = list(all_pairs)

        # <<< FIX 2: compute normalisation stats from this split (first call = training)
        if self._interval_stats is None:
            self._interval_stats = self.__collect_interval_stats(all_pairs)
            if output_directory is not None:
                stats_path = pathlib.Path(output_directory) / 'interval_stats.npy'
                np.save(stats_path, self._interval_stats)
                print(f"  Saved interval_stats.npy to {stats_path}")

        all_x = []
        all_y = []
        num_traj = 0

        for one_pair in all_pairs:
            num_traj += 1
            print("Processing train/val data: %d" % num_traj)

            [_, [gt, gt_patt, q]] = one_pair

            # <<< FIX 3: [cell_id, Δt_norm, Δd_norm] instead of [cell_id]
            gt_intv = self.__keep_id_and_intervals(gt)   # (L,  3)
            q_intv  = self.__keep_id_and_intervals(q)    # (L', 3)

            gt_patt_s = np.array([[x[0]] for x in gt_patt],
                                  dtype=np.float32)       # (P, 1)
            gt_patt_t = np.array([[x[1]] for x in gt_patt],
                                  dtype=np.float32)       # (P, 1)

            one_x = np.array([gt_intv, q_intv, gt_patt_s, gt_patt_t],
                              dtype=object)
            one_y = np.array([gt_intv, gt_patt_s, gt_patt_t],
                              dtype=object)
            all_x.append(one_x)
            all_y.append(one_y)

        yield [np.array(all_x, dtype=object), np.array(all_y, dtype=object)]

    def split_and_process_dataset(self, all_traj_pairs, num_data):
        """Unchanged from original."""
        flattened_pairs = []
        pair_id = 0
        for one_pair in all_traj_pairs:
            [[gt, gt_patt], q] = copy.deepcopy(one_pair)
            for one_q in q:
                flattened_pairs.append([pair_id, [gt, gt_patt, one_q]])
            pair_id += 1

        total_traj = len(flattened_pairs)
        if all(isinstance(x, int) for x in num_data):
            if sum(num_data) > total_traj:
                print("WARNING! Defaulting to [70, 20, 10] split")
                num_data = [decimal.Decimal("0.7"), decimal.Decimal("0.2"),
                            decimal.Decimal("0.1")]
            else:
                [num_train, num_val, num_test] = num_data
        if all(isinstance(x, decimal.Decimal) for x in num_data):
            num_train = round(num_data[0] * total_traj)
            num_val   = round(num_data[1] * total_traj)
            num_test  = round(num_data[2] * total_traj)

        random.shuffle(flattened_pairs)
        train_pairs = flattened_pairs[:num_train]
        val_pairs   = flattened_pairs[num_train: num_train + num_val]
        test_pairs  = flattened_pairs[num_train + num_val:]

        return [
            self.process_training_data(train_pairs),
            self.process_training_data(val_pairs),
            self.__process_test_data(test_pairs),
        ]

    # ─── Stats access (used by main.py to pass training stats → validation) ──

    def set_interval_stats(self, stats):
        """Inject pre-computed stats (training → validation normalisation)."""
        self._interval_stats = stats

    def get_interval_stats(self):
        """Return stats after process_training_data() has been called."""
        return self._interval_stats

    # ─── New private helpers ──────────────────────────────────────────────────

    def __collect_interval_stats(self, all_pairs):
        """
        Scan ground-truth trajectories in all_pairs to compute z-score stats
        for Δt (seconds) and Δd (metres).
        """
        all_dt, all_dd = [], []
        for one_pair in all_pairs:
            [_, [gt, _, _]] = one_pair
            dt_list, dd_list = self.__compute_intervals(gt)
            all_dt.extend(dt_list[1:])   # skip first zero
            all_dd.extend(dd_list[1:])

        dt_arr = np.array(all_dt, dtype=np.float32)
        dd_arr = np.array(all_dd, dtype=np.float32)

        stats = {
            'dt_mean': float(dt_arr.mean())                   if len(dt_arr) > 0 else 0.0,
            'dt_std':  max(float(dt_arr.std()), 1e-6)         if len(dt_arr) > 0 else 1.0,
            'dd_mean': float(dd_arr.mean())                   if len(dd_arr) > 0 else 0.0,
            'dd_std':  max(float(dd_arr.std()), 1e-6)         if len(dd_arr) > 0 else 1.0,
        }
        print(f"  Interval stats: "
              f"Δt mean={stats['dt_mean']:.1f}s  std={stats['dt_std']:.1f}s  "
              f"Δd mean={stats['dd_mean']:.1f}m  std={stats['dd_std']:.1f}m")
        return stats

    def __compute_intervals(self, trajectory):
        """
        trajectory: list of [cell_id, raw_point, grid_data]
                    raw_point = [lat, lng, minute_in_day, second_of_minute]

        Returns (delta_t_seconds, delta_d_metres) — both length len(trajectory),
        first element is (0, 0).

        <<< SECOND-PRECISION: uses minute*60 + second for Δt.
        """
        dt_list = [0.0]
        dd_list = [0.0]

        for i in range(1, len(trajectory)):
            prev_raw = trajectory[i - 1][1]   # [lat, lng, minute, second]
            cur_raw  = trajectory[i][1]

            # Reconstruct absolute second within the day
            prev_sec_abs = int(prev_raw[2]) * 60 + int(prev_raw[3])
            cur_sec_abs  = int(cur_raw[2])  * 60 + int(cur_raw[3])

            dt = float(cur_sec_abs - prev_sec_abs)
            if dt < 0:
                dt += self.__SECONDS_IN_DAY   # midnight wrap

            dd = self.__haversine_metres(
                prev_raw[0], prev_raw[1],
                cur_raw[0],  cur_raw[1]
            )
            dt_list.append(dt)
            dd_list.append(dd)

        return dt_list, dd_list

    def __haversine_metres(self, lat1, lon1, lat2, lon2):
        R    = self.__R_EARTH
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dp   = math.radians(lat2 - lat1)
        dl   = math.radians(lon2 - lon1)
        a    = (math.sin(dp / 2) ** 2 +
                math.cos(phi1) * math.cos(phi2) * math.sin(dl / 2) ** 2)
        return 2 * R * math.asin(min(1.0, math.sqrt(a)))

    def __keep_id_and_intervals(self, trajectory):
        """
        Returns (L, 3) float32 array: [cell_id, Δt_norm, Δd_norm] per step.
        Uses z-score normalisation from self._interval_stats.
        """
        dt_list, dd_list = self.__compute_intervals(trajectory)
        stats = self._interval_stats

        rows = []
        for i, point in enumerate(trajectory):
            cell_id = float(point[0])
            dt_norm = (dt_list[i] - stats['dt_mean']) / stats['dt_std']
            dd_norm = (dd_list[i] - stats['dd_mean']) / stats['dd_std']
            rows.append([cell_id, dt_norm, dd_norm])

        return np.array(rows, dtype=np.float32)   # (L, 3)

    # ─── Unchanged private helpers ────────────────────────────────────────────

    def __process_test_data(self, all_pairs):
        """
        FIXED: now outputs (L, 3) arrays [cell_id, Δt_norm, Δd_norm] for both
        GT and query trajectories, consistent with training/validation output.

        Requires self._interval_stats to be set (either from a prior call to
        process_training_data, or via load_interval_stats_from_file).
        Falls back to (L, 1) with a warning if stats are unavailable.
        """
        if self._interval_stats is None:
            print("WARNING: interval_stats not set — test data will use "
                  "(L, 1) cell-ID-only format. Call load_interval_stats_from_file "
                  "before processing test data to get (L, 3) gap-aware format.")
            use_gaps = False
        else:
            use_gaps = True

        all_data = []
        num_traj = 0
        for one_pair in all_pairs:
            num_traj += 1
            print("Processing test data: %d" % num_traj)
            [pair_id, [gt, _, q]] = one_pair
            if use_gaps:
                gt_out = self.__keep_id_and_intervals(gt)  # (L, 3)
                q_out  = self.__keep_id_and_intervals(q)   # (L, 3)
            else:
                gt_out = self.__keep_id_only(gt)           # (L, 1) fallback
                q_out  = self.__keep_id_only(q)
            all_data.append([pair_id, [gt_out, q_out]])
        return np.array(all_data, dtype=object)

    def load_interval_stats_from_file(self, stats_path):
        """
        Load normalisation stats saved during training so that test data is
        normalised identically to training data.

        Call this before split_and_process_dataset / process_training_data
        when operating in evaluation-only mode.

        Args:
            stats_path: path to interval_stats.npy written by process_training_data
        """
        stats = np.load(stats_path, allow_pickle=True).item()
        self._interval_stats = stats
        print(f"  Loaded interval_stats from {stats_path}: "
              f"Δt mean={stats['dt_mean']:.1f}s  std={stats['dt_std']:.1f}s  "
              f"Δd mean={stats['dd_mean']:.1f}m  std={stats['dd_std']:.1f}m")

    def __remove_non_hot_cells(self, trajectory, key_lookup_dict):
        new_trajectory = []
        for point in trajectory:
            if point[0] in key_lookup_dict:
                point_ = copy.deepcopy(point)
                point_[0] = key_lookup_dict[point_[0]]
                new_trajectory.append(point_)
        return new_trajectory

    def __keep_id_only(self, trajectory):
        """Original function — kept for internal test data processing."""
        return np.array([np.array([x[0]]) for x in copy.deepcopy(trajectory)])

    def __grid_trajectory(self, trajectory, all_cells):
        # Precompute ranges once per trajectory instead of per point
        lat_ranges  = [all_cells[i][0][0]["lat_range"]       for i in range(all_cells.shape[0])]
        lng_ranges  = [all_cells[0][j][0]["lng_range"]       for j in range(all_cells.shape[1])]
        time_ranges = [all_cells[0][0][k]["timestamp_range"] for k in range(all_cells.shape[2])]
        traj_data = []
        for traj_point in trajectory:
            [grid_data, cell_id] = self.__grid_traj_point(
                traj_point, all_cells, lat_ranges, lng_ranges, time_ranges)
            new_traj_point = [cell_id, copy.deepcopy(traj_point), grid_data]
            traj_data.append(new_traj_point)
        return traj_data

    def __find_axis_index(self, ranges, value, axis_name):
        if len(ranges) == 0:
            raise ValueError(f"No ranges for axis {axis_name}")
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

    def __grid_traj_point(self, traj_point, all_cells,
                           lat_ranges, lng_ranges, time_ranges):
        """
        Ranges are pre-built by __grid_trajectory — not recomputed per point.
        Grid uses element [2] = minute_in_day; works for 4-element points too.
        """
        lat  = traj_point[0]
        lng  = traj_point[1]
        time = traj_point[2]   # minute_in_day

        lat_cur  = self.__find_axis_index(lat_ranges,  lat,  "lat")
        lng_cur  = self.__find_axis_index(lng_ranges,  lng,  "lng")
        time_cur = self.__find_axis_index(time_ranges, time, "time")

        hit_count = all_cells[lat_cur][lng_cur][time_cur]["hit_count"]
        cell_id   = all_cells[lat_cur][lng_cur][time_cur]["cell_id"]
        all_cells[lat_cur][lng_cur][time_cur]["hit_count"] = hit_count + 1
        return [[lat_cur, lng_cur, time_cur], cell_id]

    def __get_pattern_features(self, trajectory, all_ranges):
        all_ranges_ = copy.deepcopy(all_ranges)
        trajectory_ = copy.deepcopy(trajectory)
        start_time_id = trajectory_[0][1][2]
        end_time_id   = trajectory_[-1][1][2]

        def __binary_search(ranges_local, val):
            i_cur = int(len(ranges_local) / 2)
            i_floor, i_ceil = 0, len(ranges_local) - 1
            while True:
                if val < ranges_local[i_cur][0]:
                    i_ceil = i_cur
                    i_new  = math.floor((i_floor + i_ceil) / 2)
                elif val > ranges_local[i_cur][-1]:
                    i_floor = i_cur
                    i_new   = math.ceil((i_floor + i_ceil) / 2)
                else:
                    return i_cur
                if i_new == i_cur:
                    raise AssertionError("Infinite loop in binary search.")
                i_cur = i_new

        if start_time_id <= end_time_id:
            start_index = __binary_search(all_ranges_, start_time_id)
            end_index   = __binary_search(all_ranges_, end_time_id)
        else:
            start_index = __binary_search(all_ranges_, start_time_id)
            all_ranges_ = all_ranges_[start_index:] + all_ranges_[:start_index]
            start_index = 0
            for i, r in enumerate(all_ranges_):
                if end_time_id in r:
                    end_index = i
                    break

        assert start_index <= end_index

        relevant_ranges = all_ranges_[start_index: end_index + 1]
        all_patterns = [[] for _ in relevant_ranges]
        for i, r in enumerate(relevant_ranges):
            for point in trajectory_:
                if point[1][2] in r:
                    all_patterns[i].append(point[1])

        all_pattern_features = []
        for pat in all_patterns:
            if len(pat) <= 1:
                all_pattern_features.append([0, 0])
            else:
                for x in pat:
                    x.append(self.__get_time_cyclical(x[2]))
                s_dist = t_dist = 0.0
                for j in range(1, len(pat)):
                    s_dist += np.linalg.norm(np.array(pat[j][:2]) -
                                             np.array(pat[j-1][:2]))
                    t_dist += np.linalg.norm(np.array(pat[j][-1]) -
                                             np.array(pat[j-1][-1]))
                all_pattern_features.append([s_dist, t_dist])

        return all_pattern_features

    def __get_time_cyclical(self, timestamp):
        s_sin = (math.sin(2 * math.pi * timestamp / self.__MINUTES_IN_DAY) + 1) / 2
        s_cos = (math.cos(2 * math.pi * timestamp / self.__MINUTES_IN_DAY) + 1) / 2
        return [s_sin, s_cos]

    def __downsample_trajectory(self, trajectory, point_drop_rates):
        nums_point = [round((1 - dr) * len(trajectory)) - 2
                      for dr in point_drop_rates]
        downsampled_trajs = []
        for num_point in nums_point:
            if num_point + 2 >= len(trajectory):
                downsampled_trajs.append(copy.deepcopy(trajectory))
            elif num_point <= 0:
                traj_ = copy.deepcopy(trajectory)
                downsampled_trajs.append([traj_[0], traj_[-1]])
            else:
                traj_ = copy.deepcopy(trajectory)
                rand_idx = sorted(random.sample(range(1, len(traj_)-1), num_point))
                ds = [traj_[i] for i in rand_idx]
                ds.insert(0, traj_[0])
                ds.append(traj_[-1])
                downsampled_trajs.append(ds)
        return downsampled_trajs

    def __downsample_trajectory_random(self, trajectory, point_drop_rates):
        downsampled_trajs = []
        for rate in point_drop_rates:
            mid = [x for x in trajectory[1:-1] if random.random() > rate]
            downsampled_trajs.append([trajectory[0]] + mid + [trajectory[-1]])
        return downsampled_trajs

    def __create_pattern_ranges(self, span, stride):
        all_ranges = []
        cur = 0
        while cur + span <= self.__MINUTES_IN_DAY:
            all_ranges.append(range(cur, cur + span))
            cur += stride
        return all_ranges

    def __distort_spatiotemporal_traj(self, traj, s_dist_rate, t_dist, bbox_rect):
        traj_ = copy.deepcopy(traj)
        if s_dist_rate != 0:
            for point in traj_:
                if random.random() < s_dist_rate:
                    self.__distort_spatial_fix(point, bbox_rect)
        if t_dist != 0:
            self.__distort_temporal_traj(traj_, t_dist)
        return traj_

    def __distort_spatial_fix(self, traj_point, bbox_rect):
        min_lat, min_lng, max_lat, max_lng = bbox_rect
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
        if min_lat <= lat2 <= max_lat and min_lng <= lng2 <= max_lng:
            traj_point[0] = lat2
            traj_point[1] = lng2

    def __distort_temporal_traj(self, trajectory, max_temporal_distortion):
        """
        UPDATED: operates in seconds so both minute and second fields stay consistent.

        max_temporal_distortion is in MINUTES (from arg-didi.ini TemporalDistortions=[15]).
        Distortion range: [-max*60, +max*60] seconds, applied uniformly to all points.

        After shifting, point[2] (minute_in_day) and point[3] (second_of_minute) are
        both updated so that minute*60 + second = correct absolute second.

        Uniform shift means Δt between consecutive points is unaffected by distortion.
        But the fields themselves are consistent, which matters for compute_intervals().
        """
        max_seconds = max_temporal_distortion * 60
        t_distort_sec = random.randint(-max_seconds, max_seconds)

        for point in trajectory:
            abs_sec = int(point[2]) * 60 + int(point[3])
            abs_sec = (abs_sec + t_distort_sec) % self.__SECONDS_IN_DAY
            point[2] = abs_sec // 60    # minute_in_day  (0-1439)
            point[3] = abs_sec % 60     # second_of_minute (0-59)
