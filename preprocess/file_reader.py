"""
file_reader_modified.py  (drop-in replacement for spa/file_reader.py)
======================================================================
Only ONE method is changed: FileReader.__read_didi.__check_point

BEFORE (3-element point):
    new_trajectory.append([lat, lng, t])
    where t = minute_in_day from CSV

AFTER (4-element point):
    new_trajectory.append([lat, lng, t_min, t_sec])
    where t_min = minute_in_day, t_sec = second_of_minute

Everything else is identical to the original file_reader.py.
"""

from datetime import datetime
from shapely.geometry import Point
from shapely.geometry import Polygon
import ast
import numpy as np
import pathlib


class FileReader:

    def __init__(self):
        self.__MINUTES_IN_DAY = 1440
        self.__SECONDS_IN_DAY = 86400
        self.__PORTO_SECOND_INCREMENT = 15
        self.__SECONDS_IN_MINUTE = 60
        self.__DEFAULT_TRAIN_SIZE_FRACTION = 0.7
        self.__DEFAULT_VALIDATION_SIZE_FRACTION = 0.2
        self.__DEFAULT_TEST_SIZE_FRACTION = 0.1

    def read_trajectory_from_file(self, in_path, dataset_mode,
                                   min_trajectory_length, max_trajectory_length,
                                   bbox_coords, traj_nums):
        min_lat, min_lng, max_lat, max_lng = bbox_coords

        self.bbox = Polygon([
            (min_lng, min_lat),
            (min_lng, max_lat),
            (max_lng, max_lat),
            (max_lng, min_lat),
            (min_lng, min_lat),
        ])

        with open(in_path, "r") as in_file:
            if dataset_mode == "porto":
                return self.__read_porto(in_file, min_trajectory_length,
                                         max_trajectory_length, traj_nums)
            elif dataset_mode == "didi":
                return self.__read_didi(in_file, min_trajectory_length,
                                        max_trajectory_length, traj_nums)
            else:
                raise ValueError(f"'{dataset_mode}' not supported.")

    def read_npy(self, input_directory, file_name):
        fullpath = pathlib.Path(input_directory) / (file_name + ".npy")
        return np.load(fullpath, allow_pickle=True)

    def __read_porto(self, in_file, min_trajectory_length,
                     max_trajectory_length, traj_nums):
        """Unchanged from original."""
        in_file.readline()
        num_train, num_validation = traj_nums
        all_train = []
        all_validation = []
        num_lines = 0

        for line in in_file:
            num_lines += 1
            trajectory = ast.literal_eval(line.split('",\"')[-1].replace('"', ''))

            if (len(trajectory) <= max_trajectory_length and
                    len(trajectory) >= min_trajectory_length):
                start_dtime = datetime.fromtimestamp(int(line.split('",\"')[5]))
                start_second = (start_dtime.hour * 3600 +
                                start_dtime.minute * 60 +
                                start_dtime.second)
                new_traj = self.__check_point_and_add_timestamp(
                    trajectory, start_second)

                if len(new_traj) >= min_trajectory_length:
                    if num_lines <= num_train:
                        all_train.append(new_traj)
                        print("READING TRAINING DATA %d" % num_lines)
                    elif num_lines < (num_validation + num_train):
                        all_validation.append(new_traj)
                        print("READING VALIDATION DATA %d" % num_lines)
                    else:
                        break

        return [all_train, all_validation]

    def __read_didi(self, in_file, min_trajectory_length,
                    max_trajectory_length, traj_nums):
        """Unchanged except __check_point now returns 4-element points."""
        in_file.readline()
        num_train, num_validation = traj_nums
        all_train = []
        all_validation = []
        num_lines = 0

        for line in in_file:
            num_lines += 1
            trajectory = ast.literal_eval(line.split('",\"')[-1].replace('"', ''))

            if (len(trajectory) <= max_trajectory_length and
                    len(trajectory) >= min_trajectory_length):
                new_traj = self.__check_point(trajectory)

                if len(new_traj) >= min_trajectory_length:
                    if num_lines <= num_train:
                        all_train.append(new_traj)
                        print("READING TRAINING DATA %d" % num_lines)
                    elif num_lines < (num_validation + num_train):
                        all_validation.append(new_traj)
                        print("READING VALIDATION DATA %d" % num_lines)
                    else:
                        break

        return [all_train, all_validation]

    def __check_point(self, trajectory):
        """
        CHANGED: now handles 4-element CSV points [lat, lng, t_min, t_sec]
        and preserves both fields in the internal trajectory.

        CSV point format:   [lat, lng, minute_in_day, second_of_minute]
        Internal format:    [lat, lng, minute_in_day, second_of_minute]

        The grid uses element [2] (minute_in_day).
        traj_processor uses both [2] and [3] for second-precision Δt.
        """
        new_trajectory = []

        for point in trajectory:
            if len(point) == 4:
                lat, lng, t_min, t_sec = point
            elif len(point) == 3:
                # Fallback for old 3-element CSV format
                lat, lng, t_min = point
                t_sec = 0
            else:
                continue

            shapely_point = Point(lng, lat)
            if self.bbox.contains(shapely_point):
                new_trajectory.append([lat, lng, t_min, t_sec])

        return new_trajectory

    def __check_point_and_add_timestamp(self, trajectory, start_second):
        """Unchanged from original (Porto only)."""
        cur_second = start_second
        new_trajectory = []

        for point in trajectory:
            lng, lat = point
            if cur_second >= self.__SECONDS_IN_DAY:
                cur_second -= self.__SECONDS_IN_DAY

            shapely_point = Point(lng, lat)
            if self.bbox.contains(shapely_point):
                cur_minute = int(cur_second / self.__SECONDS_IN_MINUTE)
                new_trajectory.append([lat, lng, cur_minute])

            cur_second += self.__PORTO_SECOND_INCREMENT

        return new_trajectory
