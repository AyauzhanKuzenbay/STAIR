import os
import numpy as np

DATA_DIR = "/home/zhouyonghao/ayauzhan/spa/xian_data"
OUT_FILE = os.path.join(DATA_DIR, "dataset_structure_report_explained.txt")


def w(f, s=""):
    f.write(str(s) + "\n")


def arr_info(arr):
    return f"shape={arr.shape}, dtype={arr.dtype}"


def safe_listdir(path):
    try:
        return sorted(os.listdir(path))
    except Exception:
        return []


def load_first_npy_from_dir(dir_path):
    files = [x for x in sorted(os.listdir(dir_path)) if x.endswith(".npy")]
    if not files:
        raise FileNotFoundError(f"No .npy files in {dir_path}")
    first_file = os.path.join(dir_path, files[0])
    arr = np.load(first_file, allow_pickle=True)
    return first_file, arr, files


def describe_object_array_sample(sample):
    lines = []
    if isinstance(sample, np.ndarray):
        lines.append(f"type={type(sample)}, shape={sample.shape}, dtype={sample.dtype}")
        if sample.dtype == object:
            try:
                for i, part in enumerate(sample):
                    if isinstance(part, np.ndarray):
                        lines.append(
                            f"  part[{i}] -> ndarray shape={part.shape}, dtype={part.dtype}"
                        )
                    else:
                        lines.append(f"  part[{i}] -> type={type(part)}, value={part}")
            except Exception as e:
                lines.append(f"  failed to inspect parts: {e}")
    else:
        lines.append(f"type={type(sample)}")
    return lines


with open(OUT_FILE, "w", encoding="utf-8") as rep:
    w(rep, "================ DATASET STRUCTURE REPORT (EXPLAINED) ================")
    w(rep, f"Dataset path: {DATA_DIR}")
    w(rep)

    w(rep, "1) HIGH-LEVEL OVERVIEW")
    w(rep, "This dataset follows the TIST-style trajectory similarity format.")
    w(rep, "It contains:")
    w(rep, "  - training data for model learning")
    w(rep, "  - validation data")
    w(rep, "  - test queries and test database")
    w(rep, "  - raw and gridded test trajectories")
    w(rep, "  - top-k cell neighbor information")
    w(rep, "  - cell dictionary / all-cells metadata / interval statistics")
    w(rep)

    w(rep, "2) FILES IN DATASET DIRECTORY")
    for name in sorted(os.listdir(DATA_DIR)):
        full = os.path.join(DATA_DIR, name)
        if os.path.isdir(full):
            w(rep, f"  [DIR]  {name}")
        else:
            w(rep, f"  [FILE] {name}")
    w(rep)

    w(rep, "3) MAIN FILE MEANINGS")
    w(rep, "  1_training_x/        -> training input shards (.npy files split into many chunks)")
    w(rep, "  1_training_y/        -> training target shards")
    w(rep, "  1_validation_x.npy   -> validation inputs")
    w(rep, "  1_validation_y.npy   -> validation targets")
    w(rep, "  1_q_drop20.npy       -> test queries (gridded / processed)")
    w(rep, "  1_db_drop20.npy      -> test database (gridded / processed)")
    w(rep, "  1_raw_q_drop20.npy   -> raw query trajectories before final grid-only representation")
    w(rep, "  1_raw_db_drop20.npy  -> raw database trajectories")
    w(rep, "  1_topk_id.npy        -> top-k neighboring cell ids for each hot cell")
    w(rep, "  1_topk_weight.npy    -> weights for those top-k neighboring cells")
    w(rep, "  1_allcells.npy       -> 3D grid structure of cells")
    w(rep, "  1_celldict.npy       -> mapping / metadata for cells")
    w(rep, "  interval_stats.npy   -> interval statistics for temporal preprocessing")
    w(rep)

    w(rep, "4) STANDARD .NPY FILE SHAPES")
    for name in sorted(os.listdir(DATA_DIR)):
        if name.endswith(".npy"):
            path = os.path.join(DATA_DIR, name)
            try:
                arr = np.load(path, allow_pickle=True)
                w(rep, f"  {name}: {arr_info(arr)}")
            except Exception as e:
                w(rep, f"  {name}: ERROR -> {e}")
    w(rep)

    w(rep, "5) TRAINING DATA")
    train_x_dir = os.path.join(DATA_DIR, "1_training_x")
    train_y_dir = os.path.join(DATA_DIR, "1_training_y")

    if os.path.isdir(train_x_dir) and os.path.isdir(train_y_dir):
        x_first_file, x_arr, x_files = load_first_npy_from_dir(train_x_dir)
        y_first_file, y_arr, y_files = load_first_npy_from_dir(train_y_dir)

        w(rep, f"Training X is stored as a DIRECTORY with {len(x_files)} shard files.")
        w(rep, f"First shard: {x_first_file}")
        w(rep, f"First shard info: {arr_info(x_arr)}")
        w(rep)

        w(rep, f"Training Y is stored as a DIRECTORY with {len(y_files)} shard files.")
        w(rep, f"First shard: {y_first_file}")
        w(rep, f"First shard info: {arr_info(y_arr)}")
        w(rep)

        w(rep, "Expected semantic meaning of one training sample:")
        w(rep, "  X sample has 4 parts: [seqA, seqB, feat1, feat2]")
        w(rep, "  Y sample has 3 parts: [seqY, feat1, feat2]")
        w(rep)

        w(rep, "Example from first training X shard, sample x[0]:")
        for line in describe_object_array_sample(x_arr[0]):
            w(rep, "  " + line)
        w(rep)

        w(rep, "Example from first training Y shard, sample y[0]:")
        for line in describe_object_array_sample(y_arr[0]):
            w(rep, "  " + line)
        w(rep)

        try:
            w(rep, "Interpreting training sample structure:")
            w(rep, f"  x[0,0] = seqA -> shape {x_arr[0,0].shape}")
            w(rep, f"  x[0,1] = seqB -> shape {x_arr[0,1].shape}")
            w(rep, f"  x[0,2] = feat1 -> shape {x_arr[0,2].shape}")
            w(rep, f"  x[0,3] = feat2 -> shape {x_arr[0,3].shape}")
            w(rep, f"  y[0,0] = seqY -> shape {y_arr[0,0].shape}")
            w(rep, f"  y[0,1] = feat1 -> shape {y_arr[0,1].shape}")
            w(rep, f"  y[0,2] = feat2 -> shape {y_arr[0,2].shape}")
            w(rep)
        except Exception as e:
            w(rep, f"Could not interpret first training sample: {e}")
            w(rep)
    else:
        w(rep, "Training directories not found.")
        w(rep)

    w(rep, "6) VALIDATION DATA")
    try:
        vx = np.load(os.path.join(DATA_DIR, "1_validation_x.npy"), allow_pickle=True)
        vy = np.load(os.path.join(DATA_DIR, "1_validation_y.npy"), allow_pickle=True)

        w(rep, f"Validation X: {arr_info(vx)}")
        w(rep, f"Validation Y: {arr_info(vy)}")
        w(rep)
        w(rep, "Validation sample format:")
        w(rep, "  validation_x[i] = [seqA, seqB, feat1, feat2]")
        w(rep, "  validation_y[i] = [seqY, feat1, feat2]")
        w(rep)

        w(rep, f"  vx[0,0] seqA shape = {vx[0,0].shape}")
        w(rep, f"  vx[0,1] seqB shape = {vx[0,1].shape}")
        w(rep, f"  vx[0,2] feat1 shape = {vx[0,2].shape}")
        w(rep, f"  vx[0,3] feat2 shape = {vx[0,3].shape}")
        w(rep, f"  vy[0,0] seqY shape = {vy[0,0].shape}")
        w(rep, f"  vy[0,1] feat1 shape = {vy[0,1].shape}")
        w(rep, f"  vy[0,2] feat2 shape = {vy[0,2].shape}")
        w(rep)

        w(rep, "Interpretation:")
        w(rep, "  - seqA / seqB / seqY are variable-length trajectories")
        w(rep, "  - shape (L, 3) means each point has 3 features after preprocessing")
        w(rep, "  - feat1 and feat2 with shape (4, 1) are extra trajectory-level features")
        w(rep)
    except Exception as e:
        w(rep, f"Validation read error: {e}")
        w(rep)

    w(rep, "7) TEST DATA")
    test_files = [
        "1_q_drop20.npy",
        "1_db_drop20.npy",
        "1_raw_q_drop20.npy",
        "1_raw_db_drop20.npy",
    ]

    for name in test_files:
        path = os.path.join(DATA_DIR, name)
        try:
            arr = np.load(path, allow_pickle=True)
            w(rep, f"{name}: {arr_info(arr)}")
            w(rep, "  Each row looks like: [trajectory_id, trajectory_array]")

            sample = arr[0]
            try:
                traj_id = sample[0]
                traj = sample[1]
                w(rep, f"  sample[0][0] trajectory_id type = {type(traj_id)} value = {traj_id}")
                w(rep, f"  sample[0][1] trajectory array shape = {traj.shape}, dtype={traj.dtype}")

                if len(traj.shape) == 2:
                    w(rep, f"  This means each trajectory has length L={traj.shape[0]} and feature_dim={traj.shape[1]}")
            except Exception as e:
                w(rep, f"  Could not parse first sample: {e}")

            if "raw" in name:
                w(rep, "  RAW version: keeps richer point-level information before final gridded representation.")
            else:
                w(rep, "  PROCESSED version: the representation actually used by the similarity model/evaluation.")
            w(rep)
        except Exception as e:
            w(rep, f"{name}: ERROR -> {e}")
            w(rep)

    w(rep, "8) IMPORTANT INTERPRETATION OF FEATURE DIMENSIONS")
    w(rep, "From your current report:")
    w(rep, "  - processed test trajectories have shape (L, 3)")
    w(rep, "  - raw test trajectories have shape (L, 4)")
    w(rep)
    w(rep, "This strongly suggests:")
    w(rep, "  - (L, 3): model-ready processed trajectory points")
    w(rep, "  - (L, 4): raw trajectory points with one extra original feature retained")
    w(rep)
    w(rep, "So another chat can understand that raw != processed.")
    w(rep)

    w(rep, "9) CELL / GRAPH SUPPORT FILES")
    try:
        topk_id = np.load(os.path.join(DATA_DIR, "1_topk_id.npy"), allow_pickle=True)
        topk_w = np.load(os.path.join(DATA_DIR, "1_topk_weight.npy"), allow_pickle=True)
        w(rep, f"1_topk_id.npy: {arr_info(topk_id)}")
        w(rep, f"1_topk_weight.npy: {arr_info(topk_w)}")
        w(rep, "Interpretation:")
        w(rep, "  - there are 13270 active / hot cells")
        w(rep, "  - each cell stores top-10 neighbor cell ids and weights")
        w(rep)
    except Exception as e:
        w(rep, f"Top-k files read error: {e}")
        w(rep)

    try:
        allcells = np.load(os.path.join(DATA_DIR, "1_allcells.npy"), allow_pickle=True)
        w(rep, f"1_allcells.npy: {arr_info(allcells)}")
        w(rep, "Interpretation:")
        w(rep, "  - grid dimensions are (17, 17, 48)")
        w(rep, "  - likely [spatial_lat_bin, spatial_lng_bin, temporal_bin]")
        w(rep, "  - each entry is an object/dict describing that cell")
        w(rep)
    except Exception as e:
        w(rep, f"1_allcells read error: {e}")
        w(rep)

    try:
        celldict = np.load(os.path.join(DATA_DIR, "1_celldict.npy"), allow_pickle=True).item()
        w(rep, f"1_celldict.npy loaded as Python object of type {type(celldict)}")
        if hasattr(celldict, "__len__"):
            w(rep, f"  number of entries: {len(celldict)}")
        w(rep)
    except Exception as e:
        w(rep, f"1_celldict read error: {e}")
        w(rep)

    try:
        interval_stats = np.load(os.path.join(DATA_DIR, "interval_stats.npy"), allow_pickle=True).item()
        w(rep, f"interval_stats.npy loaded as Python object of type {type(interval_stats)}")
        if isinstance(interval_stats, dict):
            w(rep, "  keys:")
            for k in interval_stats.keys():
                w(rep, f"    - {k}")
        w(rep)
    except Exception as e:
        w(rep, f"interval_stats read error: {e}")
        w(rep)

    w(rep, "10) SHORT SUMMARY FOR OTHER CHATS")
    w(rep, "This dataset is a TIST-style trajectory similarity dataset.")
    w(rep, "Training data is sharded across directories 1_training_x/ and 1_training_y/.")
    w(rep, "A training input sample has 4 parts: [seqA, seqB, feat1, feat2].")
    w(rep, "A training target sample has 3 parts: [seqY, feat1, feat2].")
    w(rep, "Validation has the same semantic structure in single .npy files.")
    w(rep, "Test files store rows as [trajectory_id, trajectory_array].")
    w(rep, "Processed test trajectories use shape (L, 3); raw ones use shape (L, 4).")
    w(rep, "Top-k support files define 10 neighbor cells for each hot cell.")
    w(rep, "Grid metadata is stored in 1_allcells.npy with shape (17, 17, 48).")
    w(rep)

print(f"Saved explained report to: {OUT_FILE}")
