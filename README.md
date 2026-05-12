# STAIR: Spatial-Temporal Similar Trajectory Search under Irregular Time Intervals

This repository contains an implementation of **STAIR**, a trajectory similarity search model designed for spatial-temporal trajectory retrieval under irregular time intervals.

The project is based on two parts:

1. **`preprocess/`** – preprocessing pipeline for raw GPS trajectory data.
2. **`model/`** – STAIR model training and evaluation code.

STAIR extends a GRU-based spatial-temporal trajectory encoder by adding explicit temporal-gap and spatial-displacement information to each trajectory point. The model learns trajectory embeddings and retrieves similar trajectories using embedding-space distance.

---

## Repository Structure

```text
.
├── preprocess/
│   ├── main.py                  # preprocessing entry point
│   ├── arg-xian.ini             # Xi'an
│   ├── arg-didi.ini             # Chengdu
│   ├── arg_processor.py
│   ├── cell_generator.py
│   ├── cell_processor.py
│   ├── file_reader.py
│   ├── file_writer.py
│   ├── test_file_processor.py
│   ├── traj_processor.py
│   └── dataset_report.py
│
├── model/
│   ├── main.py                  # model training/evaluation entry point
│   ├── arg.ini                  # model config example
│   ├── arg_processor.py
│   ├── keras_data_generators.py
│   ├── dnn_model.py             # STAIR model architecture
│   ├── model_processor.py       # training and evaluation logic
│   ├── log_writer.py
│   ├── file_reader.py
│   ├── traj_processor.py
│   └── resource_manager.py
│
└── README.md
```

---

## Main Features

- Spatial-temporal cell-based trajectory representation.
- Temporal interval feature `Δt` between consecutive trajectory points.
- Spatial displacement feature `Δd` between consecutive trajectory cells.
- Hybrid logarithmic-Fourier temporal gap encoding.
- Multi-scale GRU-based temporal representation learning.
- Attention-based trajectory embedding.
- Triplet-based trajectory retrieval training.
- Mean Rank and Hit@K evaluation.

---

## Requirements

Recommended environment:

```text
Python 3.8+
TensorFlow 2.x
NumPy
SciPy
Shapely
h5py
```

Install dependencies:

```bash
pip install numpy scipy shapely h5py tensorflow
```

If you use GPU training, make sure that your CUDA and cuDNN versions are compatible with your TensorFlow version.

---

## Data Format

The preprocessing code expects raw GPS trajectory data in CSV format. Each trajectory should contain ordered GPS points with location and timestamp information.

The preprocessing pipeline converts raw trajectories into `.npy` files used by the model:

```text
1_training_x.npy
1_training_y.npy
1_validation_x.npy
1_validation_y.npy
q_dropXX.npy
db_dropXX.npy
1_topk_id.npy
1_topk_weight.npy
1_celldict.npy
1_allcells.npy
interval_stats.npy
```

Each processed trajectory point is represented with spatial-temporal information, including:

```text
[cell_id, Δt_norm, Δd_norm]
```

where:

- `cell_id` is the spatial-temporal grid cell ID;
- `Δt_norm` is the normalized temporal interval between consecutive points;
- `Δd_norm` is the normalized spatial displacement between consecutive cells.

---

## Step 1: Preprocess Trajectory Data

Go to the preprocessing folder:

```bash
cd myspa
```

Edit the preprocessing configuration file:

```bash
nano arg-xian.ini
```

Important fields:

```ini
[MODE]
ProcessTrainVal = True
ProcessTest = True

[GENERAL]
InputFilePath = /path/to/raw/trajectory.csv
OutputDirectory = xian_data
DatasetMode = didi

[GRID]
SpatialGridLat = 500
SpatialGridLng = 500
TemporalGridLength = 30
HotCellsThreshold = 10

[TRAINVAL]
NumTrain = 100000
NumVal = 10000
PointDropRates = [0,0.2,0.4,0.6]
TemporalDistortions = [15]

[TEST]
TestQName = q_drop50
TestDBName = db_drop50
NumQ = 1000
NumsDB = [100000]
DropRate = 0.5
```

Run preprocessing:

```bash
python main.py --config arg-xian.ini
```

After preprocessing, the output directory should contain training, validation, test, cell dictionary, top-k cell, and interval statistics files.

---

## Step 2: Train and Evaluate STAIR

Go to the model folder:

```bash
cd ../stair_new
```

Edit the model configuration file:

```bash
nano arg.ini
```

Update the paths according to the preprocessing output:

```ini
[DIRECTORY]
TrainingXPath   = /path/to/xian_data/1_training_x.npy
TrainingYPath   = /path/to/xian_data/1_training_y.npy
ValidationXPath = /path/to/xian_data/1_validation_x.npy
ValidationYPath = /path/to/xian_data/1_validation_y.npy
TestGTPath      = /path/to/xian_data/db_drop50.npy
TestQPath       = /path/to/xian_data/q_drop50.npy
TopKIDPath      = /path/to/xian_data/1_topk_id.npy
TopKWeightsPath = /path/to/xian_data/1_topk_weight.npy
OutputDirectory = MODEL_OUT_STAIR
```

Training and evaluation settings:

```ini
[MODE]
IsTraining = True
IsEvaluating = True

[TRAINING]
ModelPath     = MODEL_OUT_STAIR/cp.h5
BatchSize     = 32
TripletMargin = 0.5
Epochs        = 100
Patience      = 5
LossWeights   = [1.0, 1.0, 0.5]
```

Run training and evaluation:

```bash
python main.py --config arg.ini
```

Run in background:

```bash
nohup python main.py --config arg.ini > train_stair.log 2>&1 &
```

Check logs:

```bash
tail -f train_stair.log
```

---

## Evaluation Metrics

The model evaluates retrieval performance using:

- **Mean Rank (MR)** – lower is better;
- **Hit@K** – higher is better, usually reported for `K = 1, 5, 10, 50`.

The evaluation compares query trajectory embeddings with database trajectory embeddings and ranks database trajectories by embedding-space distance.

---

## Notes

- The code requires preprocessed `.npy` files before model training.
- Paths in `.ini` files must be changed before running the code.
- Large raw datasets and generated `.npy` files should not be uploaded to GitHub.
- Model checkpoints, logs, and output folders should be ignored using `.gitignore`.
- If `Δt` is normalized before model input, make sure that the temporal-gap encoder handles the values safely during logarithmic transformation.

---

## Suggested `.gitignore`

```gitignore
# Python
__pycache__/
*.pyc
*.pyo
*.pyd

# Virtual environments
venv/
.env/
.venv/

# Data files
*.npy
*.npz
*.h5
*.hdf5
*.csv
*.pkl
*.pickle

# Model outputs
MODEL_OUT*/
checkpoints/
cp.h5
*.ckpt
*.weights.h5

# Logs
*.log
nohup.out

# OS / IDE
.DS_Store
.vscode/
.idea/
```


