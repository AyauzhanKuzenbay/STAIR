# STAIR

Trajectory similarity search using **TERI** (Temporal Encoding with Relative Information) and **Warpformer** — a GRU-based model with attention for learning trajectory representations.

The pipeline consists of two stages:

```
Raw CSV trajectories → [spa/] Preprocessing → .npy arrays → [traj/] Training & Evaluation
```

---

## Repository Structure

```
teriwarp/
├── spa/                        # Stage 1 — Spatiotemporal Preprocessing
│   ├── main.py                 # Entry point for SPA
│   ├── arg_processor.py        # Reads and validates the .ini config
│   ├── cell_generator.py       # Builds spatiotemporal grid cells
│   ├── cell_processor.py       # Maps trajectories to cells
│   ├── file_reader.py          # Reads raw CSV trajectories
│   ├── file_writer.py          # Writes processed .npy outputs
│   ├── traj_processor.py       # Applies noise/distortion augmentation
│   ├── test_file_processor.py  # Builds query/database sets for testing
│   ├── arg.ini                 # Example config (Chengdu)
│   └── arg-xian.ini            # Example config (Xi'an)
│
├── traj/                       # Stage 2 — Model Training & Evaluation
│   ├── main.py                 # Entry point for training/evaluation
│   ├── arg_processor.py        # Reads and validates the .ini config
│   ├── dnn_model.py            # TERI encoding + GRU/Warpformer model
│   ├── model_processor.py      # Training loop, triplet loss, evaluation
│   ├── keras_data_generators.py# Batch generators for training
│   ├── file_reader.py          # Loads .npy arrays
│   ├── traj_processor.py       # Trajectory utilities
│   ├── log_writer.py           # Logging utilities
│   ├── resource_manager.py     # GPU memory configuration
│   └── arg.ini                 # Example config for traj module
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Requirements

- Python 3.8+
- TensorFlow 2.10+
- NumPy

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

### Stage 1 — Preprocessing (SPA)

Edit `spa/arg.ini` and set your `InputFilePath` to your trajectory CSV, then run:

```bash
cd spa
python main.py --config arg.ini
```

This will generate `.npy` files in the directory specified by `OutputDirectory`.

**Config sections:**

| Section | Key parameters |
|---------|---------------|
| `[MODE]` | `ProcessTrainVal`, `ProcessTest` |
| `[GENERAL]` | `InputFilePath`, `OutputDirectory`, `DatasetMode` |
| `[GRID]` | `BoundingBoxCoords`, `SpatialGridLat/Lng`, `TemporalGridLength`, `K` |
| `[TRAINVAL]` | `NumTrain`, `NumVal`, distortion rates |
| `[TEST]` | `NumQ`, `NumsDB`, `DropRate` |

Two example configs are provided:
- `arg.ini` — Chengdu DiDi dataset
- `arg-xian.ini` — Xi'an DiDi dataset

---

### Stage 2 — Training & Evaluation

Edit `traj/arg.ini` and point `[DIRECTORY]` paths to the `.npy` files produced by Stage 1, then run:

```bash
cd traj
python main.py --config arg.ini
```

**Config sections:**

| Section | Key parameters |
|---------|---------------|
| `[DIRECTORY]` | Paths to SPA output files and `OutputDirectory` |
| `[TRAINING]` | `BatchSize`, `Epochs`, `TripletMargin`, `LossWeights` |
| `[MODEL]` | `GRUCellSize`, `NumGRULayers`, `EmbeddingSize`, `Bidirectional`, `UseAttention` |
| `[PREDICTION]` | `KS` (Recall@K values), `PredictBatchSize` |
| `[GPU]` | `GPUUsed`, `GPUMemory` |

---

## Datasets

The code was tested on the **DiDi Chuxing** trajectory datasets for Chengdu and Xi'an. These datasets are available separately; they are **not** included in this repository.

Bounding boxes used:
- **Chengdu**: `[30.652828, 104.042102, 30.727818, 104.129591]`
- **Xi'an**: `[34.204950, 108.921860, 34.278600, 109.008830]`

---

## Model Architecture

The `traj/` module implements:

- **TERI Encoding** — trainable temporal encoding layer using sinusoidal frequency weights
- **GRU encoder** — multi-layer bidirectional GRU with optional dropout
- **Attention** — self-attention over GRU hidden states
- **Triplet loss** — trains trajectory representations to be similarity-preserving

Evaluation is performed via Recall@K on a query vs. database trajectory set.
