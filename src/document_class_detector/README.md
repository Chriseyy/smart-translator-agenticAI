# Document Class Detector

Projekt zum Trainieren, Evaluieren und Analysieren von Dokumenten-Bildklassifikationsmodellen (z.B. **AlexNet**, **ResNet-50**, **Vision Transformer**) auf dem **RVL-CDIP** Dataset.

---

## Projektstruktur

```

src/document_class_detector/
├── configs/            # YAML Konfigurationen für Training
├── mcp_server/         # Modell-Interface für Supervisor / Serving-Anbindung
├── models/             # Modelldefinitionen (alexnet, resnet_50, vit)
└── scripts/            # Trainings-, Test- und Analyse-Skripte

```

---

## Voraussetzungen

- Python + `uv`
- CUDA optional (Training/Inference laufen sonst auf CPU)
- Wichtige Libraries:
  - `torch`, `torchvision`
  - `timm` (für ViT backbone)
  - `optuna` (Hyperparameter Search)
  - `captum` (Explainability)
  - `wandb` optional (Experiment-Tracking)

---

## Dataset / DataLoader

Dataset: **RVL-CDIP** (erwartet unter `data/rvl-cdip/`)

Struktur:
```

data/rvl-cdip/
├── images/
└── labels/
├── train.txt
├── val.txt
└── test.txt

```

Die `labels/*.txt` Dateien enthalten Zeilen im Format:
```

relative/path/to/image.tif <label_id>

```

Loader:
- `scripts/data_loader.py` → `data_loader(...)`
- Gibt immer `(train_dl, val_dl, test_dl)` zurück

---

## Configs (configs/)

Trainingsparameter liegen in YAML-Dateien, z.B.:
- `configs/simple_training.yaml`

Typische Inhalte:
- Dataset subset sizes (`n_train`, `n_val`, `n_test`)
- Training (`epochs`, `batch_size`, `lr`, `momentum`, `weight_decay`)
- Output (`ckpt_dir`, `exp_name`)
- optional Tracking: `track_with: wandb`

---

## Modelle (models/)

Enthält die Modelldefinitionen:

- `alexnet.py` → `build_model(num_classes=...)`
- `resnet_50.py` → `build_model(num_classes=...)`
- `vit.py` → `build_model(num_classes=...)`

Alle Modelle folgen dem gleichen Interface: `build_model(num_classes)`.

---

## Scripts (scripts/)

### 1) Training (`train.py`)
Trainiert ein Modell basierend auf YAML Config und speichert Checkpoints.

```bash
uv run src/document_class_detector/scripts/train.py \
  --config src/document_class_detector/configs/simple_training.yaml
```

Outputs:

* `checkpoints/<ckpt_dir>/<exp_name>/last.ckpt`
* `checkpoints/<ckpt_dir>/<exp_name>/best.ckpt`
* `checkpoints/<ckpt_dir>/<exp_name>/config.json`

Features:

* Optimizer: SGD/Adam
* optional LR Scheduler (`cosine`) + Warmup
* Early Stopping basierend auf `val_loss`

---

### 2) Hyperparameter Optimization (`hyperparameter_opti.py`)

Optuna Study, trainiert pro Trial und maximiert `val_f1_macro`.

```bash
uv run src/document_class_detector/scripts/hyperparameter_opti.py \
  --config src/document_class_detector/configs/simple_training.yaml \
  --trials 100
```

Outputs:

* `checkpoints/<ckpt_dir>/<exp_name>/trial_XXXX/`

  * `trial_config.json`
  * `last.ckpt`
  * `best.ckpt`
* Optuna Storage (SQLite), default:

  * `checkpoints/hyper_learbing/optuna_rvl_alexnet.db`
* CSV Export:

  * `checkpoints/hyper_learbing/<study_name>_trials.csv`

---

### 3) Test auf kompletter Testmenge (`test_model.py`)

Evaluation inkl. Confusion Matrix + Logging von Fehlklassifikationen.

```bash
uv run ./src/document_class_detector/scripts/test_model.py \
  --ckpt-path checkpoints/restnet_50/best.ckpt
```

Outputs:

* `misclassifications/misclassifications.csv`
* `confusion_matrix.png`

---

### 4) Single Image Inference (`test_single.py`)

Inference auf einem einzelnen Bild inkl. Top-k Wahrscheinlichkeiten.

```bash
uv run ./src/document_class_detector/scripts/test_single.py \
  --ckpt-path checkpoints/restnet_50/best.ckpt \
  --image-path data/rvl-cdip/images/...
```

Optional:

* `--class-names <path>` (Textfile: eine Klasse pro Zeile)

---

### 5) Explainability / Attribution (Captum)

Skripte zur Attribution-Visualisierung (Warum predicted das Modell X?).

* Integrated Gradients (IG)
* Occlusion

Funktionalität:

* Batch aus Testset laden
* Attribution-Maps berechnen
* interaktives Browsing mit Pfeiltasten (← / →)

---

## mcp_server/

Enthält die Anbindung, um ein trainiertes Modell dem Supervisor/Serving zugänglich zu machen.
Training/Evaluation funktionieren unabhängig davon.

---

## Checkpoints

Checkpoints (`.ckpt`) enthalten:

* `model_state`
* `optimizer_state`
* `config`
* best metrics (`best_val_loss`, `best_val_f1`)
* timestamp (`saved_at`)

---

## Quickstart

1. Dataset bereitstellen: `data/rvl-cdip/`
2. Trainieren:

```bash
uv run src/document_class_detector/scripts/train.py \
  --config src/document_class_detector/configs/simple_training.yaml
```

3. Testen:

```bash
uv run ./src/document_class_detector/scripts/test_model.py \
  --ckpt-path checkpoints/<exp>/best.ckpt
```

4. Single inference:

```bash
uv run ./src/document_class_detector/scripts/test_single.py \
  --ckpt-path checkpoints/<exp>/best.ckpt \
  --image-path <image>
```

**4. Kommentar zu den Modellen**

Es befinden sich drei Modelle über Git-LFS auf GitLab. Dabei ist das **ResNet_50**-Modell auch das tatsächliche Modell, das in der gesamten Anwendung verwendet wird.

Die anderen beiden Modelle sind **nicht** die Modelle aus der Auswertung, da diese leider durch einen Bedienfehler im Umgang mit GitLab verloren gegangen sind.

Die Modelle, die sich aktuell dort befinden, nutzen dieselbe Architektur, wurden jedoch aufgrund der Abgabefrist mit einem leicht abgewandelten und schnelleren Training trainiert.

Diese Modelle performen etwa **10 Prozent schlechter** als die Modelle aus der Auswertung.
