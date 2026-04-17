# Changelog

A stage-by-stage summary of how the GNSSGuard pipeline was built. Commit
titles in the history are terse; this file is the narrative version.

## Stage 0 — project scaffold
Initial repository layout, `.gitignore`, MIT `LICENSE`.

## Stage 1 — data preprocessing
`code/datapreprocessing.py` and `notebook/p1.ipynb`.

- Per-channel split of `train.csv` into `ch0 … ch7`.
- Filter rows where every observable is zero (dead channel).
- Build 100-sample history windows ending at each target timestamp
  and tag each window with a `batch_id` for fast `groupby` access.
- Temporal 70/30 split inside each class.

## Stage 2 — signal analysis
`notebook/p2.ipynb`.

- STFT, spectrogram, and Welch PSD views of PIP / PQP / CN0 /
  Carrier_phase to eyeball how spoofed vs. authentic segments differ.
- `GNSSSignalAnalyzer` helper class.

## Stage 3 — time-series transformer
`notebook/p3.ipynb`, checkpoint `best_model_6.pth`.

- `TimeSeriesTransformer`: 3-layer encoder, `d_model=128`, 8 heads,
  sinusoidal positional encoding.
- Pretrained as a forecaster: predict the target-row feature vector
  from the 100-sample history (MSE loss).
- Classifier fine-tune via `Head` MLP (`finetuned_with_head.pth`).
- XGBoost on the fused `[forecast | target_row | pre-projection latent]`
  features (`transformer_model_xgb.pkl`).

## Stage 4 — image-analysis model
`notebook/p4.ipynb` and `p5.ipynb`, checkpoints `best_cnn_model.pth`
and `best_cnn_model_p5.pth`.

- `ImageAnalysis`: 2-layer Transformer (`d_model=64`) feeding a
  `CNNBlock` (three Conv2d stages, global average pool, MLP head).
- Input is the 100-sample window with the target row concatenated on
  (101 rows × 14 features), treated as an image.
- XGBoost on `[target_row | cnn_features]` (`image_model_xgb.pkl`),
  Optuna-tuned on macro-F1 over 50 trials.

## Stage 5 — single-row XGBoost baseline
`notebook/p6.ipynb`, `xgb_single_row_model.pkl`.

- XGBoost and RandomForest classifiers on raw per-row features with
  no temporal context — the control against the windowed models.
- Used to quantify how much lift the 100-sample window actually adds.

## Stage 6 — fusion and inference
`notebook/p7.ipynb`, `final_.ipynb`, and `resulting.ipynb`.
Checkpoint `best_model.pth` (transformer + CNN + head jointly fine-tuned).

- Joint fine-tune of `TimeSeriesTransformer`, `ImageAnalysis`, and
  `Head` over 15 epochs with `BCEWithLogitsLoss`; layer-group learning
  rates so the pretrained encoders adapt slowly.
- Fused feature vector:
  `[forecast | target_row | pre-projection latent | cnn_features]`
  fed into either the MLP head or an Optuna-tuned XGBoost.
- `resulting.ipynb` runs inference on the test set and writes
  `dataset/result.csv` (`time, channel, confidence, spoofed`).

## Stage 7 — documentation
- README rewritten with architecture flowchart and class UML diagrams.
- `requirements.txt` added.
- This changelog added.
