# Changelog

Stage-by-stage notes on how GNSSGuard came together. The commit titles on a
few early commits are short (`s`, `CNN`, `final`). This file is what they
should have said.

## Stage 0. Scaffold

Repo layout, `.gitignore`, MIT `LICENSE`. Nothing interesting yet.

## Stage 1. Preprocessing

`code/datapreprocessing.py` and `notebook/p1.ipynb`.

First real pass over `train.csv`. Split by `channel` so ch0…ch7 each get
their own file, sort by `time`, and drop any row where every observable
(Carrier_Doppler, Pseudorange, TOW, Carrier_phase, EC/LC/PC, PIP/PQP,
TCD, CN0) is exactly zero. Those are tracking drop-outs and they ruin
anything downstream if you leave them in.

For every target timestamp, we grab the 100 samples that came before it
and stamp the whole group with a `batch_id`. Downstream the dataset
class does `groupby(batch_id)` once in `__init__` and then
`__getitem__` is a hash lookup.

Train/val is a temporal 70/30 inside each class. Random splits leak
because the windows overlap.

## Stage 2. What does a spoofed signal even look like?

`notebook/p2.ipynb`.

Before throwing models at the data I wanted to see the thing with my own
eyes. STFT, spectrograms, and Welch PSDs over PIP / PQP / CN0 /
Carrier_phase for authentic vs. spoofed segments. There's a
`GNSSSignalAnalyzer` class that wraps all of it. Not production code,
but the plots were the reason I ended up giving the CNN branch access
to the raw window instead of only to summary features.

## Stage 3. TimeSeriesTransformer

`notebook/p3.ipynb`. Checkpoint: `best_model_6.pth`.

Vanilla 3-layer Transformer encoder, `d_model=128`, 8 heads, sinusoidal
positional encoding. Pretrained as a forecaster: predict the target
row's feature vector from the 100-sample history, MSE loss.

Then a classifier fine-tune on top with a small `Head` MLP
(`finetuned_with_head.pth`). Interestingly, the pre-projection latent
(128-d) turned out to be more useful to downstream XGBoost than the
forecast itself. I ended up keeping both in the fused feature.

`transformer_model_xgb.pkl` is XGBoost fit on
`[forecast | target_row | pre-projection latent]`.

## Stage 4. ImageAnalysis (Transformer + CNN)

`notebook/p4.ipynb`, refined in `p5.ipynb`. Checkpoints:
`best_cnn_model.pth`, `best_cnn_model_p5.pth`.

Same window, but I concatenate the target row onto the end so the model
sees 101 rows × 14 features, then treat that matrix like a small image.
A 2-layer Transformer (`d_model=64`) does the initial shaping, a
`CNNBlock` (three Conv2d stages, BatchNorm, Dropout, adaptive global
average pool) squeezes it down, and a little MLP head emits the logit
plus a 128-d feature vector.

`image_model_xgb.pkl` is XGBoost on `[target_row | cnn_features]`,
Optuna-tuned on macro-F1 for 50 trials.

## Stage 5. Single-row XGBoost baseline

`notebook/p6.ipynb`. Saved as `xgb_single_row_model.pkl`.

Plain XGBoost and RandomForest on raw per-row features. No windows, no
embeddings, one snapshot in, one prediction out. The point is to have a
control: if the fancy temporal stack isn't beating this, the windows
aren't buying anything.

Spoiler: on most of the data the single-row baseline is extremely
competitive. Where the window pays off is on low-CN0 and edge-of-track
segments, which is also where spoofing is most worth detecting.

## Stage 6. Fusion and test-set inference

`notebook/p7.ipynb`, `final_.ipynb`, `resulting.ipynb`.
Joint checkpoint: `best_model.pth`.

Jointly fine-tune `TimeSeriesTransformer`, `ImageAnalysis`, and `Head`
for 15 epochs with `BCEWithLogitsLoss`. Layer-group learning rates so
the pretrained encoders only adapt slowly (base_lr × 0.1) while the
head trains at full speed.

Fused feature for the production classifier:
`[forecast | target_row | pre-projection latent | cnn_features]`. That
goes into either the MLP head or an Optuna-tuned XGBoost, whichever
wins on val.

`resulting.ipynb` runs inference on the test set and writes
`dataset/result.csv` with `time, channel, confidence, spoofed`.

## Stage 7. Docs

README rewritten with a Mermaid flowchart and class UML diagram,
`requirements.txt` pinned, this changelog added.
