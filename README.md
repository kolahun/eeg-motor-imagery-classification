# EEG Motor Imagery Classification — Preprocessing & Deep Learning Comparison

Comparing deep learning model performance **with vs without preprocessing** on the BCI Competition IV 2a motor imagery EEG dataset. Preprocessing pipeline: bandpass filtering → notch filtering → ICA-based artifact removal.

## Dataset

[BCI Competition IV 2a](https://www.bbci.de/competition/iv/#dataset2a) — 4-class motor imagery (left hand, right hand, feet, tongue), 9 subjects × 2 sessions (train/eval), 22 EEG + 3 EOG channels, 250 Hz.

> Dataset is not included in this repo. Download it from the link above, or use the preprocessed version on Kaggle: *[link to be added]*

## Pipeline

1. **Load** raw `.gdf` files (MNE)
2. **Channel renaming** — generic GDF labels mapped to standard 10-20 names
3. **Montage** — standard 10-20 applied
4. **Bandpass filter** — 4–38 Hz (covers mu/beta bands relevant to motor imagery ERD/ERS)
5. **Notch filter** — 50 Hz (powerline noise)
6. **ICA** — automatic EOG-correlated artifact component detection and removal

All 18 subject/session files processed successfully. Cleaned output saved as `.fif` (MNE native format).

## Status

- [x] Preprocessing pipeline built and validated (bandpass + notch + ICA)
- [x] All 18 files processed
- [ ] Epoching (trial segmentation around motor imagery cues)
- [ ] Raw (no-preprocessing) track for comparison
- [ ] Deep learning model(s) — candidates: EEGNet, ShallowConvNet, EEG-Conformer
- [ ] Training + evaluation (accuracy, Cohen's kappa)
- [ ] Results comparison: preprocessed vs raw, across model(s)

## Requirements

```
mne
numpy
matplotlib
scikit-learn
```

## Notes

Preprocessing parameters (bandpass range, notch frequency, ICA component count) are configurable via function arguments, not hardcoded.
