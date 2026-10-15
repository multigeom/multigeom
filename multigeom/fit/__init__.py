"""
Fitting to the paper's recordings (extra `[paper]`: h5py, plenoptic).

* `train_encoder`: the paper's recipe for fitting an encoder to the macaque recordings.
  It holds the TVSD data around `NeuralEncoder.fit_readout`, which is the one place
  where a readout is fitted;
* the TVSD data loaders (`tvsd`), the observation-noise calibration and the backbone
  layer sweep.

Not needed to use an encoder that is already fitted.
"""
