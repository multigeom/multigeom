# Supporting analyses

Three scripts that reproduce numbers quoted in the appendices of the paper.
They are not part of the main pipeline (`multigeom.fit`, `multigeom.diffusion`, `multigeom.pipeline`) and
write their results as JSON files with a provenance record (git commit,
command line, configs, package versions).

| Script | Paper statement | Inputs | Cost |
|---|---|---|---|
| `calibrate_sigma.py` | σ = 0.22 (V1), 0.25 (V4): App. "Observation model for responses" | TVSD `monkeyN_THINGS_normMUA.mat`, `things_imgs.mat` | seconds, CPU |
| `estimator_variance.py` | CC has 80-95% lower Monte-Carlo variance than UC; UC/CC top eigenvalue 5-50x: App. "Conditional-pair estimator" | Stage-1 encoder and Stage-3 diffusion checkpoints, test images | minutes per image, GPU |
| `layer_sweep.py` | Conv2d_4a_3x3 maximises test correlation in V1 and V4: App. "InceptionV3 model" | TVSD files, THINGS images | about one Stage-1 training per layer, GPU |

## calibrate_sigma.py

`test_MUA_reps` holds the normalised MUA of the 30 repetitions of each of the
100 test images (h5py order: `[30, 100, 1024]`). For each channel the script
takes the root-mean-square, over trials and images, of the per-trial response
minus its trial average, and reports the median over channels.

The primary number uses the top-50 channels per area by oracle reliability
(the encoder targets, selected with `multigeom.data_loaders.load_tvsd`), as did the
original measurement. The median over all channels of the area is reported
alongside it.

    python -m multigeom.fit.calibrate_sigma --out runs/analysis/sigma_calibration.json

On the released TVSD file this gives 0.2275 (V1) and 0.2451 (V4) on the
top-50 channels, and 0.2517 (V1) and 0.2591 (V4) on all channels of the area.

## estimator_variance.py

Runs the conditional-pair (CC) estimator of `multigeom.estimator` and the
unconditional-conditional (UC) form on a few test images, several times each
with independent seeds, and compares the across-run variance of the top-5
eigenvalues.

The UC row is `sqrt(w_j / N_MC) * (eps(x_t, t, r) - eps(x_t, t, 0))`, with one
response draw per sample. The factor 1/2 of the CC row is absent because the
UC difference is centred on the null-token prediction, which equals the mean
of the conditional predictions when the models are correct. Both forms then
have the same expectation; any mismatch between the null-token prediction and
that mean adds a bias to UC only. The module docstring gives the derivation.

The null-token pass that gives the denoised image is reused as the UC
subtraction term, so a UC sample costs 2 UNet passes and a CC sample costs 3.
By default (`--compute-match passes`) UC gets 1.5 times as many samples, so
that both spend the same number of passes. `--compute-match samples` uses the
same sample count for both.

    python -m multigeom.pipeline.estimator_variance \
        --encoder-cfg inception \
        --encoder-ckpt runs/encoders/inception/V1/best_model.pt \
        --diffusion-ckpt runs/diffusion/inception/V1/final_model.pt \
        --n-images 6 --n-runs 6 --n-mc 100

The output reports both the raw-variance reduction and the reduction in
squared coefficient of variation (variance divided by squared mean). The raw
variance also reflects the larger overall scale of the UC estimate, while the
coefficient of variation does not.

## layer_sweep.py

Fits the Stage-1 readout on top of the backbone truncated after each
candidate layer, with everything else fixed (input conversion, readout,
optimiser, schedule, validation split and seed from the encoder config), and
reports validation and test correlations per layer.

    python -m multigeom.fit.layer_sweep --backbone inception --area V1
    python -m multigeom.fit.layer_sweep --backbone resnet50 --area V4

For a 64x64 input the Inception feature map shrinks to 2x2 at Mixed_6a, and
Mixed_7a, Mixed_7b and Mixed_7c cannot be evaluated; they are recorded as
skipped. Each layer is extracted separately from images decoded once and held
in memory.

`--random-weights`, `--max-train-images` and `--epochs` exist for smoke tests
only.
