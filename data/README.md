# Data setup

The pipeline reads three external datasets. None are redistributed here;
download each from its source and place it under `data/` (or under the
directory named by `$NEURIPS_DATA_DIR`) with this layout:

```
data/
├── tvsd/
│   ├── monkeyN_THINGS_normMUA.mat
│   └── things_imgs.mat
├── things_images/
│   └── object_images/<concept>/<image>.jpg
└── tiny-imagenet-200/
    ├── train/<wnid>/images/<image>.JPEG
    └── val/images/<image>.JPEG
```

## 1. THINGS Ventral-stream Spiking Dataset (TVSD)

Multi-unit activity recorded with chronic Utah arrays in macaque V1, V4 and
IT while the animal viewed THINGS images (Papale et al., *Neuron*, 2025).
Download the files for monkey N following the instructions in the paper and
place them in `data/tvsd/`:

- `monkeyN_THINGS_normMUA.mat`: trial-averaged, normalised multi-unit
  activity (`train_MUA`, `test_MUA`), the 30 single-trial responses to each
  test image (`test_MUA_reps`, used by `analysis/calibrate_sigma.py`) and the
  per-electrode split-half reliability (`oracle`). Electrodes 0-511 are V1
  and 512-767 are V4.
- `things_imgs.mat`: the THINGS image path of every row of `train_MUA` and
  `test_MUA` (`train_imgs/things_path`, `test_imgs/things_path`).

## 2. THINGS image database

The stimulus images the TVSD responses are aligned to (Hebart et al., 2019).
Unzip the `object_images` folder to `data/things_images/object_images/`, so
that paths such as `aardvark/aardvark_01b.jpg` resolve. Images listed in
`things_imgs.mat` but absent on disk are skipped with a warning.

## 3. TinyImageNet

The 100,000 training images train the conditional diffusion model (stage 3);
the first 200 validation images, in sorted file order, are the held-out test
images on which G(x) and J(x) are evaluated (stages 4 and 6). Download
`tiny-imagenet-200.zip` (http://cs231n.stanford.edu/tiny-imagenet-200.zip) and
unzip it to `data/tiny-imagenet-200/`; the archive's own layout is the one
expected above.

## 4. Pretrained weights

- InceptionV3 and ResNet-50 ImageNet weights are downloaded by `torchvision`
  on first use (cached under `$TORCH_HOME`).
- The ON-OFF front-end of Berardino et al. (2017) ships with `plenoptic`
  (`OnOff(pretrained=True)`); nothing to download by hand.

## Path overrides

Set `NEURIPS_DATA_DIR=/path/to/storage` to read the datasets from elsewhere,
or pass `--data-dir <path>` to any stage script.
