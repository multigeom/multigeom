# Data setup

The pipeline reads three external assets from the `data/` directory.
None are bundled with this submission; download instructions and the
expected layout are below.

```
data/
├── tvsd/
│   ├── monkeyN_THINGS_normMUA.mat
│   └── things_imgs.mat
├── things_images/
│   └── object_images/<class>/<image>.jpg
└── tiny-imagenet-200/
    ├── train/<class>/<image>.JPEG
    └── val/<class>/<image>.JPEG
```

## 1. Papale TVSD (macaque V1/V4 responses)

The THINGS Ventral-stream Spiking Dataset contains chronically-recorded
multi-unit activity from awake macaque V1 and V4 during presentation of
natural images from the THINGS image database. The dataset is
publicly released; cite the original publication and follow the
authors' download instructions (the dataset homepage links to a
data-portal download in HDF5 (`.mat`) format).

Place the resulting files in `data/tvsd/`:

- `monkeyN_THINGS_normMUA.mat` — main neural-response file used in the paper
- `things_imgs.mat` — image-path index used to align responses to image files

## 2. THINGS image database

The THINGS image database holds the actual stimulus images that the
TVSD responses are aligned to. Cite the original publication and
follow the authors' download instructions.

Place the unzipped folder under `data/things_images/object_images/<class>/<image>.jpg`,
matching the layout the dataset ships with.

## 3. TinyImageNet

Used for training the conditional diffusion model in Stage 3. The
`tiny-imagenet-200` archive can be downloaded from the official source
and unpacked under `data/tiny-imagenet-200/`. Both `train/` and `val/`
splits are required.

## 4. ON-OFF frontend pretrained weights

The Berardino 2017 ON-OFF frontend used by the `OnOff` encoder is shipped
inside the `plenoptic` Python package and downloaded automatically the
first time `plenoptic.simulate.models.frontend.OnOff(pretrained=True)` is
instantiated; no manual setup is required.

## Path overrides

By default `src/data_loaders.py` resolves these directories relative to
the project root (the parent of `data/`). To use external storage, set
the environment variable `NEURIPS_DATA_DIR` to override the default:

```bash
export NEURIPS_DATA_DIR=/path/to/storage
```

Alternatively, every stage script accepts `--data-dir <path>` to point at
an arbitrary directory.
