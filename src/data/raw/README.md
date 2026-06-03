# Raw data

Place the [OSCD](https://ieee-datacomp.labri.fr/oscd/) dataset here:

```
src/data/raw/onera-satellite-change-detection-dataset/
├── images/<city>/imgs_1_rect/   # B01.tif … B12.tif
├── images/<city>/imgs_2_rect/
└── train_labels/<city>/cm/<city>-cm.tif
```

Or set `UCDNET_DATA_ROOT` to an existing copy elsewhere on disk.

The dataset is large and is not tracked in git.
