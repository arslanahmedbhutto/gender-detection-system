# Dataset

The face images are not included in this repository. The project uses a labelled face dataset of 2,279 images, split into training, validation and test sets:

| Split | Women | Men | Total |
|-------|------:|----:|------:|
| train | 799 | 800 | 1,599 |
| valid | 170 | 170 | 340 |
| test  | 170 | 170 | 340 |

Face datasets of this kind are listed in the [CVonline image database index](https://homepages.inf.ed.ac.uk/rbf/CVonline/Imagedbase.htm#face).

## Folder layout

Put the images here using this structure:

```
data/
└── dataset1/
    ├── train/
    │   ├── men/
    │   └── women/
    ├── valid/
    │   ├── men/
    │   └── women/
    └── test/
        ├── men/
        └── women/
```

If your images are stored somewhere else, set the `GENDER_DATA_DIR` environment variable to the folder that contains `train/`, `valid/` and `test/`.
