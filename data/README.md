# Dataset Download

To download the SLAKE dataset, use the following command:

```bash
# Download SLAKE dataset from HF Hub, make sure you have uv installed globally
uvx hf download BoKelvin/SLAKE --repo-type=dataset --local-dir ./data/SLAKE

# Unzip imgs.zip & KG.zip or manually unzip them
unzip ./data/SLAKE/imgs.zip -d ./data/SLAKE/
unzip ./data/SLAKE/KG.zip -d ./data/SLAKE/

# Delete the zip files to save space
rm ./data/SLAKE/imgs.zip
rm ./data/SLAKE/KG.zip
```

## Dataset Structure

After downloading and extracting, the dataset should have the following structure:

```bash
woa7015-medvqa/
└── data/
    └── SLAKE/
        ├── imgs/                      # Image folders (each case is one subfolder)
        │   ├── xmlab0/
        │   │   ├── detection.json
        │   │   ├── mask.png
        │   │   ├── question.json
        │   │   └── source.jpg
        │   ├── xmlab1/
        │   ├── ...
        │   └── xmlab641/
        │
        ├── KG/                        # Knowledge graph files (not used in this project)
        ├── train.json                 # Training split annotations
        ├── validation.json            # Validation split annotations
        ├── test.json                  # Test split annotations
        │
        ├── README.md
        ├── .gitattributes             # Not required for training/evaluation
        └── mask.txt                   # Not required for training/evaluation

```
