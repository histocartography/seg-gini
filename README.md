# SegGini: 
### SEGmentation using Graphs with Inexact aNd Incomplete labels 

This repository contains the code to reproduce the results of ["Learning Whole-Slide Segmentation from Inexact and Incomplete Labels using Tissue Graphs"](https://arxiv.org/pdf/2103.03129.pdf), MICCAI, 2021. 

The code is built on the [`Histocartography`](https://github.com/histocartography/histocartography) library, a python-based package that facilitates the modelling and learning of pathology images as graphs. 

The described experiments are presented for the [`SICAPv2`](https://data.mendeley.com/datasets/9xxm58dvs3/1) dataset, a cohort of Hematoxylin and Eosin (H&amp;E) stained prostate needle biopsies. 

![Overview of the proposed approach.](figs/overview.png)

## Installation 

### Cloning and handling dependencies 

Clone the repo:

```
git clone https://github.com/histocartography/seg-gini.git && cd seg-gini
```

Create a conda environment and activate it:

```
conda env create -f environment.yml
conda activate seggini
```

### Downloading and preparing the SICAPv2 dataset 

SICAPv2 is a database containing Hematoxylin and Eosin (H&E) stained patches (512x512) from 155 prostate whole-slide images across 95 patients. The dataset contains local patch-level segmentation masks for Gleason patterns (Non cancerous, Grade3, Grade4, Grade5) and global Gleason scores (Primary + Secondary).  

The SICAPv2 dataset can be downloaded, and the whole-slide images and corresponding Gleason pattern segmentation masks can be recreated by running:

```
cd bin
python create_sicap_data.py
```


![Overview of the dataset.](figs/overview.png)
  

## Running the code 

The proposed HACT-Net architecture operates on a HieArchical Cell-to-Tissue representation that is further processed by a Graph Neural Network. Running HACT-Net requires 2 steps:

### Step 1: HieArchical Cell-to-Tissue (HACT) generation 

The HACT representation can be generated for the `train` set by running: 

```
cd core
python generate_hact_graphs.py --image_path <PATH-TO-BRACS>/BRACS/train/ --save_path <SOME-SAVE-PATH>/hact-net-data
```

For generating HACT on the `test` and `val` set, simply replace the `image_path` by `<PATH-TO-BRACS>/BRACS/val/` or `<PATH-TO-BRACS>/BRACS/test/`. 

The script will automatically create three directories containing for each image:
- a cell graph as a `.bin` file
- a tissue graph as a `.bin` file
- an assignment matrix as an `.h5` file

After the generation of HACT graphs on the whole BRACS set, the `hact-net-data` dir should look like:

```
hact-net-data
|
|__ cell_graphs 
    |
    |__ train
    |
    |__ test
    |
    |__ val
|
|__ tissue_graphs
    |
    |__ train
    |
    |__ test
    |
    |__ val
|
|__ assignment_matrices 
    |
    |__ train
    |
    |__ test
    |
    |__ val
```

### Step 2: Training HACTNet 

We provide the option to train 3 types of models, namely a Cell Graph model, Tissue Graph model and HACTNet model. 


Training HACTNet as:

```
python train.py --cg_path <SOME-SAVE-PATH>/hact-net-data/cell_graphs/ --tg_path <SOME-SAVE-PATH>/hact-net-data/tissue_graphs/ --assign_mat_path <SOME-SAVE-PATH>/hact-net-data/assignment_matrices/  --config_fpath ../data/config/hact_bracs_hactnet_7_classes_pna.yml -b 8 --in_ram --epochs 60 -l 0.0005 
```


Training a Cell Graph model as:

```
python train.py --cg_path <SOME-SAVE-PATH>/hact-net-data/cell_graphs/ --config_fpath ../data/config/cg_bracs_cggnn_7_classes_pna.yml -b 8 --in_ram --epochs 60 -l 0.0005 

```

Training a Tissue Graph model as:

```
python train.py --tg_path <SOME-SAVE-PATH>/hact-net-data/tissue_graphs/ --config_fpath ../data/config/tg_bracs_tggnn_7_classes_pna.yml -b 8 --in_ram --epochs 60 -l 0.0005 

```

Usage is:

```
usage: train.py [-h] [--cg_path CG_PATH] [--tg_path TG_PATH]
                [--assign_mat_path ASSIGN_MAT_PATH] [-conf CONFIG_FPATH]
                [--model_path MODEL_PATH] [--in_ram] [-b BATCH_SIZE]
                [--epochs EPOCHS] [-l LEARNING_RATE] [--out_path OUT_PATH]
                [--logger LOGGER]

optional arguments:
  -h, --help            show this help message and exit
  --cg_path CG_PATH     path to the cell graphs.
  --tg_path TG_PATH     path to tissue graphs.
  --assign_mat_path ASSIGN_MAT_PATH
                        path to the assignment matrices.
  -conf CONFIG_FPATH, --config_fpath CONFIG_FPATH
                        path to the config file.
  --model_path MODEL_PATH
                        path to where the model is saved.
  --in_ram              if the data should be stored in RAM.
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        batch size.
  --epochs EPOCHS       epochs.
  -l LEARNING_RATE, --learning_rate LEARNING_RATE
                        learning rate.
  --out_path OUT_PATH   path to where the output data are saved (currently
                        only for the interpretability).
  --logger LOGGER       Logger type. Options are "mlflow" or "none"
```

The output of this script will be a directory containing three models corresponding to the best validation loss, validation accuracy and weighted F1-score. 

### (Step 3: Inference on HACTNet)

We also provide a script for running inference with the option to use a pretrained model.

For instance, running inference with a pretrained HACTNet model: 

```
python inference.py --cg_path <SOME-SAVE-PATH>/hact-net-data/cell_graphs/ --tg_path <SOME-SAVE-PATH>/hact-net-data/tissue_graphs/ --assign_mat_path <SOME-SAVE-PATH>/hact-net-data/assignment_matrices/  --config_fpath ../data/config/hact_bracs_hactnet_7_classes_pna.yml --pretrained
```

We provide 3 pretrained checkpoints performing as:

| Model | Accuracy | Weighted F1-score |
| ----- |:--------:|:-----------------:|
| Cell Graph Model   | 58.1 | 56.7 |
| Tissue Graph Model | 58.6 | 57.8 |
| HACTNet Model      | 61.7   | 61.5 |


If you use this code, please consider citing our work:

```
@inproceedings{pati2021,
    title = "Hierarchical Graph Representations in Digital Pathology",
    author = "Pushpak Pati, Guillaume Jaume, Antonio Foncubierta, Florinda Feroce, Anna Maria Anniciello, Giosuè Scognamiglio, Nadia Brancati, Maryse Fiche, Estelle Dubruc, Daniel Riccio, Maurizio Di Bonito, Giuseppe De Pietro, Gerardo Botti, Jean-Philippe Thiran, Maria Frucci, Orcun Goksel, Maria Gabrani",
    booktitle = "arXiv",
    url = "https://arxiv.org/abs/2102.11057",
    year = "2021"
} 
```

