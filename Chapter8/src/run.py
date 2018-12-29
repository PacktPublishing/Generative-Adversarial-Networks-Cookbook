#!/usr/bin/env python3
from train import Trainer

# Command Line Argument Method
CUBE_SIDE=16
EPOCHS = 100000
BATCH = 64
CHECKPOINT = 10
LATENT_SPACE_SIZE = 256
DATA_DIR = "/3d-mnist/full_dataset_vectors.h5"

trainer = Trainer(side=CUBE_SIDE, \
                 latent_size=LATENT_SPACE_SIZE, \
                 epochs =EPOCHS,\
                 batch=BATCH,\
                 checkpoint=CHECKPOINT,\
                 data_dir = DATA_DIR)
trainer.train()