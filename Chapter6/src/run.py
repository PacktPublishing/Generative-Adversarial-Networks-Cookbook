#!/usr/bin/env python3
from train import Trainer

# Command Line Argument Method
HEIGHT  = 256
WIDTH   = 256
CHANNELS = 3
EPOCHS = 100
BATCH = 1
CHECKPOINT = 50
TRAIN_PATH = "/data/cityscapes/cityscapes/train/"
TEST_PATH = "/data/cityscapes/cityscapes/val/"

trainer = Trainer(height=HEIGHT,width=WIDTH, channels=CHANNELS,epochs =EPOCHS,\
                 batch=BATCH,\
                 checkpoint=CHECKPOINT,\
                 train_data_path=TRAIN_PATH,\
                 test_data_path=TEST_PATH)
trainer.train()
