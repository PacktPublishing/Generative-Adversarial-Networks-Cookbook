#!/usr/bin/env python3
from train import Trainer

# Command Line Argument Method
HEIGHT  = 64
WIDTH   = 64
CHANNEL = 3
EPOCHS = 100
BATCH = 1
CHECKPOINT = 100
TRAIN_PATH_A = "/data/summer2winter_yosemite/trainA/"
TRAIN_PATH_B = "/data/summer2winter_yosemite/trainB/"
TEST_PATH_A = "/data/summer2winter_yosemite/testA/"
TEST_PATH_B = "/data/summer2winter_yosemite/testB/"

trainer = Trainer(height=HEIGHT,width=WIDTH,epochs =EPOCHS,\
                 batch=BATCH,\
                 checkpoint=CHECKPOINT,\
                 train_data_path_A=TRAIN_PATH_A,\
                 train_data_path_B=TRAIN_PATH_B,\
                 test_data_path_A=TEST_PATH_A,\
                 test_data_path_B=TEST_PATH_B,\
                 lambda_cycle=10.0,\
                 lambda_id=1.0)
trainer.train()
