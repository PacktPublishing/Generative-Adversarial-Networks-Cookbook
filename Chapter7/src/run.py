#!/usr/bin/env python3
from train import Trainer

# Command Line Argument Method
HEIGHT  = 55
WIDTH   = 33
CHANNELS = 1
EPOCHS = 100
BATCH = 16
CHECKPOINT = 50
SIM_PATH = "/data/eye-gaze/original/"
REAL_PATH = "/data/eye-gaze/MPII/Data/Original/"

trainer = Trainer(height=HEIGHT,width=WIDTH, channels=CHANNELS,epochs =EPOCHS,\
                 batch=BATCH,\
                 checkpoint=CHECKPOINT,\
                 sim_path=SIM_PATH,\
                 real_path=REAL_PATH)
trainer.train()