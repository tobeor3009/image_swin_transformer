import os
gpu_number = "0"
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_number

import tensorflow as tf
from tensorflow.keras import backend as keras_backend
import numpy as np

gpu_on = True

if gpu_on :
    gpu_devices = tf.config.experimental.list_physical_devices("GPU")
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    gpu_devices = tf.config.experimental.list_physical_devices("CPU")

print(gpu_devices)

import sys
from glob import glob
import numpy as np
sys.path.append("../../../../CNN_total/")

import json

def read_json_as_dict(json_path):
    json_file = open(json_path, encoding="utf-8")
    json_str = json_file.read()
    json_dict = json.loads(json_str)
    
    return json_dict

def is_in_condition(str_object):
    is_in = False
    for phase in ["no_person", "single_person"]:
        if phase in str_object:
            is_in = True
            break
    return is_in

from src.data_loader.pose_estimation import PoseDataloader
from glob import glob

task = "classification"
data_set_name = "detect_lvi"
batch_size = 32
num_workers = 12
on_memory = False
augmentation_proba = 0.625
target_size = (256, 256)
interpolation = "bilinear"
# class_mode = "binary"
class_mode = "categorical"
dtype="float32"

train_image_path_list = glob(f"../data/2. Split Data/train/*/*")
valid_image_path_list = glob(f"../data/2. Split Data/valid/*/*")
test_image_path_list = glob(f"../data/2. Split Data/test/*/*")

train_image_path_list = list(filter(is_in_condition, train_image_path_list))
valid_image_path_list = list(filter(is_in_condition, valid_image_path_list))
test_image_path_list = list(filter(is_in_condition, test_image_path_list))

train_valid_annotation_dict = read_json_as_dict("../data/2. Split Data/single_person_keypoints_train2017.json")
test_annotation_dict = read_json_as_dict("../data/2. Split Data/single_person_keypoints_val2017.json")

augmentation_policy_dict = {
    "positional": False,
    "noise": True,
    "elastic": False,
    "randomcrop": False,
    "brightness_contrast": False,
    "color": False,
    "to_jpeg": False
}

common_arg_dict = {
    "augmentation_policy_dict": augmentation_policy_dict,
    "preprocess_input": "-1~1",
    "target_size": target_size,
    "interpolation": interpolation,
    "dtype": dtype
}

train_data_loader = PoseDataloader(image_path_list=train_image_path_list,
                                   total_annotation_dict=train_valid_annotation_dict,
                                       batch_size=batch_size,
                                       num_workers=num_workers,
                                       on_memory=on_memory,
                                       augmentation_proba=augmentation_proba,
                                       shuffle=True,
                                       **common_arg_dict
)
valid_data_loader = PoseDataloader(image_path_list=valid_image_path_list,
                                   total_annotation_dict=train_valid_annotation_dict,
                                       batch_size=batch_size,
                                       num_workers=1,
                                       on_memory=on_memory,
                                       augmentation_proba=0,
                                       shuffle=True,
                                       **common_arg_dict
)
test_data_loader = PoseDataloader(image_path_list=test_image_path_list,
                                   total_annotation_dict=test_annotation_dict,
                                       batch_size=batch_size,
                                       num_workers=1,
                                       on_memory=False,
                                       augmentation_proba=0,
                                       shuffle=False,
                                       **common_arg_dict
)

from src.model.vision_transformer.pose_estimation import get_swin_pose_estimation_2d

input_shape = (256, 256, 3)
last_channel_num = 11
filter_num_begin = 128     # number of channels in the first downsampling block; it is also the number of embedded dimensions
depth = 3                  # the depth of SwinUNET; depth=4 means three down/upsampling levels and a bottom level 
stack_num_per_depth = 2         # number of Swin Transformers per downsampling level
patch_size = (4, 4)        # Extract 4-by-4 patches from the input image. Height and width of the patch must be equal.
stride_mode = "same"
num_heads = [4, 8, 8, 8]   # number of attention heads per down/upsampling level
window_size = [4, 2, 2, 2] # the size of attention window per down/upsampling level
num_mlp = 256              # number of MLP nodes within the Transformer
act = "gelu"
last_act = "sigmoid"
shift_window = True          # Apply window shifting, i.e., Swin-MSA
swin_v2 = True
model = get_swin_pose_estimation_2d(input_shape, 
                                  filter_num_begin, depth, stack_num_per_depth,
                                  patch_size, stride_mode, num_heads, window_size, num_mlp,
                                  act=act, last_act=last_act, shift_window=shift_window, 
                                   swin_v2=swin_v2)
print(f"model param num: {model.count_params()}")
print(f"model input shape: {model.input}")
print(f"model output shape: {model.output}")

from datetime import date

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.losses import MeanAbsoluteError

today = date.today()

# YY/MM/dd
today_str = today.strftime("%Y-%m-%d")
today_weight_path = f"./result_daily/{task}/{data_set_name}/{today_str}/gpu_{gpu_number}/target_size_{target_size}/weights/" 
today_image_path = f"./result_daily/{task}/{data_set_name}/{today_str}/gpu_{gpu_number}/target_size_{target_size}/images/"
today_logs_path = f"./result_daily/{task}/{data_set_name}/{today_str}/gpu_{gpu_number}/target_size_{target_size}/"
os.makedirs(today_weight_path, exist_ok=True)
os.makedirs(today_logs_path, exist_ok=True)
optimizer = Nadam(1e-4, clipnorm=1)

save_c = ModelCheckpoint(
    today_weight_path+"/weights_{val_loss:.4f}_{loss:.4f}_{epoch:02d}.hdf5",
    monitor='val_loss',
    verbose=0,
    save_best_only=False,
    save_weights_only=True,
    mode='min')


def scheduler(epoch, lr):
    if epoch <= 40:
        new_lr = 2e-5
    elif epoch <= 100:
        new_lr = 2e-4
    elif epoch <= 200:
        new_lr = 2e-5
    else:
        new_lr = 2e-6
    return new_lr
scheduler_callback = LearningRateScheduler(scheduler, verbose=1)
csv_logger = CSVLogger(f'{today_logs_path}/log.csv', append=False, separator=',')
loss_function = MeanAbsoluteError()

model.compile(optimizer=optimizer, loss=loss_function, metrics=['mae'])

start_epoch = 0
epochs = 300

model.fit(
    train_data_loader,
    validation_data=valid_data_loader,
    epochs=epochs,
    steps_per_epoch=len(train_data_loader),
    validation_steps=len(valid_data_loader),    
    callbacks=[scheduler_callback, save_c, csv_logger], 
    initial_epoch=start_epoch
)