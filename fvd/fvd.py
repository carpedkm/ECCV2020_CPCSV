# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Example code that computes FVD for some empty frames.
The FVD for this setup should be around 131.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v1 as tf
#from frechet_video_distance import frechet_video_distance as fvd
from .frechet_video_distance import calculate_fvd, create_id3_embedding, preprocess

import torch.utils.data
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torchvision
import functools
import PIL
import re
import pdb
import argparse
from tqdm import tqdm
from .loader import VideoGenerateDataset
import random
import os
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '0'
from story_fid_model import VideoResNet

# Number of videos must be divisible by 16.
#NUMBER_OF_VIDEOS = 320
#VIDEO_LENGTH = 9

def calculate_fvd_from_inference_result(gen_path, ref_path='./Evaluation/ref', num_of_video=16, video_length=10):
  # fsd setting

  fsd = True
  cuda_enable = False

  VIDEO_LENGTH = video_length
  if fsd == True:
    VIDEO_LENGTH = 5
  print('{}'.format(video_length))
  base_ref = VideoGenerateDataset(ref_path, min_len=VIDEO_LENGTH)
  base_tar = VideoGenerateDataset(gen_path, min_len=VIDEO_LENGTH)

  bs = num_of_video
  bs = 64
  assert bs%16 == 0

  videoloader_ref = torch.utils.data.DataLoader( # evaluate with only the first 272 clips? 
    base_ref, batch_size=bs,  #len(videodataset),
    drop_last=True, shuffle=False)
  videoloader_tar = torch.utils.data.DataLoader(
    base_tar, batch_size=bs,  #len(videodataset),
    drop_last=True, shuffle=False)
  # images_ref = []
  # images_tar = []
  with tqdm(total=len(videoloader_ref), dynamic_ncols=True) as pbar:
    
    for i, data in enumerate(videoloader_ref):
      # images_ref.append(data.numpy())
      images_ref = data.numpy()
      break # if you want to 
    
    for i, data in enumerate(videoloader_tar):
      # images_tar.append(data.numpy())
      images_tar = data.numpy()
      
      break # if you want to -> you should delete .append -> in order to make it not list
  print('##### Batch complete #####')
  # images_ref = np.concatenate(images_ref, axis=0)
  # images_tar = np.concatenate(images_tar, axis=0)
  if fsd != True :
    # ref_tf = tf.convert_to_tensor(images_ref, dtype=tf.uint8)
    # tar_tf = tf.convert_to_tensor(images_tar, dtype=tf.uint8)
    # i3d_ref = create_id3_embedding(preprocess(ref_tf, (224, 224)), bs)
    # i3d_gen = create_id3_embedding(preprocess(gen_tf, (224, 224)), bs)
    
    # result = calculate_fvd(i3d_ref, i3d_gen)
    # return result
    with tf.Graph().as_default():
      ref_tf = tf.convert_to_tensor(images_ref, dtype=tf.uint8)
      tar_tf = tf.convert_to_tensor(images_tar, dtype=tf.uint8)

      first_set_of_videos = ref_tf #14592
      second_set_of_videos = tar_tf
      i3d_ref = create_id3_embedding(preprocess(first_set_of_videos,
                                                  (224, 224)), bs)
      i3d_gen = create_id3_embedding(preprocess(second_set_of_videos,
                                                  (224, 224)), bs)

      with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())
        temp = sess.run(i3d_ref)
        temp_2 = sess.run(i3d_gen)
    result = calculate_fvd(
    tf.convert_to_tensor(temp, dtype=tf.float32), tf.convert_to_tensor(temp_2, dtype=tf.float32)
    )
    return float(result)
  else:
    images_ref = preprocess(tf.convert_to_tensor(images_ref),
                                                  (224, 224))
    images_gen = preprocess(tf.convert_to_tensor(images_tar), (224, 224))
    
    images_ref = torch.Tensor(images_ref.numpy())
    images_gen = torch.Tensor(images_gen.numpy())
    
    images_ref = torch.Tensor(images_ref).permute(0, 4, 1, 2, 3)
    images_gen = torch.Tensor(images_gen).permute(0, 4, 1, 2, 3)
    
    
    print('>>>>> TF Tensor made <<<<<')
    model = torchvision.models.video.r2plus1d_18(pretrained=True, progress=True)
    print('##### Making R (2+1) D embedding #####')
    if cuda_enable == True:
      torch.device('cuda')
      to_cuda_ref = torch.Tensor(images_ref).cuda()
      to_cuda_gen = torch.Tensor(images_gen).cuda()
      model = model.cuda() # i think i should deal with it by making it to batch
      r2plus1d_ref = model(to_cuda_ref) # should do preprocess first ? to match with i3d
      r2plus1d_gen = model(to_cuda_gen)
      
      print('##### Embedding complete #####')
      tf_ref = tf.convert_to_tensor(r2plus1d_ref.detach().numpy(), dtype=tf.float32)
      tf_gen = tf.convert_to_tensor(r2plus1d_gen.detach().numpy(), dtype=tf.float32)
      print('##### CALC FSD #####')
      # with tf.Graph().as_default():
        
      result = calculate_fvd(
          tf_ref, tf_gen
          )
        # print('##### FSD Calculation complete #####')
        # with tf.Session() as sess:
        #   print('##### Run TF session #####')
        #   sess.run(tf.global_variables_initializer())
        #   sess.run(tf.tables_initializer())
        #   return sess.run(result)
      return float(result)
    else :
      r2plus1d_ref = model(torch.Tensor(images_ref))
      r2plus1d_gen = model(torch.Tensor(images_gen))
      
      tf_ref = tf.convert_to_tensor(r2plus1d_ref.detach().numpy(), dtype=tf.float32)
      tf_gen = tf.convert_to_tensor(r2plus1d_gen.detach().numpy(), dtype=tf.float32)
      
      result = calculate_fvd(
          tf_ref, tf_gen
          )
      return float(result)
  # else :
  #   VIDEO_LENGTH = 5 # for the case of pororo dataset
  #   print('{}'.format(video_length))
  #   base_ref = VideoGenerateDataset(ref_path, min_len=VIDEO_LENGTH)
  #   base_tar = VideoGenerateDataset(gen_path, min_len=VIDEO_LENGTH)
  #   bs = 272
  #   assert bs%16 == 0

  #   videoloader_ref = torch.utils.data.DataLoader( # evaluate with only the first 272 clips? 
  #     base_ref, batch_size=bs,  #len(videodataset),
  #     drop_last=True, shuffle=False)
  #   videoloader_tar = torch.utils.data.DataLoader(
  #     base_tar, batch_size=bs,  #len(videodataset),
  #     drop_last=True, shuffle=False)
  #   # images_ref = []
  #   # images_tar = []
  #   with tqdm(total=len(videoloader_ref), dynamic_ncols=True) as pbar:
      
  #     for i, data in enumerate(videoloader_ref):
  #       # images_ref.append(data.numpy())
  #       images_ref = data.numpy()
  #       break # if you want to 
      
  #     for i, data in enumerate(videoloader_tar):
  #       # images_tar.append(data.numpy())
  #       images_tar = data.numpy()
  #       break # if you want to -> you should delete .append -> in order to make it not list
  #   # images_ref = np.concatenate(images_ref, axis=0)
  #   # images_tar = np.concatenate(images_tar, axis=0)

  #   model = VideoResNet(BasicBlock, )