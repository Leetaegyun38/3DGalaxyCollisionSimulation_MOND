import numpy as np 
import matplotlib.pyplot as plt 
import imageio
import os
import random
from tqdm import tqdm

### 이미지 gif 변환
directory = r'C:\Users\enter1\Desktop\chang\galaxychang\singlegalaxy_image'
file_type = r'png'
save_gif_name = r'singlegalaxy'
time = 0.01
speed_sec = { 'duration':time}
images = []
for file_name in os.listdir(directory):
    if file_name.endswith('.{}'.format(file_type)):
        file_path = os.path.join(directory, file_name)
        images.append(imageio.imread(file_path))
imageio.mimsave('{}/{}.gif'.format(directory, save_gif_name), images, **speed_sec)

directory = r'C:\Users\enter1\Desktop\chang\galaxychang\singlegalaxy_velocity'
file_type = r'png'
save_gif_name = r'velocity'
time = 0.01
speed_sec = { 'duration':time}
images = []
for file_name in os.listdir(directory):
    if file_name.endswith('.{}'.format(file_type)):
        file_path = os.path.join(directory, file_name)
        images.append(imageio.imread(file_path))
imageio.mimsave('{}/{}.gif'.format(directory, save_gif_name), images, **speed_sec)

print("Complete:")