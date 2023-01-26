import imageio
import os

directory = r'C:\Users\enter1\Desktop\chang\galaxychang\test'
file_type = r'png'
save_gif_name = r'Galactic_interation'
time = 0.01
speed_sec = { 'duration':time}

images = []
for file_name in os.listdir(directory):
    if file_name.endswith('.{}'.format(file_type)):
        file_path = os.path.join(directory, file_name)
        images.append(imageio.imread(file_path))

imageio.mimsave('{}/{}.gif'.format(directory, save_gif_name), images, **speed_sec)
print("Complete:")