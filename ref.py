import os

cur_dir = os.path.dirname(os.path.abspath(__file__))
# rootDir = os.path.join(cur_dir, '..') # root directory

oriRes = 300
inputRes = 256  # vgg:224

eps = 1e-6

momentum = 0.0
weightDecay = 0.0
alpha = 0.99
epsilon = 1e-8

expDir = os.path.join(cur_dir, 'exp')
read_data_pth = '/home/aa/data/face_CAD_data/arranged image'
