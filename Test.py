#Parte 1

import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import time

from Parte1 import test_time, plot_times, random_matrix, unit_test
from scipy.fftpack import dct as scipy_dct

unit_test()
path = str(pathlib.Path(__file__).parent.absolute())


min = 10
max = 2510
step = 500

matrix = random_matrix(max)
start = time.time()
k = scipy_dct(scipy_dct(matrix.T, norm='ortho').T, norm='ortho')
end = time.time()
print(end-start)

df = test_time(min, max, step)
print(df)
plot_times(df, path=path+"\\figura.png", naive=False, log=True)