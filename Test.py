import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import time

from DCT import test_time, plot_times, random_matrix,test_time1, unit_test
from scipy.fftpack import dct as scipy_dct

unit_test()
path = str(pathlib.Path(__file__).parent.absolute())



matrix = random_matrix(300)
start = time.time()
k = scipy_dct(scipy_dct(matrix.T, norm='ortho').T, norm='ortho')
end = time.time()
print(end-start)
print(k)

df = test_time1(300, 340, 10)
print(df)
plot_times(df, path=path+"\\figura.png", naive=False, log=True)