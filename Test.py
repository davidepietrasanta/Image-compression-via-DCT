import scipy as sp
import numpy as np

from DCT import dct, dct2, naive_dtc2, idct, fix_inverse, idct2


#x = np.array([1.0, 2.0, 1.0, -1.0, 1.5])
#x = np.array([1.0, 2.0, 1.0])
x = np.array([231, 32, 233, 161, 24, 71, 140, 245])
#  4.01e+02 6.60e+00 1.09e+02 -1.12e+02 6.54e+01 1.21e+02 1.16e+02 2.88e+01

print("\n ARRAY:")
print(x)
print(" \n My DCT:")
print(dct(x))
print(" \n My IDCT:")
print(idct(dct(x)))




#print(naive_dtc2(matrix))

test_matrix = np.array ( [[ 231, 32, 233, 161, 24, 71, 140, 245],
                        [ 247, 40, 248, 245, 124, 204, 36, 107],
                        [ 234, 202, 245, 167, 9, 217, 239, 173],
                        [ 193, 190, 100, 167, 43, 180, 8, 70],
                        [ 11, 24, 210, 177, 81, 243, 8, 112],
                        [ 97, 195, 203, 47, 125, 114, 165, 181],
                        [ 193, 70, 174, 167, 41, 30, 127, 245],
                        [ 87, 149, 57, 192, 65, 129, 178, 228]])


print("\n Naive:")
print(naive_dtc2(test_matrix))
print("\n MY DCT2:")
print(dct2(test_matrix))
print("\n MY IDCT2:")
print(idct2( dct2(test_matrix) ))
