import scipy as sp
import numpy as np
from scipy.fftpack import dct as scipy_dct

import matplotlib.pyplot as plt
from matplotlib.pyplot import loglog as loglog
import time
from scipy.linalg import norm

def normalize(v):
    """
        normalize a one dimension np.array.
        v: Is a np.array of one dimension
        return: a np.array witch modelu is one
    """
    norm = np.linalg.norm(v)
    normal_array = v/norm
    return normal_array


def dct(v):
    """
        DCT for one dimension np.array.
        v: Is a np.array of one dimension [1:N]
        return: Discrete cosine transform of v
    """
    N = len(v)
    c = np.zeros(N) #[0:N-1]

    for k in range(0,N):
        if k == 0:
            a_k = 1/np.sqrt(N)
        else:
            a_k = np.sqrt(2)/np.sqrt(N)

        sum = 0
        for j in range(0,N):
            sum = sum + ( np.cos(k * np.pi * (2*(j+1) - 1)/(2*N)) * v[j] ) 

        c[k] = a_k * sum

    return c


def dct2(matrix):
    """
        DCT for two dimension np.array.
        matrix: Is a np.array of one dimension [M:N]
        return: Discrete cosine transform of matrix
    """
    N = len(matrix[0])
    M = len(matrix) 
    matrix_r = np.zeros( shape=(M, N) , dtype=np.float64 ) #To store the discrete cosine transform
    for i in range(0,M):
        matrix_r[i] = dct(matrix[i])

    for j in range(0, N):
        temp = dct(matrix_r[:,j])
        for k in range(0, M):
            matrix_r[k,j] = temp[k]

    return matrix_r


def naive_dct2(matrix):
    """
        Naive DCT for two dimension np.array.
        matrix: Is a np.array of one dimension [M:N]
        return: Discrete cosine transform of matrix
    """
    
    N = len(matrix[0])
    M = len(matrix) 
    matrix_r = np.zeros( shape=(M, N) , dtype=np.float64 ) #To store the discrete cosine transform


    for i in range(0,M):
        for j in range(0, N):
 
            if (i == 0):
                ai = 1 / np.sqrt(M)
            else:
                ai = np.sqrt(2)/np.sqrt(M)
            
            if (j == 0):
                aj = 1 / np.sqrt(M)
            else:
                aj = np.sqrt(2)/np.sqrt(M)
 
            w = 0
            for k in range(0,M):
                for l in range(0,N):
                    sum = matrix[k][l] * np.cos((2 * k + 1) * i * np.pi / (2 * M)) * np.cos((2 * l + 1) * j * np.pi / (2 * N))
                    w = w + sum

            matrix_r[i][j] = ai * aj * w
            #Da normalizzare

    return matrix_r


def idct(c):
    """
        IDCT for one dimension np.array.
        c: Is a np.array of one dimension [0:N-1]
        return: Inverse discrete cosine transform of c
    """
    N = len(c)
    v = np.zeros(N)
    for j in range(0,N):
        sum = 0
        for k in range(0,N):
            if k == 0:
                a_k = 1/np.sqrt(N)
            else:
                a_k = np.sqrt(2)/np.sqrt(N)

            sum = sum +  ( np.cos(k * np.pi * (2*(j+1) - 1)/(2*N)) * c[k] * a_k)

        v[j] = sum

    return v


def idct2(matrix):
    """
        IDCT for two dimension np.array.
        matrix: Is a np.array of one dimension [M:N]
        return: Inverse discrete cosine transform of matrix
    """
    N = len(matrix[0])
    M = len(matrix) 
    matrix_r = np.zeros( shape=(M, N) , dtype=np.float64 ) #To store the discrete cosine transform
    for i in range(0,M):
        matrix_r[i] = idct(matrix[i])

    for j in range(0, N):
        temp = idct(matrix_r[:,j])
        for k in range(0, M):
            matrix_r[k,j] = temp[k]

    return matrix_r


def is_integer_num(n):
    """
        Check if a number is an integer or not.
        n: number
        return: [boolean] True if the number is integer, False if it is decimal
    """
    if isinstance(n, int):
        return True
    if isinstance(n, float):
        return n.is_integer()
    return False


def fix_inverse(v):
    """
        While compressing idtc can produce floating numbers.
        We need to fix that, returning value in [0, 255]
        v: Is a np.array of one dimension [0:N-1]
        return: A np.array of one dimension [0:N-1]
    """
    for x in v:
        if x > 255:
            x = 255

        if x < 0:
            x = 0

        if not is_integer_num(x):
            x = round(x)
    
    return v


def random_matrix(N):
    """
        Creare a random matrix NxN
        N: integer number
        return: A np.array matrix of NxN random elements
    """
    matrix = np.random.randint(low=0, high=256, size= (N, N) )
    return matrix


def test_time(N, naive=True):
    """
        Return time for different test matrices.
        The tested matrices are multiple of 10 (dimension).
        N: number of Matrix tested
        naive: If True calculate also the naive_dct2 time
        return: df of different times recorded by functions
    """
    times = np.zeros( shape=(3, N) )
    x = np.zeros(shape= N) 
    for i in range(0, N):
        x[i] = (i+1) * 10
        matrix = random_matrix( 10 * (i+1)  )

        time.time()
        start = time.time()
        #scipy_dct(scipy_dct(matrix.T, norm='ortho').T, norm='ortho')
        scipy_dct(matrix, shape=(N, N))
        end = time.time()
        times[0, i] = end - start
        print(times[0, i])

        start = time.time()
        dct2(matrix)
        end = time.time()
        times[1, i] = end - start
        

        if(naive):
            start = time.time()
            naive_dct2(matrix)
            end = time.time()
            times[2, i] = end - start

    
    if(naive):
        df = {'time_lib_dct':times[0], 'time_dct2':times[1], 'time_naive_dct2':times[2], 'x_axis':x}
    else:
        df = {'time_lib_dct':times[0], 'time_dct2':times[1], 'x_axis':x}

    return df


def test_time1(min, max, step):
    times_my = []
    times_scipy = []
    x = []

    for n in range(min, max+step, step):
        matrix = np.random.randint(0, 256, (n,n) )

        start_time = time.time()
        dct2(matrix)
        times_my.append( time.time() - start_time)

        start_time = time.time()
        k = scipy_dct(scipy_dct(matrix.T, norm='ortho').T, norm='ortho')
        times_scipy.append( time.time() - start_time) #( ( Decimal(time.time()) - Decimal(start_time)) ) 

        x.append(n)
        
        print(times_scipy)
        #print(k)

    df = {'time_lib_dct': np.array(times_scipy), 'time_dct2': np.array(times_my), 'x_axis': np.array(x)}
    return df
    

def plot_times(df, naive=True, path=None, log=True):
    """
        Plot time for different test matrices.
        df: of different times recorded by functions,
            should be the output of test_time function
        naive: If True plot also the naive_dct2 time
            only if test_time function was runned with naive=True
        path: path where to save the plot, if None doesn't save.
        log: If True plot in log-log scale.
    """

    x = df['x_axis']

    if(log):
        my_plot = loglog
    else:
        my_plot = plt.plot


    #Plot scipy-library dct
    my_plot(x, df['time_lib_dct'], color ='r')

    #Plot my dct2
    my_plot(x, df['time_dct2'], color='b')
    
    if(naive):
        #Plot my naive_dct2
        my_plot(x, df['time_naive_dct2'], color='g')

    #Asintotic time
    my_plot(x, (x**2), '--', color='r') 
    my_plot(x, (x**3), '--', color='b')

    if(naive):
        my_plot(x, (x**4), '--', color='g') 
        plt.legend(["Scipy DCT2",  "My DCT2", "My Naive_DCT2", "x^2", "x^3", "x^4"])
    else:
        plt.legend(["Scipy DCT2",  "My DCT2", "x^2", "x^3"])

    if path is not None:
        plt.savefig(path)

    plt.title("Comparison between different DTC")
    plt.xlabel("N")
    plt.ylabel("Time")
    

    plt.show()


def array_equals(a, b, err):
    """
        See if 2 np.array are equal or close to equal
        a,b: two np.array
        err: the error from witch you consider a and b equals
        return: True or False
    """
    if np.array_equal(a,b):
        return True
    else:
        return norm(a - b)/norm(a) <= err


def unit_test():
    """
        Unit test
    """
    x = np.array([231, 32, 233, 161, 24, 71, 140, 245])
    x_dct = np.array( [4.01e+02, 6.60e+00, 1.09e+02, -1.12e+02, 6.54e+01, 1.21e+02, 1.16e+02, 2.88e+01])

    assert array_equals( dct(x), x_dct, err=0.01 )
    assert array_equals( idct(dct(x)),  x, err=0.01)

    test_matrix = np.array ( [[ 231, 32, 233, 161, 24, 71, 140, 245],
                        [ 247, 40, 248, 245, 124, 204, 36, 107],
                        [ 234, 202, 245, 167, 9, 217, 239, 173],
                        [ 193, 190, 100, 167, 43, 180, 8, 70],
                        [ 11, 24, 210, 177, 81, 243, 8, 112],
                        [ 97, 195, 203, 47, 125, 114, 165, 181],
                        [ 193, 70, 174, 167, 41, 30, 127, 245],
                        [ 87, 149, 57, 192, 65, 129, 178, 228]])

    test_matrix_dct = np.array ([[1.11e+03, 4.40e+01, 7.59e+01, -1.38e+02, 3.50e+00, 1.22e+02, 1.95e+02, -1.01e+02],
                        [7.71e+01, 1.14e+02, -2.18e+01, 4.13e+01, 8.77e+00, 9.90e+01, 1.38e+02, 1.09e+01],
                        [4.48e+01, -6.27e+01, 1.11e+02, -7.63e+01, 1.24e+02, 9.55e+01, -3.98e+01, 5.85e+01],
                        [-6.99e+01, -4.02e+01, -2.34e+01, -7.67e+01, 2.66e+01, -3.68e+01, 6.61e+01, 1.25e+02],
                        [-1.09e+02, -4.33e+01, -5.55e+01, 8.17e+00, 3.02e+01, -2.86e+01, 2.44e+00, -9.41e+01],
                        [-5.38e+00, 5.66e+01, 1.73e+02, -3.54e+01, 3.23e+01, 3.34e+01, -5.81e+01, 1.90e+01],
                        [7.88e+01, -6.45e+01, 1.18e+02, -1.50e+01, -1.37e+02, -3.06e+01, -1.05e+02, 3.98e+01],
                        [1.97e+01, -7.81e+01, 9.72e-01, -7.23e+01, -2.15e+01, 8.13e+01, 6.37e+01, 5.90e+00]])

    assert array_equals( naive_dct2(test_matrix),  test_matrix_dct, err=0.01 )
    assert array_equals( dct2(test_matrix),  test_matrix_dct, err=0.01 )
    assert array_equals( idct2( dct2(test_matrix) ), test_matrix, err=0.01 )
    assert array_equals( naive_dct2(test_matrix), dct2(test_matrix), err=0.01 )