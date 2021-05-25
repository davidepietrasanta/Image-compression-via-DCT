import scipy as sp
import numpy as np

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
    a = np.zeros(N) #[0:N-1]
    w = np.zeros( shape=(N, N), dtype=np.float64 ) #[0:N-1]*N

    c1 = np.zeros(N)
    a1 = np.zeros(N)

    for k in range(0, N):
        for j in range(0, N):
            w[k][j] = np.cos( k * np.pi * (2*(j+1) - 1)/(2*N) ) ##

    for k in range(0, N):
        a1[k] = 1 / np.dot( w[k], w[k] ) ##

        if k == 0:
            a[k] = 1/np.sqrt(N)
        else:
            a[k] = np.sqrt(2)/np.sqrt(N)

    for k in range(0,N):
        if k == 0:
            a_k = 1/np.sqrt(N)
        else:
            a_k = np.sqrt(2)/np.sqrt(N)

        sum = 0
        for j in range(0,N):
            sum = sum + ( np.cos(k * np.pi * (2*(j+1) - 1)/(2*N)) * v[j] ) ##

        c[k] = a_k * sum
        c1[k] = a1[k] * sum

    c = normalize(c)
    c1 = normalize(c1)
    return {'w':w, 'a':a, 'a1':a1, 'c':c, 'c1':c1 }


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
        matrix_r[i] = dct(matrix[i])['c1'] 

    for j in range(0, N):
        temp = dct(matrix_r[:,j])['c1']
        for k in range(0, M):
            matrix_r[k,j] = temp[k]

    return matrix_r


def naive_dtc2(matrix):
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




