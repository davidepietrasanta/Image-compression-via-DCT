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
    #a = np.zeros(N) #[0:N-1]
    #w = np.zeros( shape=(N, N), dtype=np.float64 ) #[0:N-1]*N

    #for k in range(0, N):
    #    for j in range(0, N):
    #        w[k][j] = np.cos( k * np.pi * (2*(j+1) - 1)/(2*N) ) 

    #for k in range(0, N):
    #    if k == 0:
    #        a[k] = 1/np.sqrt(N)
    #    else:
    #        a[k] = np.sqrt(2)/np.sqrt(N)

    for k in range(0,N):
        if k == 0:
            a_k = 1/np.sqrt(N)
        else:
            a_k = np.sqrt(2)/np.sqrt(N)

        sum = 0
        for j in range(0,N):
            sum = sum + ( np.cos(k * np.pi * (2*(j+1) - 1)/(2*N)) * v[j] ) 

        c[k] = a_k * sum

    #c = normalize(c)
    #return {'w':w, 'a':a, 'c':c }
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