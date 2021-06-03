# Parte 2

from numpy.core.shape_base import block
from scipy.fftpack import dct, idct
import pathlib
import cv2 as cv

from skimage.io import imread
from skimage.color import rgb2gray
import numpy as np
import matplotlib.pylab as plt

from PIL import Image
from numpy import asarray
import copy


def JPEG_simple(path):
    """
        Simple dct/idct to see if correct
    """

    # read lena RGB image and convert to grayscale
    im = cv.imread(path, 0)
    im = im[0:im.shape[0] - 1, 0:im.shape[1] - 1]
    imF = cv.dct(im / 1.0)
    dim_cut = 200
    for r in range(0,im.shape[0]):
        for c in range(0, im.shape[1]):
            if r + c > dim_cut:
                imF[r][c] = 0
    im1 = cv.idct(imF / 1.0)

    # check if the reconstructed image is nearly equal to the original image
    np.allclose(im, im1)

    # plot original and reconstructed images with matplotlib.pylab
    plt.gray()
    plt.subplot(121), plt.imshow(im), plt.axis('off'), plt.title('original image', size=20)
    plt.subplot(122), plt.imshow(im1), plt.axis('off'), plt.title('reconstructed image (DCT+IDCT)', size=20)
    plt.show()


def array_to_image(array, bw=False, save=False, path=None):
    """
        Convert an array to an image.
        bw: "Black and White", should be true if the image is B&W
        save: If true save the image
        path: If save is true the image is saved in following path
    """
    if bw:
        img = Image.fromarray(array, 'L')  # Black and White
    else:
        img = Image.fromarray(array, 'RGB')

    img.show()

    if save:
        path = path + "_i.jpg"
        img.save(path)


def divide_immage_in_blocks(path, F):
    """
        Divide the image into square blocks f of pixels of size FxF,
        starting top left and discarding leftovers.
        path: Image path
        F: Block size
        return: Matrix of matrix
    """
    # img = Image.open(path)
    img = cv.imread(path, 0)

    # Convert PIL images into NumPy arrays
    picture = asarray(img)

    block_size = F

    # Create the blocks
    N = picture.shape[0]
    M = picture.shape[1]
    original_shape = [N, M]
    blocks = []
    for r in range(0, picture.shape[0] - block_size, block_size):
        for c in range(0, picture.shape[1] - block_size, block_size):
            window = picture[r:r + block_size, c:c + block_size]
            blocks.append(window)
            # array_to_image(window, bw=True)

    return [blocks, original_shape]


def list_index_to_matrix(i, d):
    """
        Simple converter to index of a list to index of a matrix.
    """
    n = np.floor(i / d)  # row
    m = i % d  # column
    return [n, m]


def compression(list_of_blocks):
    """
        Apply the dct on the list of blocks
    """
    #list_of_blocks = copy.deepcopy(list_of_blocks)
    list_blocks_dct = []
    for block in list_of_blocks:
        block_r = cv.dct(block / 1.0)
        list_blocks_dct.append(block_r)

    return list_blocks_dct


def cut_block(list_of_blocks, d):
    """
        Cut the frequency of the blocks
    """
    list_of_blocks_cut = [] #copy.deepcopy(list_of_blocks)
    block_x = len(list_of_blocks[0])
    block_y = len(list_of_blocks[0][0])
    for block in list_of_blocks:
        list_of_blocks_cut.append(block)
    for block in list_of_blocks_cut:
        for k in range(0, block_x):
            for l in range(0, block_y):
                if (k + l >= d):
                    block[k][l] = 0

    return list_of_blocks_cut


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


def fix_inverse(block):
    """
        While compressing idtc can produce floating numbers.
        We need to fix that, returning value in [0, 255]
        v: Is a np.array of one dimension [0:N-1]
        return: A np.array of one dimension [0:N-1]
    """
    x = len(block)
    y = len(block[0])
    for i in range(0, x):
        for j in range(0, y):
            # print(type(block[i][j]))
            # print(block[i][j])
            if block[i][j] > 255:
                block[i][j] = 255

            if block[i][j] < 0:
                block[i][j] = 0

            if not is_integer_num(block[i][j]):
                block[i][j] = round(block[i][j])

    return block


def inverse(list_of_blocks):
    """
        Apply idct to the list of blocks
    """
    list_of_blocks_inv = []
    for block in list_of_blocks:
        block_inv = cv.idct(block / 1.0)
        block_inv_fix = fix_inverse(block_inv)
        list_of_blocks_inv.append(block_inv_fix)
    return list_of_blocks_inv

'''
def recompose_image(path, list_of_blocks, original_shape, F):
    n_block_x = int(np.floor(original_shape[0] / F))
    n_block_y = int(np.floor(original_shape[1] / F))
    # print(n_block_x)
    # print(n_block_y)
    # N = len(list_of_blocks)
    # for i in range(0, n_block_x):
    #    for j in range(0, n_block_y):
    # img = Image.open(path)
    img = cv.imread(path, 0)
    # Convert PIL images into NumPy arrays
    picture = asarray(img)
    picture_cut = picture[0:n_block_x * F, 0:n_block_y * F]
    block_size = F
    # Create the blocks
    i = 0
    for r in range(0, picture.shape[0] - block_size, block_size):
        for c in range(0, picture.shape[1] - block_size, block_size):
            picture_cut[r:r + block_size, c:c + block_size] = list_of_blocks[i]
            i = i + 1
    print(picture_cut)
    print(type(picture))
    picture_cut = picture_cut.reshape((n_block_x * F, n_block_y * F))
    print(picture_cut)
    print(type(picture_cut))
    array_to_image(picture_cut / 1.0, bw=True, save=True, path=path)
    cout = 0
    for arr in picture_cut:
        for n in arr:
            if (n == 0):
                cout = cout + 1
    print(cout)
    return picture_cut
'''

def recomp(list_of_blocks, original_shape, F):
    n_block_x = int(np.floor(original_shape[0] / F))
    n_block_y = int(np.floor(original_shape[1] / F))
    real_x = n_block_x * F
    real_y = n_block_y * F

    matrix = np.zeros(shape=(real_x, real_y))
    matrix = matrix.astype(np.uint8)
    N_blocks = len(list_of_blocks)

    for i in range(0, N_blocks):
        #array_to_image(list_of_blocks[i], bw=True)
        for y in range(0, F):
            for x in range(0, F):
                m_y = (i % n_block_y) * F + y
                m_x = int( np.floor( i / n_block_y) * F) + x
                matrix[m_x][m_y] = int(list_of_blocks[i][x][y])

    '''
    block_row = []
    for y in range(0, n_block_y):
        block_row.append(list_of_blocks[y * n_block_x])
        matrix
    # np.concatenate((A,B), axis=1 )
    for i in range(0, N_blocks):
        # array_to_image(list_of_blocks[i], bw=True)
        for x in range(1, n_block_x):
            for y in range(1, n_block_x):
                block_row[y] = np.concatenate((block_row[y - 1], list_of_blocks[y * n_block_x + x]), axis=1)
    # matrix = [0]
    # for y in range(1, n_block_x):
    #    matrix = np.concatenate((matrix, block_row[y]), axis=0 )
    print(block_row)
    '''

    return matrix

def JPEG_compession(path, F, d):
    """
        Apply JPEG compression of image.
        path: Path of the image
        F: Dimension of each block FxF
        d: Cut parameter
    """
    [list_b, original_shape] = divide_immage_in_blocks(path, F)
    list_c = compression(list_b)

    #for block in list_c:
    #    array_to_image(block, bw=True)
    cut = cut_block(list_c, d)
    inv = inverse(cut)
    #im = recompose_image(path, inv, original_shape, F)
    im = recomp(inv, original_shape, F)
    # im = recompose_image(path, inv, original_shape, F)
    return im


path = str(pathlib.Path(__file__).parent.absolute())
# JPEG_simple(path + '\\40682662_ml.jpg')
F = 300
d = 50
#JPEG_simple(path + '\\deer.bmp')
im = JPEG_compession(path + '\\deer.bmp', F, d)
array_to_image(im, bw=True) 