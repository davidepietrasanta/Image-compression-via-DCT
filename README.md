# Image-compression-via-DCT

The purpose of this project is to use the DCT2 (Discrete Cosine Transform) implementation in an environment
open source and to study the effects of jpeg-like compression (without using a
quantization matrix) on grayscale images.

## First part
Implement DCT2 as explained in class in an open source environment of your choice
and compare the execution times with the DCT2 obtained using the environment library
used, which is assumed to be in the fast version (FFT).

In particular, obtain square arrays N N with increasing N and represent on a
graph on a semilogarithmic scale (only the ordinates) as N varies the time taken ad
run DCT2 with your homemade algorithm and library algorithm.

The times should have been proportional to N3 for the homemade DCT2 and to N2 for the version
fast (more precisely to N2 log (N)). The times obtained with the fast version could have
an irregular trend due to the type of algorithm used.

## Second part
Write software that performs the following tasks:
- Create a simple interface so that the user can choose from the lesystem
a `.bmp` image in shades of gray;
- Allow the user to choose:
    - an integer F which will be the width of the window in which DCT2 is executed;
    - an integer d between 0 and (2F-2) which will be the cutoff threshold of the frequencies.
- divide the image into square blocks f of pixels of size FxF starting
top left, discarding leftovers;
- for each block f carry out the following operations:
    - apply DCT2 (from the library): c = DCT2 (f);
    - delete frequencies c_kl with k + l >= d (I am assuming that frequencies
    start from 0: if d = 0 I delete them all, if d = (2F-2) I delete only the most
    high, i.e. the one with k = F-1, l = F-1).
    - apply the inverse DCT2 to array c as modified: ff = IDCT2 (c);
    - round ff to the nearest integer, set negative values to zero and 255
    those greater than 255 in order to have admissible values (1 byte);
- recompose the image by putting the ff blocks together in the right order.
- display on the screen side by side: the original image and the image obtained
after modifying the frequencies in the blocks.
