import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def dct1d(f):
    N = len(f)
    seq = np.arange(N)
    cu = np.zeros(N).reshape(N,1)
    u = np.arange(N).reshape(1,N)
    u = np.repeat(u,N,axis=1).reshape(N,N)
    cu[0] = np.sqrt(1/N)
    cu[1:] = np.sqrt(2/N)
    term = np.cos(((seq+0.5)*u*np.pi/N))
    dct = (cu*np.sum(f*term,axis = 1, keepdims = True)).flatten()
    return dct.flatten()

def idct1d(dct):
    N = len(dct)
    seq = (np.arange(N).reshape(N,1)+0.5)*np.pi/N
    u = np.arange(N).reshape(1,N)
    cu = np.zeros((N,N))
    cu[:,0] = np.sqrt(1/N)
    cu[:,1:] = np.sqrt(2/N)
    u = np.repeat(u,N,axis=0).reshape(N,N)
    idct = (cu*dct*np.cos(seq*u)).sum(axis = 1)
    return idct

def dct2d(img):
    N,M = img.shape
    process_img1 = dct1d(img[0,:]).reshape(1,M)
    for i in range(1,N):
        col_vec = dct1d(img[i,:]).reshape(1,M)
        process_img1 = np.vstack((process_img1,col_vec))
    process_img2 = dct1d(process_img1[:,0]).reshape(N,1)
    for j in range(1,M):
        col_vec = dct1d(process_img1[:,j]).reshape(N,1)
        process_img2 = np.hstack((process_img2,col_vec))
    return process_img2

def idct2d(img):
    N,M = img.shape
    process_img1 = idct1d(img[0,:]).reshape(1,M)
    for i in range(1,N):
        col_vec = idct1d(img[i,:]).reshape(1,M)
        process_img1 = np.vstack((process_img1,col_vec))
    process_img2 = idct1d(process_img1[:,0]).reshape(N,1)
    for j in range(1,M):
        col_vec = idct1d(process_img1[:,j]).reshape(N,1)
        process_img2 = np.hstack((process_img2,col_vec))
    return process_img2


def DFT1d(array):
    N = array.shape[0]
    seq = np.arange(N)
    k = seq.reshape((N, 1))
    pi = np.pi
    Matrix = np.exp(-2j * pi * k * seq / N)
    return np.dot(Matrix, array)

def fft2d(img):
    def fft1d(array):
        N = array.shape[0]
        if N <= 2:
            return DFT1d(array)
        else:
            even_term = fft1d(array[::2])
            odd_term = fft1d(array[1::2])
            seq = np.arange(N)
            pi = np.pi
            coef = np.exp(-2j*pi*seq/N)
            half_N = int(N/2)
            fft_res = np.hstack((even_term+coef[:half_N]*odd_term, even_term+coef[half_N:]*odd_term))
            return fft_res
    N, M = img.shape
    process_img1 = fft1d(img[0,:]).reshape(1,M)
    for i in range(1,N):
        col_vec = fft1d(img[i,:]).reshape(1,M)
        process_img1 = np.vstack((process_img1, col_vec))
    process_img2 = fft1d(process_img1[:,0]).reshape(N,1)
    for j in range(1,M):
        col_vec = fft1d(process_img1[:,j]).reshape(N,1)
        process_img2 = np.hstack((process_img2, col_vec))
    return process_img2

def ifft2d(img):
    def ifft1d(array):
        N = array.shape[0]
        k = np.arange(N).reshape(N, 1)
        w = np.exp(2j*np.pi*k*np.arange(N)/N)
        return (array.dot(w.T) / N)
    N, M = img.shape
    process_img1 = ifft1d(img[0, :]).reshape(1, M)
    for i in range(1, N):
        col_vec = ifft1d(img[i, :]).reshape(1, M)
        process_img1 = np.vstack((process_img1, col_vec))
    process_img2 = ifft1d(process_img1[:, 0]).reshape(N, 1)
    for j in range(1, M):
        col_vec = ifft1d(process_img1[:, j]).reshape(N, 1)
        process_img2 = np.hstack((process_img2, col_vec))
    return process_img2.real


if __name__ == '__main__':
    pic = np.array(Image.open('./sample.png'))
    dct_pic = dct2d(pic)
    fft_pic = fft2d(pic)
    idct_pic = idct2d(dct_pic)
    ifft_pic = ifft2d(fft_pic)
    plt.imshow(np.log(abs(fft_pic)+1))
    plt.savefig(fname = 'FFT2d Result.png')
    plt.show()
    plt.imshow(np.log(abs(dct_pic)+1))
    plt.savefig(fname = 'DCT2d Result.png')
    plt.show()
    plt.imshow(ifft_pic)
    plt.savefig(fname = 'iFFT2d Result.png')
    plt.show()
    plt.imshow(idct_pic)
    plt.savefig(fname = 'iDCT2d Result.png')
    plt.show()
    print('Check Result:', np.allclose(idct_pic, pic), np.allclose(ifft_pic, pic))



