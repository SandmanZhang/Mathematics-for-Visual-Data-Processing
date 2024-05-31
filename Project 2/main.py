import numpy as np
import scipy.signal
import matplotlib.pyplot as plt

def haar2d():
    H = 1/2*np.array([[1,1],[1,1]])
    G1 = 1/2*np.array([[1,1],[-1,-1]])
    G2 = 1/2*np.array([[1,-1],[1,-1]])
    G3 = 1/2*np.array([[1,-1],[-1,1]])
    return H,G1,G2,G3

def down_sampling2d(img, mode = 'valid'):
    if mode == 'valid':
        return img[::2,::2]
    if mode == 'same' or mode == 'full':
        return img[1::2,1::2]

def up_sampling2d(img, mode = 'valid'):
    n, m = img.shape
    zero_arr = np.zeros((int(2*n),int(2*m)))
    if mode == 'valid':
        zero_arr[::2,::2] = img
        return zero_arr
    if mode == 'same' or mode == 'full':
        zero_arr[1::2,1::2] = img
        return zero_arr

def dwt2d(im,lvl):
    H, G1, G2, G3 = haar2d()
    coef_list = []
    for i in range(lvl):
        s1 = scipy.signal.convolve2d(im, np.flip(H), mode='valid')
        w1 = scipy.signal.convolve2d(im, np.flip(G1), mode='valid')
        w2 = scipy.signal.convolve2d(im, np.flip(G2), mode='valid')
        w3 = scipy.signal.convolve2d(im, np.flip(G3), mode='valid')
        ds1 = down_sampling2d(s1)
        dw1 = down_sampling2d(w1)
        dw2 = down_sampling2d(w2)
        dw3 = down_sampling2d(w3)
        coef_list.append((dw1, dw2, dw3))
        im = ds1
    coef_list.append(im)
    new_img = coef_list[-1]
    for i in range(lvl-1, -1, -1):
        w = coef_list[i]
        new_img = np.concatenate((new_img, w[0]), axis = 1)
        img_2 = np.concatenate((w[1], w[2]), axis = 1)
        new_img = np.concatenate((new_img,img_2), axis = 0)
    return new_img

def recover_coef_list(coef, lvl):
    coef_list = []
    for j in range(lvl - 1, -1,-1):
        n, m = coef.shape
        n_, m_ = int(n/2), int(m/2)
        w1 = coef[:n_, m_:]
        w2 = coef[n_:,:m_]
        w3 = coef[n_:,m_:]
        coef = coef[:n_,:m_]
        coef_list.append((w1,w2,w3))
        if j == 0:
            coef_list.append(coef)
    return coef_list

def idwt2d(coef, lvl):
    coef = recover_coef_list(coef, lvl)
    s = coef[-1]
    H, G1, G2, G3 = haar2d()
    for i in range(lvl - 1, -1, -1):
        w1, w2, w3 = coef[i]
        up_sam_s = up_sampling2d(s)
        up_sam_w1 = up_sampling2d(w1)
        up_sam_w2 = up_sampling2d(w2)
        up_sam_w3 = up_sampling2d(w3)
        s1_conv = scipy.signal.convolve2d(up_sam_s, H, mode='same')
        w1_conv = scipy.signal.convolve2d(up_sam_w1, G1, mode='same')
        w2_conv = scipy.signal.convolve2d(up_sam_w2, G2, mode='same')
        w3_conv = scipy.signal.convolve2d(up_sam_w3, G3, mode='same')
        s = s1_conv + w1_conv + w2_conv + w3_conv
    return s

def recover_coef_list(coef, lvl):
    coef_list = []
    for j in range(lvl - 1, -1,-1):
        n, m = coef.shape
        n_, m_ = int(n/2), int(m/2)
        w1 = coef[:n_, m_:]
        w2 = coef[n_:,:m_]
        w3 = coef[n_:,m_:]
        coef = coef[:n_,:m_]
        coef_list.append((w1,w2,w3))
        if j == 0:
            coef_list.append(coef)
    return coef_list


if __name__ == '__main__':
    from PIL import Image
    img = Image.open('./sample.png')
    img = np.array(img)
    lvl = 1
    coef = dwt2d(img, lvl)
    fig, ax = plt.subplots(1,1,figsize = (20,20))
    ax.imshow(coef, cmap = plt.cm.gray)
    ax.set_title('Graph of haar wavelet transform with Level {}'.format(lvl), fontsize = 15)
    plt.savefig('./wavelet transform with Level {}.png'.format(lvl))
    plt.show()
    recover_img = idwt2d(coef, lvl)
    img_ = np.concatenate((recover_img, img), axis = 1)
    fig, ax = plt.subplots(1,1,figsize = (12,8))
    ax.imshow(img_, cmap = plt.cm.gray)
    ax.set_title('Reconstructed Image(left) & Original Image(Right)', fontsize = 15)
    plt.savefig('./Reconstructed Image.png')
    plt.show()
    print('Whether Equals to Original Image:', np.allclose(recover_img, img))
