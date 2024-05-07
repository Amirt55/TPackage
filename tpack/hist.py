import cv2
import numpy as np
import matplotlib.pyplot as plt
import math


def calc_hist(image):
  '''
  calculate image histogram
  input(s):
    image (ndarray): input image
  output(s):
    hist (ndarray): computed input image histogram
  '''
  hist = np.zeros(256,dtype=int)
  unique_values, counts = np.unique(image, return_counts=True)
  hist[unique_values] = counts
  return hist


def calc_cdf(channel):
  '''
  calculate image cdf
  input(s):
    channel (ndarray): input image channel
  output(s):
    cdf (ndarray): computed cdf for input image channel
  '''
  hist = calc_hist(channel)
  cdf = np.cumsum(hist)
  cdf = cdf.astype(np.float64)
  cdf /= np.sum(hist)
  cdf *= (255)
  cdf = np.round(cdf,0)
  cdf = cdf.astype(np.int32)
  return cdf


def hist_matching(src_image,ref_image):
  '''
  input(s):
    src_image (ndarray): source image
    ref_image (ndarray): reference image
  output(s):
    output_image (ndarray): transformation of source image so that its histogram matches histogram of refrence image
  '''
  output_image = src_image.copy()
  channels = [(0, 'Blue channel'), (1, 'Green channel'), (2, 'Red channel')]
  for channel, title in channels:
    src_image_channel = src_image[:,:,channel]
    ref_image_channel = ref_image[:,:,channel]
    src_image_channel_flatten = src_image_channel.flatten()
    output_image_channel_flatten = np.zeros_like(src_image_channel_flatten)
    src_image_channel_cdf = calc_cdf(src_image_channel)
    ref_image_channel_cdf = calc_cdf(ref_image_channel)
    mapping_indices = np.searchsorted(ref_image_channel_cdf,src_image_channel_cdf )
    output_image_channel_flatten = mapping_indices[src_image_channel_flatten]
    output_image[:,:,channel] = output_image_channel_flatten.reshape(src_image_channel.shape)
  return output_image


def CLAHE(image, clipLimit=2.0, tileGridSize=(8,8)):
  img_gray = image.convert('L')
  img_arr = np.array(img_gray)
  clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
  img_clahe = clahe.apply(img_arr)
  plt.hist(img_clahe.ravel(),256,[0,256])
  plt.show()
  return img_clahe


def GHE(image):
  img_gray = image.convert('L')
  img_arr = np.array(img_gray)
  img_ghe = cv2.equalizeHist(img_arr)
  plt.hist(img_ghe.ravel(), 256, [0, 256])
  plt.show()
  return img_ghe


def BBHE(image):
  img_gray = image.convert('L')
  img_arr = np.array(img_gray)
  hist, bins = np.histogram(img_arr.flatten(), 256, [0,256])
  cdf = hist.cumsum()
  nj = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
  N = img_arr.flatten().shape[0]
  bbhe_img = nj[img_arr.flatten()].reshape(img_arr.shape)
  hist_bbhe, bins_bbhe = np.histogram(bbhe_img.flatten(), 256, [0,256])
  plt.plot(hist_bbhe)
  plt.show()
  return bbhe_img, hist_bbhe
