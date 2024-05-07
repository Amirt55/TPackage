import numpy as np
from PIL import Image
import cv2
from matplotlib.colors import rgb_to_hsv


def readImage(imagepath):
  image = Image.open(imagepath)
  return np.array(image)


def piltoarray(list_pilimage):
  arr_img = list()
  shapes = list()
  hsv_list = list()
  for idx in range(len(list_pilimage)):
    one_arr = np.array(list_pilimage[idx])
    norm_arr = (one_arr - np.min(one_arr)) / (np.max(one_arr) - np.min(one_arr))
    one_hsv = rgb_to_hsv(norm_arr)
    hsv_list.append(one_hsv)
    shape_of_image = one_arr.shape
    shapes.append(shape_of_image)
    # lst_one = list(one_arr)
    # arr_img.append(lst_one)
    arr_img.append(one_arr)
  return arr_img, norm_arr, shapes, hsv_list


def canny_edgedetect(image, kernel_size=5, low_threshold=50, high_threshold=150):
  grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  blur = cv2.GaussianBlur(grayscale, (kernel_size, kernel_size), 0)
  edges = cv2.Canny(blur, low_threshold, high_threshold)
  return edges


def hough_transform(image, rho=1, theta=np.pi/180, threshold=20, minLineLength=20, maxLineGap=15):
	return cv2.HoughLinesP(image, rho = rho, theta = theta, threshold = threshold, minLineLength = minLineLength, maxLineGap = maxLineGap)


def quantization_levels(max_val, min_val, level):
  n = int((max_val - min_val) / level) + 1
  lst_level = [(i * n) for i in range(level)]
  lst_level.append(max_val)
  return lst_level


def quantization(img, lst_level):
  img_arr = np.array(img)
  quantized_pixels = np.zeros_like(img_arr)
  for i in range(len(lst_level)-1):
    mask = np.logical_and(img_arr >= lst_level[i], img_arr < lst_level[i+1])
    quantized_pixels[mask] = lst_level[i]
  quantized_img = Image.fromarray(quantized_pixels.astype('uint8'), mode='L')
  return quantized_img


def quantize(image, quantize_level):
  images = {'name': [], "image": []}
  images['image'].append(image)
  images['name'].append('orginal')
  for q in quantize_level:
    bins = np.histogram_bin_edges([-1, 255], q)
    dig = np.digitize(image, bins=bins, right=True)
    images['image'].append(dig.astype(np.uint8))
    images['name'].append(f'gray level{q}')
  return images


def resize(image, resize_level):
  images = {'name': [], "image": []}
  images['image'].append(image)
  images['name'].append(str('orginal'))
  input_size = 1024
  for output_size in resize_level:
    bin_size = input_size // output_size
    small_image = image.reshape((1, output_size, bin_size,
                                  output_size, bin_size)).max(4).max(2)

    images['image'].append(small_image[0])
    images['name'].append(str(f'resize{output_size}'))
  return images
