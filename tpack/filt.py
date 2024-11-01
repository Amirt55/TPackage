import numpy as np
import cv2
from google.colab.patches import cv2_imshow
from tpack.savim import saveall


def enhance(input, control=[1.5, 0], viewout=True, outret=True, saved=False):
  """"
  Args:
      input: numpy array of input images with shape (Hi, Wi, 2 or 3)
      control: list of alpha and beta for contrast control and brightness control
      viewout: view output images
      outret: whether or not to output enhanced image matrices
      saved: save images to local drive
  Returns:
      if the input image is Grayscale:
          normalized_img (1): normalized input image
          enhanced_image (2): thresholded image from Gaussian (blurred) enhancement
          median_filtered (3): the output image from the median filter
          contrast_enhanced (4): convertScaleAbs with Contrast control and Brightness control
          sharpened_image (5): the output image from a two-dimensional filter with a special kernel
      if the input image is RGB:
          contrast_enhanced (1): convertScaleAbs with Contrast control and Brightness control
          sharpened_image (2): the output image from a two-dimensional filter with a special kernel
  """
  alpha = control[0] # Contrast control (1.0-3.0)
  beta = control[1] # Brightness control (0-100)
  if len(input.shape) == 2:
    normalized_img = cv2.normalize(input, None, 0, 255, cv2.NORM_MINMAX)
    blurred = cv2.GaussianBlur(normalized_img, (5, 5), 0)
    _, enhanced_image = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    median_filtered = cv2.medianBlur(enhanced_image, 5)
    contrast_enhanced = cv2.convertScaleAbs(input, alpha=alpha, beta=beta)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened_image = cv2.filter2D(contrast_enhanced, -1, kernel)
    if viewout == True:
      cv2_imshow(enhanced_image)
      cv2_imshow(median_filtered)
      cv2_imshow(contrast_enhanced)
      cv2_imshow(sharpened_image)
    if saved == True:
      savelist = [normalized_img, enhanced_image, median_filtered, contrast_enhanced, sharpened_image]
      saveall(savelist)
    if outret == True: return normalized_img, enhanced_image, median_filtered, contrast_enhanced, sharpened_image
  if len(input.shape) == 3:
    contrast_enhanced = cv2.convertScaleAbs(input, alpha=alpha, beta=beta)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened_image = cv2.filter2D(contrast_enhanced, -1, kernel)
    if viewout == True:
      cv2_imshow(contrast_enhanced)
      cv2_imshow(sharpened_image)
    if saved == True:
      savelist = [contrast_enhanced, sharpened_image]
      saveall(savelist)
    if outret == True: return contrast_enhanced, sharpened_image


def conv(image, kernel):
  """"
  Args:
      image: numpy array of shape (Hi, Wi)
      kernel: numpy array of shape (Hk, Wk)
  Returns:
      out: numpy array of shape (Hi, Wi)
  """
  Hi, Wi = image.shape
  Hk, Wk = kernel.shape
  out = np.zeros((Hi, Wi))
  for m in range(Hi):
    for n in range(Wi):
      sum = 0
      for i in range(Hk):
        for j in range(Wk):
          if m + 1 - i < 0 or n + 1 - j < 0 or m + 1 - i >= Hi or n + 1 - j >= Wi: sum += 0
          else: sum += kernel[i][j] * image[m + 1 - i][n + 1 - j]
      out[m][n] = sum
  return out


def zero_pad(image, pad_height, pad_width):
  """
  Args:
      image: numpy array of shape (H, W)
      pad_width: width of the zero padding (left and right padding)
      pad_height: height of the zero padding (bottom and top padding)
  Returns:
      out: numpy array of shape (H+2*pad_height, W+2*pad_width)
  """
  if image.ndim == 2:
    # H, W = image.shape
    # out = np.zeros((H + 2 * pad_height, W + 2 * pad_width))
    # out[pad_height: H + pad_height, pad_width: W + pad_width] = image
    out = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)))
  else:
    # H, W, _ = image.shape
    # out = np.zeros((H + 2 * pad_height, W + 2 * pad_width, _))
    # out[pad_height: H + pad_height, pad_width: W + pad_width] = image
    out = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width), (0, 0)))
  return out


def conv_fast(image, kernel):
  """
  Args:
      image: numpy array of shape (Hi, Wi)
      kernel: numpy array of shape (Hk, Wk)
  Returns:
      out: numpy array of shape (Hi, Wi)
  """
  if image.ndim == 2:
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))
  else:
    Hi, Wi, _ = image.shape
    Hk, Wk, _ = kernel.shape
    out = np.zeros((Hi, Wi, _))
  image = zero_pad(image, Hk // 2, Wk // 2)
  kernel = np.flip(kernel, 0)
  kernel = np.flip(kernel, 1)
  for m in range(Hi):
    for n in range(Wi):
      out[m, n] = np.sum(image[m: m + Hk, n: n + Wk] * kernel)
  return out


def conv_faster(image, kernel):
  """
  Args:
      image: numpy array of shape (Hi, Wi)
      kernel: numpy array of shape (Hk, Wk)

  Returns:
      out: numpy array of shape (Hi, Wi)
  """
  Hi, Wi = image.shape
  Hk, Wk = kernel.shape
  out = np.zeros((Hi, Wi))
  image = zero_pad(image, Hk // 2, Wk // 2)
  kernel = np.flip(np.flip(kernel, 0), 1)
  mat = np.zeros((Hi * Wi, Hk * Wk))
  for i in range(Hi * Wi):
    row = i // Wi
    col = i % Wi
    mat[i, :] = image[row: row + Hk, col: col + Wk].reshape(1, Hk * Wk)
  out = mat.dot(kernel.reshape(Hk * Wk, 1)).reshape(Hi, Wi)
  return out


def cross_correlation(f, g):
  """ Cross-correlation of f and g
  Args:
      f: numpy array of shape (Hf, Wf)
      g: numpy array of shape (Hg, Wg)

  Returns:
      out: numpy array of shape (Hf, Wf)
  """
  g = np.flip(np.flip(g, 0), 1)
  out = conv_fast(f, g)
  return out