import cv2
from matplotlib import pyplot as plt


def rescale(image, path):
  """"
  Args:
      image: numpy array of input image
      path: the path where the rescaled image is stored
  Returns:
      Nothing
  """
  plt.imsave(path, image, cmap='gray')


# def save(image=None, list_image=None, all=False):
#   """"
#   Args:
#       image: numpy array of input image
#       path: the path where the rescaled image is stored
#   Returns:
#       out: numpy array of shape (Hi, Wi)
#   """
#   if all == False: saveone(image)
#   elif all == True: saveall(list_image)
#   else: print("Input Invalid")


def saveone(image):
  """"
  Args:
      image: numpy array of input image
  Returns:
      Nothing
  """
  name = input("The name of Image: ")
  formatsave = input("Format for save (png, jpg, jpeg): ")
  cv2.imwrite(name+"."+formatsave, image)


def saveall(list_image):
  """"
  Args:
      list_image: list of numpy arrays of input images
  Returns:
      Nothing
  """
  formatsave = input("Format for save (png, jpg, jpeg): ")
  for idx in range(len(list_image)):
    cv2.imwrite("Idx "+str(idx)+"."+formatsave, list_image[idx])
