import cv2
from matplotlib import pyplot as plt


def save(image=None, list_image=None, all=False):
  if all == False: saveone(image)
  elif all == True: saveall(list_image)
  else: print("Input Invalid")


def saveone(image):
  name = input("The name of Image: ")
  formatsave = input("Format for save (png, jpg, jpeg): ")
  cv2.imwrite(name+"."+formatsave, image)


def saveall(list_image):
  formatsave = input("Format for save (png, jpg, jpeg): ")
  for idx in range(len(list_image)):
    cv2.imwrite("Idx "+str(idx)+"."+formatsave, list_image[idx])


def rescale(image, path):
  plt.imsave(path, image, cmap='gray')
