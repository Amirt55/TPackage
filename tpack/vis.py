import matplotlib.pylab as plt
import numpy as np
from mpl_toolkits import mplot3d



def show_grays(images, cols=2):
    plt.rcParams['figure.figsize'] = (15, 20)
    imgs = images['image'] if isinstance(images, dict) else images

    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, ax = plt.subplots(ncols=cols, nrows=np.ceil(len(imgs) / cols).astype(np.int8), squeeze=False)
    for i, img in enumerate(imgs):
        ax[i // cols, i % cols].imshow(np.asarray(img), cmap='gray')
        ax[i // cols, i % cols].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        if isinstance(images, dict): ax[i // cols, i % cols].title.set_text(images['name'][i])
    plt.show()


def plotfig(listforplot, ncol=3, nrow=1, figs=(30,27), axissituation='off'):
  fig, plotter = plt.subplots(ncols= ncol , nrows= nrow , figsize=figs)
  for idy in range(nrow):
    for idx in range(ncol):
      ax = plotter[idy][idx] if nrow > 1 else plotter[idx]
      if idx + idy * ncol >= len(listforplot): break
      ax.imshow(listforplot[idx + idy * ncol])
      title = input(f"Enter title for image {idx+1} for {idy+1}th row: ")
      if title: ax.set_title(title)
      ax.axis(axissituation)
  plt.show()


def plot3d(X, Y, z):
  fig = plt.figure()
  ax = plt.axes(projection='3d')
  ax.plot_surface(X, Y, z, cmap='viridis')
  ax.set_xlabel(input(f"Title for x: "))
  ax.set_ylabel(input(f"Title for y: "))
  ax.set_zlabel(input(f"Title for z = F(x,y): "))
  ax.set_title('3D Surface Plot')
  plt.show()


def gridfor3d(x, y):
  X, Y = np.meshgrid(x, y)
  return X, Y


def rescale(image, path):
    plt.imsave(path, image, cmap='gray')