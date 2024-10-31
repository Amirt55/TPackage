import cv2
import numpy as np
from matplotlib import pyplot as plt


def get_fps(inputvideo):
  return int(inputvideo.get(cv2.CAP_PROP_FPS))


def framextractor(ok, frame, index, framelist, framedim=None, framedimout=False, plot=False):
  breakloop = 0
  if ok == True:
    framelist.append(frame)
    if framedimout == True:
      h, w, c = frame.shape
      dim = [h, w, c]
      framedim.append(dim)
    print(f"Frame {index}")
    index += 1
    if plot == True:
      plt.imshow(frame[:,:,::-1])
      plt.show()
  else:
    print("All frames have been imported")
    breakloop = 1
  return framelist, framedim, index, breakloop


def savevideo(videoframes, frame):
  videoframes.write(frame)


def process_video(inputvideopath, outputvideopath):
  input_video = cv2.VideoCapture(inputvideopath)
  width = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
  height = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
  fps = get_fps(input_video)
  print(f"The FPS of the input video = {fps}")
  output_video = cv2.VideoWriter(outputvideopath, cv2.VideoWriter_fourcc(*"XVID"), fps, (width, height))
  while True:
      ok, frame = input_video.read()
      if not ok:
          break
      # Your Code For Processing#
      processed_frame = frame
      # #########################
      output_video.write(processed_frame)
  input_video.release()
  output_video.release()
  print("|---------- Done ----------|")
