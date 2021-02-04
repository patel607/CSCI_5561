import cv2
import numpy as np
import matplotlib.pyplot as plt

# output the differential filters.
def get_differential_filter():
  # sobel filter
  filter_x = np.array([[1,0,-1], [2,0,-2], [1,0,-1]])
  filter_y = np.array([[1,2,1], [0,0,0], [-1,-2,-1]])
  return filter_x, filter_y

# Given an image and filter, you will compute the filtered image.
def filter_image(im, filter):
  im_filtered = np.zeros((im.shape))
  # leaving a pad of zeros along the boundary
  for i in range(1, im.shape[0]-1):
    for j in range(1, im.shape[1]-1):
      z = 0
      for x in range(filter.shape[0]):
        for y in range(filter.shape[1]):
          # basic convolution
          z += im[i+x-1][j+y-1] * filter[x][y]
      im_filtered[i][j] = z

  return im_filtered

# Given the differential images, you will compute the magnitude and angle
# of the gradient.
def get_gradient(im_dx, im_dy):
  # following equations for magntiude and angle
  grad_mag = np.sqrt(np.add(np.square(im_dx), np.square(im_dy)))
  grad_angle = np.arctan2(im_dy, im_dx)
  # make sure the angle is [0, pi)
  grad_angle = np.where(grad_angle < 0, grad_angle + np.pi, grad_angle)
  grad_angle = np.where(grad_angle >= np.pi, grad_angle - np.pi, grad_angle)
  return grad_mag, grad_angle

# Given the magnitude and orientation of the gradients per pixel, 
# you can build the histogram of oriented gradients for each cell.
def build_histogram(grad_mag, grad_angle, cell_size):
  # number of cells along y and x axes
  m,n  = grad_angle.shape
  M = np.floor(m/cell_size).astype(int)
  N = np.floor(n/cell_size).astype(int)

  ori_histo = np.zeros((M, N, 6))

  # sum up magnitudes for specific "bins" of angles
  for a in range(M):
    for b in range(N):
      i = a*cell_size
      j = b*cell_size
      z = 0
      for u in range(cell_size):
        for v in range(cell_size):
          deg = np.degrees(grad_angle[i + u][j + v])
          if (165 <= deg < 180 or 0 <= deg < 15):
            ori_histo[a][b][0] += grad_mag[i + u][j + v]
          elif  15 <= deg < 45:
            ori_histo[a][b][1] += grad_mag[i + u][j + v]
          elif  45 <= deg < 75:
            ori_histo[a][b][2] += grad_mag[i + u][j + v]
          elif  75 <= deg < 105:
            ori_histo[a][b][3] += grad_mag[i + u][j + v]
          elif  105 <= deg < 135:
            ori_histo[a][b][4] += grad_mag[i + u][j + v]
          elif  135 <= deg < 165:
            ori_histo[a][b][5] += grad_mag[i + u][j + v]

  return ori_histo

# Given the histogram of oriented gradients, you
# apply specific L_2 normalization
def get_block_descriptor(ori_histo, block_size):
  M = ori_histo.shape[0]
  N = ori_histo.shape[1]
  # normalization constant to prevent division by zero
  e = 0.001
  ori_histo_normalized = np.zeros((M - (block_size - 1), N - (block_size - 1), 6 * block_size ** 2))

  for i in range(M - (block_size - 1)):
    for j in range(N - (block_size - 1)):
      z = np.zeros(0)
      for u in range(block_size):
        for v in range(block_size):
          # concatenate to form one 24-element vector
          z= np.concatenate((z, ori_histo[i + u][j + v]))
      # sum of squares for normalization
      s = np.sum(np.square(z))
      # assigning normalized descriptor
      ori_histo_normalized[i][j] = z / np.sqrt(s + e ** 2)

  return ori_histo_normalized


def extract_hog(im):
  # convert grey-scale image to double format
  im = im.astype('float') / 255.0
  # HOG algorithm
  filter_x, filter_y = get_differential_filter()
  im_filtered_x = filter_image(im, filter_x)
  im_filtered_y = filter_image(im, filter_y)
  grad_mag, grad_angle = get_gradient(im_filtered_x, im_filtered_y)
  histo = build_histogram(grad_mag, grad_angle, 8)
  descriptor = get_block_descriptor(histo, 2)
  hog = descriptor.flatten()
  # visualize to verify
  #visualize_hog(im, hog, 8, 2)

  return hog


# visualize histogram of each block
def visualize_hog(im, hog, cell_size, block_size):
  num_bins = 6
  max_len = 7  # control sum of segment lengths for visualized histogram bin of each block
  im_h, im_w = im.shape
  num_cell_h, num_cell_w = int(im_h / cell_size), int(im_w / cell_size)
  num_blocks_h, num_blocks_w = num_cell_h - block_size + 1, num_cell_w - block_size + 1
  histo_normalized = hog.reshape((num_blocks_h, num_blocks_w, block_size**2, num_bins))
  histo_normalized_vis = np.sum(histo_normalized**2, axis=2) * max_len  # num_blocks_h x num_blocks_w x num_bins
  angles = np.arange(0, np.pi, np.pi/num_bins)
  mesh_x, mesh_y = np.meshgrid(np.r_[cell_size: cell_size*num_cell_w: cell_size], np.r_[cell_size: cell_size*num_cell_h: cell_size])
  mesh_u = histo_normalized_vis * np.sin(angles).reshape((1, 1, num_bins))  # expand to same dims as histo_normalized
  mesh_v = histo_normalized_vis * -np.cos(angles).reshape((1, 1, num_bins))  # expand to same dims as histo_normalized
  plt.imshow(im, cmap='gray', vmin=0, vmax=1)
  for i in range(num_bins):
      plt.quiver(mesh_x - 0.5 * mesh_u[:, :, i], mesh_y - 0.5 * mesh_v[:, :, i], mesh_u[:, :, i], mesh_v[:, :, i],
                  color='white', headaxislength=0, headlength=0, scale_units='xy', scale=1, width=0.002, angles='xy')
  plt.show()


def face_recognition(I_target, I_template):
  bounding_boxes = np.zeros((0,3))   
  temp = extract_hog(I_template)
  # necessary stride so runtime is not forever
  step = 8
  for i in range(0, I_target.shape[0] - I_template.shape[0], step):
    for j in range(0, I_target.shape[1] - I_template.shape[1], step):
      # calculate new hog for each 50 x 50 square of target
      targ = extract_hog(I_target[i:I_template.shape[0]+i, j:I_template.shape[1]+j])
      # ncc score
      s = np.dot(targ, temp) / (np.linalg.norm(targ) * np.linalg.norm(temp))
      # threshold of 0.
      if s > 0.6:
        bounding_boxes = np.append(bounding_boxes, [[j,i,s]], axis=0)

  # start of non-maximum suppression
  final = np.zeros((0,3))
  # continue while we haven't checked all BBs
  while len(bounding_boxes) > 0:
    max_score = np.array([0,0,0])
    # find BB with highest score
    for b in bounding_boxes:
      if b[2] > max_score[2]:
        max_score = b
    
    final = np.append(final, [max_score], axis=0)
    bounding_boxes = np.delete(bounding_boxes, np.where(bounding_boxes.T[2] == max_score[2]), axis = 0)

    i = 0
    while i < len(bounding_boxes):
      b = bounding_boxes[i]
      # intersection calculation from following link:
      # https://math.stackexchange.com/questions/99565/simplest-way-to-calculate-the-intersect-area-of-two-rectangles
      x_overlap = max(0, min(max_score[1]+I_template.shape[0], b[1]+I_template.shape[0]) - max(max_score[1], b[1]));
      y_overlap = max(0, min(max_score[0]+I_template.shape[0], b[0]+I_template.shape[0]) - max(max_score[0], b[0]));
      intersect = x_overlap * y_overlap;
      union = 2 * I_template.shape[0] ** 2 - intersect
      IoU = intersect / union

      # remove all BBs with IoU greater than 0.5
      if IoU > 0.5:
        bounding_boxes = np.delete(bounding_boxes, i, axis = 0)
        i += -1

      i += 1

  bounding_boxes = final
  return  bounding_boxes


def visualize_face_detection(I_target,bounding_boxes,box_size):

  hh,ww,cc=I_target.shape

  fimg=I_target.copy()
  for ii in range(bounding_boxes.shape[0]):

      x1 = bounding_boxes[ii, 0]
      x2 = bounding_boxes[ii, 0] + box_size 
      y1 = bounding_boxes[ii, 1]
      y2 = bounding_boxes[ii, 1] + box_size

      if x1<0:
          x1=0
      if x1>ww-1:
          x1=ww-1
      if x2<0:
          x2=0
      if x2>ww-1:
          x2=ww-1
      if y1<0:
          y1=0
      if y1>hh-1:
          y1=hh-1
      if y2<0:
          y2=0
      if y2>hh-1:
          y2=hh-1
      fimg = cv2.rectangle(fimg, (int(x1),int(y1)), (int(x2),int(y2)), (255, 0, 0), 1)
      cv2.putText(fimg, "%.2f"%bounding_boxes[ii,2], (int(x1)+1, int(y1)+2), cv2.FONT_HERSHEY_SIMPLEX , 0.5, (0, 255, 0), 2, cv2.LINE_AA)


  plt.figure(3)
  plt.imshow(fimg, vmin=0, vmax=1)
  plt.show()




if __name__=='__main__':

  im = cv2.imread('cameraman.tif', 0)
  hog = extract_hog(im)

  I_target= cv2.imread('target.png', 0)
  #MxN image

  I_template = cv2.imread('template.png', 0)
  #mxn  face template

  bounding_boxes=face_recognition(I_target, I_template)

  I_target_c= cv2.imread('target.png')
  # # MxN image (just for visualization)
  visualize_face_detection(I_target_c, bounding_boxes, I_template.shape[0])
  #this is visualization code.




