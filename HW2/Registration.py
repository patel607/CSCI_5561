import cv2
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy import interpolate


def find_match(img1, img2):
  x1 = np.zeros((0,2))
  x2 = np.zeros((0,2))
  # key points and descriptors
  sift = cv2.xfeatures2d.SIFT_create()
  kp1, des1 = sift.detectAndCompute(img1,None)
  kp2, des2 = sift.detectAndCompute(img2,None)
  # translate key points to actual x, y coordinates
  pts1 = np.array([kp1[idx].pt for idx in range(0, len(kp1))]).reshape(-1, 2)
  pts2 = np.array([kp2[idx].pt for idx in range(0, len(kp2))]).reshape(-1, 2)
  # nearest neighbors fitting larger image
  neigh = NearestNeighbors(n_neighbors=2)
  neigh.fit(des2)
  distances, indices = neigh.kneighbors(des1)
  for i in range(len(distances)):
    # ratio test
    if distances[i, 0]/distances[i, 1] < 0.7:
      x1 = np.append(x1, [pts1[i]], axis = 0)
      x2 = np.append(x2, [pts2[indices[i,0]]], axis = 0)

  return x1, x2

def align_image_using_feature(x1, x2, ransac_thr, ransac_iter):
  max_inliers = -1;
  # ransac algorithm followed
  for i in range(ransac_iter):
    # sample 3 corresponding points without replacement
    points = np.random.choice(len(x1), 3, replace=False)
    inp = np.zeros((0,6))
    out = np.zeros((0,1))
    for p in points:
      inp = np.append(inp, [[x1[p][0],x1[p][1], 1, 0,0,0]], axis=0)
      inp = np.append(inp, [[0,0,0, x1[p][0],x1[p][1], 1]], axis=0)
      out = np.append(out, [[x2[p][0]]], axis=0)
      out = np.append(out, [[x2[p][1]]], axis=0)
    # calculation to calculate affine matrix from input and output
    AtA = np.matmul(inp.T, inp)
    inv = np.linalg.inv(AtA)
    invAt = np.matmul(inv, inp.T)
    affine = np.matmul(invAt, out)
    affine = np.array([[affine[0,0], affine[1,0], affine[2,0]], [affine[3,0], affine[4,0], affine[5,0]], [0,0,1]])

    inlier_count = 0
    for i in range(len(x1)):
      # affine transform coordinates to get expected values
      u1 = np.array([[x1[i,0]], [x1[i,1]], [1]])
      exp = np.matmul(affine, u1)
      u2 = np.array([[x2[i,0]], [x2[i,1]], [1]])
      # check if distance under threshold for inlier
      if (abs(np.linalg.norm(exp-u2)) < ransac_thr):
        inlier_count+=1
      # keep track of best affine matix
      if (inlier_count > max_inliers):
        max_inliers = inlier_count
        A = affine
  return A

def warp_image(img, A, output_size):
  img_warped = np.zeros(output_size)
  # for each coordinate in template, affine transform to get to target
  for i in range(output_size[0]):
    for j in range(output_size[1]):
      x2 = np.array([[j],[i],[1]])
      warped = np.floor(np.matmul(A, x2))
      x1 = img[int(warped[1,0])][int(warped[0,0])]
      img_warped[i][j] = x1
  return img_warped

# inverse compositional image alignment
def align_image(template, target, A):
  inc = 0
  A_refined = A.copy()
  # p is the first 6 values from A
  p = A.copy().reshape((1,9))[0][0:6].reshape((6,1))
  # compute gradient of template image
  filter_x, filter_y = get_differential_filter()
  I_y, I_x = filter_image(template, filter_y), filter_image(template, filter_x)

  steepest = np.zeros((template.shape[0], template.shape[1], 1,6))
  hessian = np.zeros((6,6))
  errors = []
  # construct jacobian for each coordinate in template
  # use each jacobian to calculate steepest descent images for each coordinate
  # sum up every SDI.T * SDI to get 6x6 hessian
  for u in range(template.shape[0]):
    for v in range(template.shape[1]):
      jacobian = np.array([[u,v,1,0,0,0],[0,0,0,u,v,1]])
      steepest[u,v] = np.matmul(np.array([I_x[u,v],I_y[u,v]]), jacobian)
      hessian = np.add(hessian, np.matmul(steepest[u,v].T, steepest[u,v]))
  
  while (np.linalg.norm(p) > 0.05 and inc < 200):
    I_warped = warp_image(target, A_refined, template.shape)
    # error image and corresponding error magnitude
    I_err = I_warped - template
    errors.append(np.linalg.norm(I_err))
    # calculate F by summing SDI.T * Error for every coordinate in template
    F = np.zeros((6,1))
    for i in range(I_err.shape[0]):
      for j in range(I_err.shape[1]):
        F += np.multiply(steepest[i,j].T, I_err[i,j])
    # delta p =H^(-1)F
    del_p = np.matmul(np.linalg.inv(hessian), F)
    new_p = del_p.copy()
    new_p[0] += 1
    new_p[4] += 1
    affine = np.append(new_p, [0,0,1]).reshape((3,3))
    # W(x;p) <- W(W^(-1)*x;delta_p);p)
    A_refined = np.matmul(A_refined, np.linalg.inv(affine))
    inc += 1
  return A_refined, np.array(errors)


def track_multi_frames(template, img_list):
    A_list = np.zeros((len(img_list), 3, 3))
    x1, x2 = find_match(template, img_list[0])
    # initialize A
    A = align_image_using_feature(x1, x2, 4, 1000)
    for i in range(len(img_list)):
      A_ref, e = align_image(template, img_list[i], A)
      A_list[i] = A_ref
      # update template and A every frame
      template = warp_image(img_list[i], A_ref, template.shape)
      A = A_ref
    return A_list

# output the differential filters.
def get_differential_filter():
  # sobel filter
  filter_x = np.array([[-1,0,1],
                       [-2,0,2],
                       [-1,0,1]])
  filter_y = np.array([[-1,-2,-1],
                       [0,0,0],
                       [1,2,1]])
  return filter_x, filter_y

# Given an image and filter, compute the filtered image.
def filter_image(im, filter):
  im = im.astype('double')
  im_filtered = np.zeros(im.shape, dtype="float")
  im_row, im_col = im.shape
  filter_row, filter_col = filter.shape

  im_padded = np.zeros((im_row + 2, im_col + 2))
  im_padded[1:im_padded.shape[0] - 1, 1:im_padded.shape[1] - 1] = im
  im_filtered = np.zeros(im.shape)
  for row in range(im_row):
    for col in range(im_col):
      im_filtered[row,col] = np.sum(filter* im_padded[row:row+filter_row, col:col+filter_col])

  return im_filtered



def visualize_find_match(img1, img2, x1, x2, img_h=500):
    assert x1.shape == x2.shape, 'x1 and x2 should have same shape!'
    scale_factor1 = img_h/img1.shape[0]
    scale_factor2 = img_h/img2.shape[0]
    img1_resized = cv2.resize(img1, None, fx=scale_factor1, fy=scale_factor1)
    img2_resized = cv2.resize(img2, None, fx=scale_factor2, fy=scale_factor2)
    x1 = x1 * scale_factor1
    x2 = x2 * scale_factor2
    x2[:, 0] += img1_resized.shape[1]
    img = np.hstack((img1_resized, img2_resized))
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    for i in range(x1.shape[0]):
        plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'b')
        plt.plot([x1[i, 0], x2[i, 0]], [x1[i, 1], x2[i, 1]], 'bo')
    plt.axis('off')
    plt.show()

def visualize_align_image(template, target, A, A_refined, errors=None):
  img_warped_init = warp_image(target, A, template.shape)
  img_warped_optim = warp_image(target, A_refined, template.shape)
  err_img_init = np.abs(img_warped_init - template)
  err_img_optim = np.abs(img_warped_optim - template)
  img_warped_init = np.uint8(img_warped_init)
  img_warped_optim = np.uint8(img_warped_optim)
  overlay_init = cv2.addWeighted(template, 0.5, img_warped_init, 0.5, 0)
  overlay_optim = cv2.addWeighted(template, 0.5, img_warped_optim, 0.5, 0)
  plt.subplot(241)
  plt.imshow(template, cmap='gray')
  plt.title('Template')
  plt.axis('off')
  plt.subplot(242)
  plt.imshow(img_warped_init, cmap='gray')
  plt.title('Initial warp')
  plt.axis('off')
  plt.subplot(243)
  plt.imshow(overlay_init, cmap='gray')
  plt.title('Overlay')
  plt.axis('off')
  plt.subplot(244)
  plt.imshow(err_img_init, cmap='jet')
  plt.title('Error map')
  plt.axis('off')
  plt.subplot(245)
  plt.imshow(template, cmap='gray')
  plt.title('Template')
  plt.axis('off')
  plt.subplot(246)
  plt.imshow(img_warped_optim, cmap='gray')
  plt.title('Opt. warp')
  plt.axis('off')
  plt.subplot(247)
  plt.imshow(overlay_optim, cmap='gray')
  plt.title('Overlay')
  plt.axis('off')
  plt.subplot(248)
  plt.imshow(err_img_optim, cmap='jet')
  plt.title('Error map')
  plt.axis('off')
  plt.show()

  if errors is not None:
    plt.plot(errors * 255)
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.show()


def visualize_track_multi_frames(template, img_list, A_list):
    bbox_list = []
    for A in A_list:
        boundary_t = np.hstack((np.array([[0, 0], [template.shape[1], 0], [template.shape[1], template.shape[0]],
                                        [0, template.shape[0]], [0, 0]]), np.ones((5, 1)))) @ A[:2, :].T
        bbox_list.append(boundary_t)

    plt.subplot(221)
    plt.imshow(img_list[0], cmap='gray')
    plt.plot(bbox_list[0][:, 0], bbox_list[0][:, 1], 'r')
    plt.title('Frame 1')
    plt.axis('off')
    plt.subplot(222)
    plt.imshow(img_list[1], cmap='gray')
    plt.plot(bbox_list[1][:, 0], bbox_list[1][:, 1], 'r')
    plt.title('Frame 2')
    plt.axis('off')
    plt.subplot(223)
    plt.imshow(img_list[2], cmap='gray')
    plt.plot(bbox_list[2][:, 0], bbox_list[2][:, 1], 'r')
    plt.title('Frame 3')
    plt.axis('off')
    plt.subplot(224)
    plt.imshow(img_list[3], cmap='gray')
    plt.plot(bbox_list[3][:, 0], bbox_list[3][:, 1], 'r')
    plt.title('Frame 4')
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
  template = cv2.imread('./Hyun_Soo_template.jpg', 0)  # read as grey scale image
  target_list = []
  for i in range(4):
      target = cv2.imread('./Hyun_Soo_target{}.jpg'.format(i+1), 0)  # read as grey scale image
      target_list.append(target)

  x1, x2 = find_match(template, target_list[0])
  visualize_find_match(template, target_list[0], x1, x2)
  
  ransac_thr, ransac_iter = 4, 1000
  A = align_image_using_feature(x1, x2, ransac_thr, ransac_iter)

  img_warped = warp_image(target_list[0], A, template.shape)
  plt.imshow(img_warped, cmap='gray', vmin=0, vmax=255)
  plt.axis('off')
  plt.show()

  A_refined, errors = align_image(template, target_list[0], A)
  visualize_align_image(template, target_list[0], A, A_refined, errors)

  A_list = track_multi_frames(template, target_list)
  visualize_track_multi_frames(template, target_list, A_list)
