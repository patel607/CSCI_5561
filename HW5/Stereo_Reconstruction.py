import cv2
import numpy as np
import scipy.io as sio
from scipy.linalg import null_space
import matplotlib.pyplot as plt
import sys
from sklearn.neighbors import NearestNeighbors
from mpl_toolkits.mplot3d import Axes3D

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


def compute_F(pts1, pts2):
    ransac_iter = 1000
    threshold = 0.05
    max_inliers = -1
    F = None
    for i in range(ransac_iter):
        eight_point = np.random.choice(pts1.shape[0], 8, replace=False)
        pts1_eight = pts1[eight_point, :]
        pts2_eight = pts2[eight_point, :]
        A = np.zeros((8, 9))
        for j in range(8):
            u = pts1_eight[j]
            v = pts2_eight[j]
            A[j, 0] = u[0] * v[0]
            A[j, 1] = u[1] * v[0]
            A[j, 2] = v[0]
            A[j, 3] = u[0] * v[1]
            A[j, 4] = u[1] * v[1]
            A[j, 5] = v[1]
            A[j, 6] = u[0]
            A[j, 7] = u[1]
            A[j, 8] = 1

        F_1 = null_space(A)
        F_1 = F_1[:, 0].reshape(3, 3)

        u, s, vh = np.linalg.svd(F_1)
        s[2] = 0
        F_clean = np.matmul( np.matmul(u, np.diag(s)) , vh)

        num_inlier = 0
        for j in range(len(pts1)):
            u = np.array([pts1[j,0], pts1[j,1], 1])
            v = np.array([pts2[j,0], pts2[j,1], 1])
            # err = np.matmul( np.matmul(v.T, F_clean) , u)
            err = np.matmul(v.T, np.matmul(F_clean, u))
            if np.abs(err) < threshold:
                num_inlier += 1

        if num_inlier > max_inliers:
            max_inliers = num_inlier
            F = F_clean

    # F = np.asarray(F)
    print(num_inlier)
    return F


def skew(vector):
    a, b, c = vector
    skew = np.asarray([
        [0, -c, b],
        [c, 0, -a],
        [-b, a, 0],
    ])
    return skew

def triangulation(P1, P2, pts1, pts2):
    # TO DO
    pts3D = np.zeros((len(pts1), 3))
    for i in range(len(pts1)):
        u = skew(np.append(pts1[i,:], 1))
        v = skew(np.append(pts2[i,:], 1))

        u_P1 = np.matmul(u, P1)
        v_P2 = np.matmul(v, P2)

        A = np.concatenate((u_P1, v_P2), axis=0)
        x = null_space(A, rcond=0.1)
        x = x[:,0]
        x = x/x[3]
        pts3D[i,:] = x[:3]

    return pts3D


def disambiguate_pose(Rs, Cs, pts3Ds):
    # TO DO
    best_idx = -1
    max_pts = -1

    for i in range(len(Rs)):
        nValid = 0
        c = Cs[i].reshape((3,))
        r = Rs[i][2,:]

        for p in pts3Ds[i]:
            cam_1 = (p-c)[2]
            cam_2 = np.matmul(p-c, r)
            if cam_1 > 0 and cam_2 > 0:
                nValid += 1
        if nValid > max_pts:
            max_pts = nValid
            best_idx = i

    R = Rs[best_idx]
    C = Cs[best_idx]
    pts3D = pts3Ds[best_idx]
    return R, C, pts3D


def compute_rectification(K, R, C):
    # TO DO
    # C = C.reshape(3,)
    C = C.reshape(-1)

    R_x = -C / np.linalg.norm(C)
    z_tilde = np.array([0,0,1])
    R_z = z_tilde - R_x * np.dot(z_tilde, R_x)
    R_z = R_z / np.linalg.norm(R_z)
    R_y = np.cross(R_z, R_x)

    R_rect = np.asarray([R_x, R_y, R_z])

    H1 = K @ R_rect @ np.linalg.inv(K)
    H2 = K @ R_rect @ R.T @ np.linalg.inv(K)
    return H1, H2


def dense_match(img1, img2):
    # TO DO
    sift = cv2.xfeatures2d.SIFT_create()
    size = 10
    kp = []
    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            kp.append(cv2.KeyPoint(j, i, size))

    kp_1, des_1 = sift.compute(img1, kp)

    kp = []
    for i in range(img2.shape[0]):
        for j in range(img2.shape[1]):
            kp.append(cv2.KeyPoint(j, i, size))

    kp_2, des_2 = sift.compute(img2, kp)

    des_1 = des_1.reshape((img1.shape[0], img1.shape[1], 128))
    des_2 = des_2.reshape((img2.shape[0], img2.shape[1], 128))
    disparity = np.zeros(img1.shape)

    for u in range(img1.shape[0]):
        for i in range(img1.shape[1]):
            if img1[u, i] == 0:
                continue
            d_1 = des_1[u, i]
            dist = []
            for j in range(i+1):
                d_2 = des_2[u, j]
                dist.append(np.linalg.norm(d_1 - d_2)**2)
            disparity[u, i] = np.abs(np.argmin(dist) - i)

    return disparity


# PROVIDED functions
def compute_camera_pose(F, K):
    E = K.T @ F @ K
    R_1, R_2, t = cv2.decomposeEssentialMat(E)
    # 4 cases
    R1, t1 = R_1, t
    R2, t2 = R_1, -t
    R3, t3 = R_2, t
    R4, t4 = R_2, -t

    Rs = [R1, R2, R3, R4]
    ts = [t1, t2, t3, t4]
    Cs = []
    for i in range(4):
        Cs.append(-Rs[i].T @ ts[i])
    return Rs, Cs


def visualize_img_pair(img1, img2):
    img = np.hstack((img1, img2))
    if img1.ndim == 3:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.show()


def visualize_find_match(img1, img2, pts1, pts2):
    assert pts1.shape == pts2.shape, 'x1 and x2 should have same shape!'
    img_h = img1.shape[0]
    scale_factor1 = img_h/img1.shape[0]
    scale_factor2 = img_h/img2.shape[0]
    img1_resized = cv2.resize(img1, None, fx=scale_factor1, fy=scale_factor1)
    img2_resized = cv2.resize(img2, None, fx=scale_factor2, fy=scale_factor2)
    pts1 = pts1 * scale_factor1
    pts2 = pts2 * scale_factor2
    pts2[:, 0] += img1_resized.shape[1]
    img = np.hstack((img1_resized, img2_resized))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    for i in range(pts1.shape[0]):
        plt.plot([pts1[i, 0], pts2[i, 0]], [pts1[i, 1], pts2[i, 1]], 'b.-', linewidth=0.5, markersize=5)
    plt.axis('off')
    plt.show()


def visualize_epipolar_lines(F, pts1, pts2, img1, img2):
    assert pts1.shape == pts2.shape, 'x1 and x2 should have same shape!'
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)
    ax1.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    ax2.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))

    for i in range(pts1.shape[0]):
        x1, y1 = int(pts1[i][0] + 0.5), int(pts1[i][1] + 0.5)
        ax1.scatter(x1, y1, s=5)
        p1, p2 = find_epipolar_line_end_points(img2, F, (x1, y1))
        ax2.plot([p1[0], p2[0]], [p1[1], p2[1]], linewidth=0.5)

    for i in range(pts2.shape[0]):
        x2, y2 = int(pts2[i][0] + 0.5), int(pts2[i][1] + 0.5)
        ax2.scatter(x2, y2, s=5)
        p1, p2 = find_epipolar_line_end_points(img1, F.T, (x2, y2))
        ax1.plot([p1[0], p2[0]], [p1[1], p2[1]], linewidth=0.5)

    ax1.axis('off')
    ax2.axis('off')
    plt.show()


def find_epipolar_line_end_points(img, F, p):
    img_width = img.shape[1]
    el = np.dot(F, np.array([p[0], p[1], 1]).reshape(3, 1))
    p1, p2 = (0, -el[2] / el[1]), (img.shape[1], (-img_width * el[0] - el[2]) / el[1])
    _, p1, p2 = cv2.clipLine((0, 0, img.shape[1], img.shape[0]), p1, p2)
    return p1, p2


def visualize_camera_poses(Rs, Cs):
    assert(len(Rs) == len(Cs) == 4)
    fig = plt.figure()
    R1, C1 = np.eye(3), np.zeros((3, 1))
    for i in range(4):
        R2, C2 = Rs[i], Cs[i]
        ax = fig.add_subplot(2, 2, i+1, projection='3d')
        draw_camera(ax, R1, C1)
        draw_camera(ax, R2, C2)
        set_axes_equal(ax)
        ax.set_xlabel('x axis')
        ax.set_ylabel('y axis')
        ax.set_zlabel('z axis')
        ax.view_init(azim=-90, elev=0)
    fig.tight_layout()
    plt.show()


def visualize_camera_poses_with_pts(Rs, Cs, pts3Ds):
    assert(len(Rs) == len(Cs) == 4)
    fig = plt.figure()
    R1, C1 = np.eye(3), np.zeros((3, 1))
    for i in range(4):
        R2, C2, pts3D = Rs[i], Cs[i], pts3Ds[i]
        ax = fig.add_subplot(2, 2, i+1, projection='3d')
        draw_camera(ax, R1, C1, 5)
        draw_camera(ax, R2, C2, 5)
        ax.plot(pts3D[:, 0], pts3D[:, 1], pts3D[:, 2], 'b.')
        set_axes_equal(ax)
        ax.set_xlabel('x axis')
        ax.set_ylabel('y axis')
        ax.set_zlabel('z axis')
        ax.view_init(azim=-90, elev=0)
    fig.tight_layout()
    plt.show()


def draw_camera(ax, R, C, scale=0.2):
    axis_end_points = C + scale * R.T  # (3, 3)
    vertices = C + scale * R.T @ np.array([[1, 1, 1], [-1, 1, 1], [-1, -1, 1], [1, -1, 1]]).T  # (3, 4)
    vertices_ = np.hstack((vertices, vertices[:, :1]))  # (3, 5)

    # draw coordinate system of camera
    ax.plot([C[0], axis_end_points[0, 0]], [C[1], axis_end_points[1, 0]], [C[2], axis_end_points[2, 0]], 'r-')
    ax.plot([C[0], axis_end_points[0, 1]], [C[1], axis_end_points[1, 1]], [C[2], axis_end_points[2, 1]], 'g-')
    ax.plot([C[0], axis_end_points[0, 2]], [C[1], axis_end_points[1, 2]], [C[2], axis_end_points[2, 2]], 'b-')

    # draw square window and lines connecting it to camera center
    ax.plot(vertices_[0, :], vertices_[1, :], vertices_[2, :], 'k-')
    ax.plot([C[0], vertices[0, 0]], [C[1], vertices[1, 0]], [C[2], vertices[2, 0]], 'k-')
    ax.plot([C[0], vertices[0, 1]], [C[1], vertices[1, 1]], [C[2], vertices[2, 1]], 'k-')
    ax.plot([C[0], vertices[0, 2]], [C[1], vertices[1, 2]], [C[2], vertices[2, 2]], 'k-')
    ax.plot([C[0], vertices[0, 3]], [C[1], vertices[1, 3]], [C[2], vertices[2, 3]], 'k-')


def set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range, x_middle = abs(x_limits[1] - x_limits[0]), np.mean(x_limits)
    y_range, y_middle = abs(y_limits[1] - y_limits[0]), np.mean(y_limits)
    z_range, z_middle = abs(z_limits[1] - z_limits[0]), np.mean(z_limits)

    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def visualize_disparity_map(disparity):
    plt.imshow(disparity, cmap='jet')
    plt.show()


if __name__ == '__main__':
    # read in left and right images as RGB images
    img_left = cv2.imread('./left.bmp', 1)
    img_right = cv2.imread('./right.bmp', 1)
    # visualize_img_pair(img_left, img_right)

    # Step 1: find correspondences between image pair
    pts1, pts2 = find_match(img_left, img_right)
    # visualize_find_match(img_left, img_right, pts1, pts2)

    # Step 2: compute fundamental matrix
    F = compute_F(pts1, pts2)
    # visualize_epipolar_lines(F, pts1, pts2, img_left, img_right)

    # Step 3: computes four sets of camera poses
    K = np.array([[350, 0, 960/2], [0, 350, 540/2], [0, 0, 1]])
    Rs, Cs = compute_camera_pose(F, K)
    # visualize_camera_poses(Rs, Cs)

    # Step 4: triangulation
    pts3Ds = []
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    for i in range(len(Rs)):
        P2 = K @ np.hstack((Rs[i], -Rs[i] @ Cs[i]))
        pts3D = triangulation(P1, P2, pts1, pts2)
        pts3Ds.append(pts3D)
    # visualize_camera_poses_with_pts(Rs, Cs, pts3Ds)

    # Step 5: disambiguate camera poses
    R, C, pts3D = disambiguate_pose(Rs, Cs, pts3Ds)

    # Step 6: rectification
    H1, H2 = compute_rectification(K, R, C)
    img_left_w = cv2.warpPerspective(img_left, H1, (img_left.shape[1], img_left.shape[0]))
    img_right_w = cv2.warpPerspective(img_right, H2, (img_right.shape[1], img_right.shape[0]))
    # visualize_img_pair(img_left_w, img_right_w)

    # Step 7: generate disparity map
    img_left_w = cv2.resize(img_left_w, (int(img_left_w.shape[1] / 2), int(img_left_w.shape[0] / 2)))  # resize image for speed
    img_right_w = cv2.resize(img_right_w, (int(img_right_w.shape[1] / 2), int(img_right_w.shape[0] / 2)))
    img_left_w = cv2.cvtColor(img_left_w, cv2.COLOR_BGR2GRAY)  # convert to gray scale
    img_right_w = cv2.cvtColor(img_right_w, cv2.COLOR_BGR2GRAY)
    disparity = dense_match(img_left_w, img_right_w)
    # visualize_disparity_map(disparity)

    # # save to mat
    sio.savemat('stereo.mat', mdict={'pts1': pts1, 'pts2': pts2, 'F': F, 'pts3D': pts3D, 'H1': H1, 'H2': H2,
                                     'img_left_w': img_left_w, 'img_right_w': img_right_w, 'disparity': disparity})
