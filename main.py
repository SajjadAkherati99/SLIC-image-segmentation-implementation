import cv2
import numpy as np
from scipy import ndimage
from skimage.segmentation import find_boundaries
from skimage.segmentation import slic
from skimage import img_as_float

def initialing_clusters(img, k):
    img_shape = img.shape[0:2]
    k_axis = int(np.sqrt(k))
    l_x, l_y = int(img_shape[0]/k_axis), int(img_shape[1]/k_axis)
    cluster_center = np.zeros([k, 2])
    for i in range(k):
        cluster_center[i, 0] = int((i%k_axis)*l_x + l_x/2)
        cluster_center[i, 1] = int(int(i/k_axis) * l_y + l_y / 2)

    return  np.int_(cluster_center)

def find_cluster_centers_gradient_method(features, initial_centers, k):
    kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    kernel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    sobel = np.zeros(features.shape[0:2])
    for i in range(0, features.shape[2]):
        edge_x = ndimage.convolve(features[:, :, i], kernel_x)
        edge_y = ndimage.convolve(features[:, :, i], kernel_y)
        sol = np.hypot(edge_x, edge_y)
        sol = sol/np.max(np.abs(sol))
        sobel[:, :] = sobel + sol**2
    centers = initial_centers
    for k in range(k):
        center_x, center_y = initial_centers[k, 0], initial_centers[k, 1]
        cx, cy = center_x, center_y
        for i in range(cx-2, cx+3):
            if (i >= 0 & i  < features.shape[0]):
                for j in range(cy-2, cy+3):
                    if (j >= 0 & j < features.shape[1]):
                        if sobel[i, j] < sobel[center_x, center_y]:
                            center_x, center_y = i, j
            centers[k, 0], centers[k, 1] = center_x, center_y
    return centers

def cluster_shape(centers, k_sqr):
    cluster_centers = np.zeros([k_sqr, k_sqr, 2])
    for i in range(k_sqr):
        for j in range(k_sqr):
            cluster_centers[i, j, 0], cluster_centers[i, j, 1] = centers[j+i*k_sqr, 0], centers[j+i*k_sqr, 1]

    return np.int_(cluster_centers)

def initial_distance(img_shape):
    distance = np.inf*np.ones(img_shape)
    return  distance

def make_features_xys(img):
    features = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    x_len, y_len = img.shape[0], img.shape[1]
    xs = np.transpose(np.array(y_len*[np.array(range(x_len))]))
    ys = np.array(x_len*[np.array(range(y_len))])
    # np.info(xs)
    # np.info(ys)
    xys_shape = (x_len, y_len, 2)
    xys = np.zeros(xys_shape)
    xys[:, :, 0] = xs
    xys[:, :, 1] = ys
    return features, np.int_(xys)

def update_distance_labels(features, xys, center, l, k, distances, labels, a = 10):
    # k_sqr = int(np.sqrt(k))
    # center_dist_x, center_dist_y = int(features.shape[0]/k_sqr), int(features.shape[1]/k_sqr)
    length = int(np.sqrt(features.shape[0]*features.shape[1]/k))
    center_dist_x, center_dist_y = length, length
    src_x, src_y = max(center[0]-center_dist_x, 0), max(center[1]-center_dist_y, 0)
    dest_x, dest_y = min(center[0]+center_dist_x, features.shape[0]), min(center[1]+center_dist_y, features.shape[1])
    features_i = np.copy(features[src_x:dest_x, src_y:dest_y, :])
    xys_i = np.copy(xys[src_x:dest_x, src_y:dest_y, :])
    distances_i = np.copy(distances[src_x:dest_x, src_y:dest_y])
    labels_i = np.copy(labels[src_x:dest_x, src_y:dest_y])

    for i in range(features_i.shape[2]):
        features_i[:, :, i] = features_i[:, :, i] - features[center[0], center[1], i]

    for i in range(xys_i.shape[2]):
        xys_i[:, :, i] = xys_i[:, :, i] - xys[center[0], center[1], i]

    new_distance_features_i = np.zeros(distances_i.shape)
    for i in range(features_i.shape[2]):
        new_distance_features_i += features_i[:, :, i]**2

    new_distance_xys_i = np.zeros(distances_i.shape)
    for i in range(xys_i.shape[2]):
        new_distance_xys_i += xys_i[:, :, i]**2

    a0 = a/length
    new_distances_i = np.sqrt(new_distance_features_i) + a0 * np.sqrt(new_distance_xys_i)
    x, y = np.where(new_distances_i < distances_i)
    distances_i[x, y] = new_distances_i[x, y]
    labels_i[x, y] = l

    new_distances = distances
    new_labels = labels
    new_distances[src_x:dest_x, src_y:dest_y] = distances_i
    new_labels[src_x:dest_x, src_y:dest_y] = labels_i
    return new_distances, new_labels

def initial_label(len_x, len_y):
    labels = np.zeros([len_x, len_y])
    return np.int_(labels)

def update_cluster_centers(cluster_centers, labels, k):
    k_sqr = int(np.sqrt(k))
    new_cluster_centers = cluster_centers
    for i in range(k):
        x, y = np.where(labels == i)
        if (len(x) > 0):
            new_cluster_centers[i//k_sqr, i%k_sqr, 0], new_cluster_centers[i//k_sqr, i%k_sqr, 1] = int(np.mean(x)), int(np.mean(y))
    return  new_cluster_centers

def Slic(features0, xys0, cluster_centers,  k, distance, labels, a,iterations= 10):
    new_labels = np.copy(labels)
    new_distance = np.copy(distance)
    xys, features = np.copy(xys0), np.copy(features0)
    for i in range(iterations):
        l = 0
        for row in cluster_centers:
            for center in row:
                new_distance, new_labels = update_distance_labels(features, xys, center, l, k,
                                                                  new_distance, new_labels, a)
                l += 1
        cluster_centers = update_cluster_centers(cluster_centers, new_labels, k)
    return np.int_(new_labels)

img =  cv2.imread('slic.jpg')
img_shape = img.shape
length = int((img.shape[0]/8))
width = int((img.shape[1]/8))
img = cv2.resize(img, (width, width), interpolation= cv2.INTER_AREA)
img0 = np.copy(img)
img1 = np.copy(img0)
j = 5
for k, a, s in zip([64, 256, 1024, 2048], [0.2 ,0.2 , 0.2 , 0.15], [2.5, 2.5, 3, 4]):
    sigma = np.array([s, s, s])
    img0 = ndimage.gaussian_filter(img0, sigma)
    img = np.copy(img0)
    k_sqr = int(np.sqrt(k))
    cluster_centers = initialing_clusters(img, k)
    cluster_centers = find_cluster_centers_gradient_method(img, cluster_centers, k)
    cluster_centers = cluster_shape(cluster_centers, k_sqr)
    distances = initial_distance(img.shape[0:2])
    features, xys = make_features_xys(img)
    features = img_as_float(features)
    labels = initial_label(features.shape[0], features.shape[1])

    labels = Slic(features, xys, cluster_centers,  k, distances, labels, a, iterations= 10)
    labels_unique = np.unique(labels)
    n_cluster = len(labels_unique)
    print(n_cluster)
    boundaries = find_boundaries(labels, mode='outer', connectivity=1).astype(np.uint8)
    x, y = np.where(boundaries==1)
    img = np.copy(img1)
    img[x, y, :] = 0
    img = cv2.resize(img, (img_shape[1], img_shape[0]), interpolation= cv2.INTER_AREA)
    cv2.imwrite('res0{i}.jpg'.format(i = j), img)
    j += 1

# img = cv2.imread('slic.jpg')
# img_shape = img.shape
# length = int((img.shape[0]/8))
# width = int((img.shape[1]/8))
# img = cv2.resize(img, (width, length), interpolation= cv2.INTER_AREA)
# segments = slic(img, n_segments=64, sigma=5, enforce_connectivity=0, max_iter=10)
# # np.info(segments)
# labels_unique = np.unique(segments)
# # # print(labels_unique)
# n_cluster = len(labels_unique)
# print(n_cluster)
# boundaries = find_boundaries(segments, mode='outer').astype(np.uint8)
# x, y = np.where(boundaries==1)
# img[x, y, :] = 0
# img = cv2.resize(img, (img_shape[1], img_shape[0]), interpolation= cv2.INTER_AREA)
# cv2.imwrite('res01.jpg', img)
