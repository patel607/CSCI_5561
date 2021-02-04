import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC
from scipy import stats
from pathlib import Path, PureWindowsPath


def extract_dataset_info(data_path):
  # extract information from train.txt
  f = open(os.path.join(data_path, "train.txt"), "r")
  contents_train = f.readlines()
  label_classes, label_train_list, img_train_list = [], [], []
  for sample in contents_train:
    sample = sample.split()
    label, img_path = sample[0], sample[1]
    if label not in label_classes:
      label_classes.append(label)
    label_train_list.append(sample[0])
    img_train_list.append(os.path.join(data_path, Path(PureWindowsPath(img_path))))
  print('Classes: {}'.format(label_classes))

  # extract information from test.txt
  f = open(os.path.join(data_path, "test.txt"), "r")
  contents_test = f.readlines()
  label_test_list, img_test_list = [], []
  for sample in contents_test:
    sample = sample.split()
    label, img_path = sample[0], sample[1]
    label_test_list.append(label)
    img_test_list.append(os.path.join(data_path, Path(PureWindowsPath(img_path))))  # you can directly use img_path if you run in Windows

  return label_classes, label_train_list, img_train_list, label_test_list, img_test_list


def compute_dsift(img, stride, size):  
  kp = []
  # collect keypoints for stride and size
  for i in range(0, img.shape[0], stride):
    for j in range(0, img.shape[1], stride):
      kp.append(cv2.KeyPoint(i + size, j + size, size))
      
  sift = cv2.xfeatures2d.SIFT_create()
  extra, dense_feature = sift.compute(img, kp)
  # dense_feature should be number of locations x 128
  return dense_feature

# resize the image to output size and normalize it
def get_tiny_image(img, output_size):
  feature = cv2.resize(img, output_size, interpolation = cv2.INTER_AREA)
  feature = feature / np.linalg.norm(feature)
  return feature

# use nearest neighbors to predict test set after fitting training set
def predict_knn(feature_train, label_train, feature_test, k):
  knn = KNeighborsClassifier(k).fit(feature_train, label_train)
  label_test_pred = knn.predict(feature_test)
  return label_test_pred


def classify_knn_tiny(label_classes, label_train_list, img_train_list, label_test_list, img_test_list):
  accuracy = 0
  confusion = np.zeros((15,15))
  feature_train = np.zeros((0,256))
  label_train = []
  feature_test = np.zeros((0,256))
  # format training set as necessary
  for i in range(len(img_train_list)):
    img = cv2.imread(img_train_list[i], 0)
    represent = get_tiny_image(img, (16,16)).flatten()
    feature_train = np.append(feature_train, [represent], axis = 0)
    label_train.append(label_train_list[i])

  # format test set as necessary
  for i in range(len(img_test_list)):
    img = cv2.imread(img_test_list[i], 0)
    represent = get_tiny_image(img, (16,16)).flatten()
    feature_test = np.append(feature_test, [represent], axis = 0)

  label_test_pred = predict_knn(feature_train, label_train, feature_test, 1)

  label_classes_np = np.array(label_classes)
  label_test_list_np = np.array(label_test_list)

  #number_of_bedroom_data = len(np.where(label_test_list == "Bedroom"))

  for i in range(len(label_test_pred)):
    # get indices for confusion matrix
    truth = np.where(label_classes_np == label_test_list[i])[0][0]
    pred = np.where(label_classes_np == label_test_pred[i])[0][0]

    confusion[truth][pred] += 1

  # divide by total number of entries of row; also add up diagonal for accuracy
  for i in range(confusion.shape[0]):
    confusion[i] /= len(np.where(label_test_list_np == label_classes[i])[0])
    accuracy += confusion[i][i]/15

  #visualize_confusion_matrix(confusion, accuracy, label_classes)
  return confusion, accuracy


def build_visual_dictionary(dense_feature_list, dic_size):
  pool = []
  # dense_feature_list is a pool of sift descriptors but is shaped number of images x number of descriptors x 128
  for i in range(len(dense_feature_list)):
    for j in range(len(dense_feature_list[i])):
      pool.append(dense_feature_list[i][j])
  # resize so all images are flattened, just rows of sift descriptors now
  kmeans = KMeans(n_clusters = dic_size, n_init=10, max_iter=100)
  kmeans.fit(pool)
  vocab = kmeans.cluster_centers_
  # cluster centers make up the dicionary
  np.savetxt('visual_dictionary2.txt', vocab)
  return vocab


def compute_bow(feature, vocab):
  # labels for each dictionary item
  y = np.arange(0, vocab.shape[0])
  knn = KNeighborsClassifier(n_neighbors=1).fit(vocab, y) 
  pred = knn.predict(feature)
  # bincount gives histogram values and then we normalize
  bow_feature = np.bincount(pred, minlength = vocab.shape[0])
  bow_feature = bow_feature/np.linalg.norm(bow_feature)

  return bow_feature


def classify_knn_bow(label_classes, label_train_list, img_train_list, label_test_list, img_test_list):
  stride = 16
  size = 16
  dic_size = 50
  feature_train = []
  feature_test = []
  label_train = []
  # COMMENTED CODE IS TO BUILD VISUAL DICTIONARY
  # dense_feature_list = []
  # for i in range(len(img_train_list)):
  #   img = cv2.imread(img_train_list[i], 0)
  #   sifted = compute_dsift(img, stride, size)
  #   dense_feature_list.append(sifted)
    
  # print("Starting to build dictionary")
  # vocab = build_visual_dictionary(dense_feature_list, dic_size)
  # print("Finished building dictionary")
  vocab = np.loadtxt("visual_dictionary.txt")

  # feature_train is composed of bag-of-words histograms
  for i in range(len(img_train_list)):
    img = cv2.imread(img_train_list[i], 0)
    feature = compute_dsift(img, stride, size)
    bow_feature = compute_bow(feature, vocab)
    feature_train.append(bow_feature)
    label_train.append(label_train_list[i])
  print("feature train done")

  # same with feature_test
  for i in range(len(img_test_list)):
    img = cv2.imread(img_test_list[i], 0)
    feature = compute_dsift(img, stride, size)
    bow_feature = compute_bow(feature, vocab)
    feature_test.append(bow_feature)
  print("feature test done")
  
  label_test_pred = predict_knn(feature_train, label_train, feature_test, 1)
  print("Predictions made")

  label_classes_np = np.array(label_classes)
  label_test_list_np = np.array(label_test_list)

  accuracy = 0
  confusion = np.zeros((15,15))

  for i in range(len(label_test_pred)):
    # get indices for confusion matrix
    truth = np.where(label_classes_np == label_test_list[i])[0][0]
    pred = np.where(label_classes_np == label_test_pred[i])[0][0]

    confusion[truth][pred] += 1

  # divide by total number of entries of row; also add up diagonal for accuracy
  for i in range(confusion.shape[0]):
    confusion[i] /= len(np.where(label_test_list_np == label_classes[i])[0])
    accuracy += confusion[i][i]/15

  #visualize_confusion_matrix(confusion, accuracy, label_classes)
  return confusion, accuracy


def predict_svm(feature_train, label_train, feature_test, n_classes):
  # lambda value
  C = 8
  label_test_pred = np.zeros(len(feature_test))
  # area of confidence values for each classfier
  confidence = np.zeros((n_classes, len(feature_train)))
  for c in range(n_classes):
    svm = LinearSVC(C = C)
    # labels all occurences of current label as 1, and everything else as 0
    labels_new = list(map(lambda x:1 if x == (c+1) else 0, label_train))
    # decision_function returns list of confidence scores for test set
    confidence[c] = np.array(svm.fit(feature_train, labels_new).decision_function(feature_test))

  # get the index of the classifier for the maximum score for the particular test image
  for i in range(len(feature_test)):
    label_test_pred[i] = np.argmax(confidence[:,i])
    
  return label_test_pred


def classify_svm_bow(label_classes, label_train_list, img_train_list, label_test_list, img_test_list):
  stride = 16
  size = 16
  dic_size = 50
  feature_train = []
  feature_test = []
  label_train = []
  # COMMENTED CODE IS TO BUILD VISUAL DICTIONARY
  # dense_feature_list = []
  # for i in range(len(img_train_list)):
  #   img = cv2.imread(img_train_list[i], 0)
  #   sifted = compute_dsift(img, stride, size)
  #   dense_feature_list.append(sifted)
    
  # print("Starting to build dictionary")
  # vocab = build_visual_dictionary(dense_feature_list, dic_size)
  # print("Finished building dictionary")
  vocab = np.loadtxt("visual_dictionary.txt")

  label_classes_np = np.array(label_classes)
  # still composed of bag-of-word histograms
  for i in range(len(img_train_list)):
    img = cv2.imread(img_train_list[i], 0)
    feature = compute_dsift(img, stride, size)
    bow_feature = compute_bow(feature, vocab)
    feature_train.append(bow_feature)
    # translate labels to indices in label_classes
    label_train_ind = np.where(label_classes_np == label_train_list[i])[0][0] + 1
    label_train.append(label_train_ind)
  print("feature train done")

  for i in range(len(img_test_list)):
    img = cv2.imread(img_test_list[i], 0)
    feature = compute_dsift(img, stride, size)
    bow_feature = compute_bow(feature, vocab)
    feature_test.append(bow_feature)
  print("feature test done")

  label_test_pred = predict_svm(feature_train, label_train, feature_test, len(label_classes))
  print("Predictions made")

  
  label_test_list_np = np.array(label_test_list)

  accuracy = 0
  confusion = np.zeros((15,15))

  for i in range(len(label_test_pred)):
    # get indices for confusion matrix
    truth = np.where(label_classes_np == label_test_list[i])[0][0]
    pred = int(label_test_pred[i])
    confusion[truth][pred] += 1

  # divide by total number of entries of row; also add up diagonal for accuracy
  for i in range(confusion.shape[0]):
    confusion[i] /= len(np.where(label_test_list_np == label_classes[i])[0])
    accuracy += confusion[i][i]/15
  #visualize_confusion_matrix(confusion, accuracy, label_classes)
  return confusion, accuracy


def visualize_confusion_matrix(confusion, accuracy, label_classes):
  plt.title("accuracy = {:.3f}".format(accuracy))
  plt.imshow(confusion)
  ax, fig = plt.gca(), plt.gcf()
  plt.xticks(np.arange(len(label_classes)), label_classes)
  plt.yticks(np.arange(len(label_classes)), label_classes)
  # set horizontal alignment mode (left, right or center) and rotation mode(anchor or default)
  plt.setp(ax.get_xticklabels(), rotation=-30, ha="center", rotation_mode="default")
  # avoid top and bottom part of heatmap been cut
  ax.set_xticks(np.arange(len(label_classes) + 1) - .5, minor=True)
  ax.set_yticks(np.arange(len(label_classes) + 1) - .5, minor=True)
  ax.tick_params(which="minor", bottom=False, left=False)
  fig.tight_layout()
  plt.show()


if __name__ == '__main__':
  # To do: replace with your dataset path
  label_classes, label_train_list, img_train_list, label_test_list, img_test_list = extract_dataset_info("./scene_classification_data")
  
  classify_knn_tiny(label_classes, label_train_list, img_train_list, label_test_list, img_test_list)
  
  classify_knn_bow(label_classes, label_train_list, img_train_list, label_test_list, img_test_list)
  
  classify_svm_bow(label_classes, label_train_list, img_train_list, label_test_list, img_test_list)
  # img = cv2.imread('./image_0043.jpg', 0)
  # plt.imshow(img, cmap='gray')
  # plt.show()
  # tiny_img = get_tiny_image(img, (20,20))
  # plt.imshow(tiny_img, cmap='gray')
  # plt.show()




