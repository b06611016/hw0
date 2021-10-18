import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from sklearn.decomposition import PCA

num_person = 40
num_sample = 10
testing_set = []
training_set = []
testing_label = []
training_label = []
data_path = './p1_data'
row = 56
column = 46
num_vectors = [3, 50, 170, 240, 345]

def readfile():
    testing_set = []
    training_set = []
    testing_label = []
    training_label = []
    for i in range(1, num_person + 1):
        for j in range(1, num_sample + 1):
            img_path = os.path.join(data_path, str(i) + '_' + str(j) + '.png')
            img = mpimg.imread(img_path)
            #print(img)
            if j == num_sample:
                testing_set.append(img.flatten())
                testing_label.append(i)
            else:
                training_set.append(img.flatten())
                training_label.append(i)
    testing_set = np.array(testing_set)
    training_set = np.array(training_set)
    testing_label = np.array(testing_label)
    training_label = np.array(training_set)
    return testing_set, training_set, testing_label, training_label

def pca_reduction(eigenvectors, num_vectors, tran_img, mean_face):
    reconstructed_faces = []
    for i in range(len(num_vectors)):
        for j in range(num_vectors[i]):
            if j == 0:
                reconstruction = np.dot(np.array(eigenvectors[:, j]).flatten(), tran_img.flatten())*np.array(eigenvectors[:,j])
            else:
                reconstruction += np.dot(np.array(eigenvectors[:, j]).flatten(), tran_img.flatten())*np.array(eigenvectors[:,j])
        reconstruction = reconstruction + mean_face.reshape(2576,1)
        reconstructed_faces.append(reconstruction)
    reconstructed_faces = np.array(reconstructed_faces)
    return reconstructed_faces

def mse(training_img, new_pics, mean_face):
    pic_mse = []
    mse = 0
    for i in range(new_pics.shape[0]):
        for j in range(len(training_img)):
            mse += (255*training_img[j] - 255*new_pics[i,j,0] + 255*mean_face[j])**2
        mse = mse/len(training_img)
        pic_mse.append(mse)
        mse = 0
    return pic_mse

testing_set, training_set, testing_label, training_label = readfile()

mean_face = training_set.mean(0)
#print(mean_face.shape)
for i in range(len(training_set)):
    training_set[i] = training_set[i] - mean_face
#print(training_set.shape)
covariance_matrix = np.matrix(training_set.transpose()) * np.matrix(training_set)
#print(covariance_matrix)
#print(training_set)
covariance_matrix = covariance_matrix / training_set.shape[0]
eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
order = eigenvalues.argsort()[::-1]    #sorting in decreasing order
eigenvalues = eigenvalues[order]
#print(eigenvalues[0])
#print(eigenvectors[:,0])
eigenvectors = eigenvectors[:, order]
print(eigenvectors.shape)
plt.subplot(151)
plt.imshow(mean_face.reshape(56,46), plt.cm.gray)
plt.subplot(152)
plt.imshow(eigenvectors[:,0].reshape(56,46).real, plt.cm.gray)
plt.subplot(153)
plt.imshow(eigenvectors[:,1].reshape(56,46).real, plt.cm.gray)
plt.subplot(154)
plt.imshow(eigenvectors[:,2].reshape(56,46).real, plt.cm.gray)
plt.subplot(155)
plt.imshow(eigenvectors[:,3].reshape(56,46).real, plt.cm.gray)

plt.show()

#second question
new_face = pca_reduction(eigenvectors, num_vectors, training_set[63], mean_face.flatten())
#print(new_face.shape)
for i in range(5):
    plt.subplot(151 + i)
    plt.imshow(new_face[i, :, :].reshape(56,46).real, plt.cm.gray)
plt.show()

print(mse(training_set[63], new_face, mean_face))