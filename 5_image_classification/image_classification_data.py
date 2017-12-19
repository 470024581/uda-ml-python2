import helper
import numpy as np

cifar10_dataset_folder_path = 'cifar-10-batches-py'
import problem_unittests as tests
tests.test_folder_path(cifar10_dataset_folder_path)

# 归一化处理
from sklearn.preprocessing import MinMaxScaler
def normalize(x):
    """
    Normalize a list of sample image data in the range of 0 to 1
    : x: List of image data.  The image shape is (32, 32, 3)
    : return: Numpy array of normalize data
    """
    print(x.shape) #(562, 32, 32, 3)
#     (x-np.min(x))/(np.max(x)-np.min(x))

    arr_3d = []
    arr_2d = []
    for image_set in x :
        for image in image_set :
            minmax_2d = MinMaxScaler().fit_transform(image)
            if len(arr_2d) == 0:
                arr_2d = minmax_2d
                continue
            arr_2d = np.vstack((arr_2d, minmax_2d))
        arr_2d = arr_2d.reshape(x.shape[1],x.shape[2],x.shape[3])
        if len(arr_3d) == 0:
            arr_3d = arr_2d
            arr_2d = []
            continue
        arr_3d = np.vstack((arr_3d, arr_2d))
        arr_2d = []
    arr_3d = arr_3d.reshape(len(x),x.shape[1],x.shape[2],x.shape[3])
#     print(arr_3d.shape)
#     print(x.shape)
#     print(arr_3d.max())
    return arr_3d

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_normalize(normalize)

# 二值化处理
from sklearn.preprocessing import LabelBinarizer
label_binarizer = LabelBinarizer() 
label_binarizer.fit(range(10))

def one_hot_encode(x):
    """
    One hot encode a list of sample labels. Return a one-hot encoded vector for each label.
    : x: List of sample Labels
    : return: Numpy array of one-hot encoded labels
    """
    # TODO: Implement Function
# lb.fit([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    return label_binarizer.fit_transform(x)


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_one_hot_encode(one_hot_encode)

# helper.preprocess_and_save_data(cifar10_dataset_folder_path, normalize, one_hot_encode)
