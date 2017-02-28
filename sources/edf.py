import numpy as np
import mahotas as mh


def cv(x):
    '''
    Coefficient of variation
    '''
    x = x.ravel()
    mu = x.mean()
    return x.std()/(mu + 1.e-4)


def sobel_cv_best(image):
    return image[sobel_cv_best_arg(image)]

def sobel_cv_best_arg(image):
    value = np.array([cv(mh.sobel(t, just_filter=True)) for t in image])
    return value.argmax()
