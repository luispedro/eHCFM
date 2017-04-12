import numpy as np
import mahotas as mh
def sobel_edf(image):
    '''
    edf = sobel_edf(image)
    '''
    stack,h,w = image.shape
    best = np.argmax([mh.sobel(t, just_filter=True) for t in image], 0)
    image = image.reshape((stack,-1)).transpose()
    return image[np.arange(len(image)),best.ravel()].reshape((h,w))

def sobel_best(image):
    stack,h,w = image.shape
    best = np.array([mh.sobel(t, just_filter=True).mean() for t in image])
    best = best.argmax()
    return image[best]

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
