import numpy as np
import mahotas as mh

def roundness(channel):
    import mahotas.features.shape
    r = (mh.dilate(channel > channel.mean() + channel.std()*3))
    r  = mh.open(r, np.ones((5,5)))
    return mahotas.features.shape.roundness(r)

def corr1(x,y, t=8):
    x = x.ravel()
    y = y.ravel()
    valid = np.zeros(1)
    while not valid.sum():
        valid = (x >= t)|(y >= t)
        t //= 2
    return np.corrcoef(x[valid],y[valid])[0,1]

def bcorr1(x,y, t=8):
    x = x.ravel()
    y = y.ravel()

    x = (x > x.mean() + x.std()).astype(float)
    y = (y > y.mean() + y.std()).astype(float)

    return np.corrcoef(x,y)[0,1]

def central_zernike(im):
    '''Compute Zernike moments
    '''
    area = np.sum(mh.gaussian_filter(im, 1.4, mode='constant')> im.mean())
    area = np.sqrt(area)
    area /= np.pi
    return mh.features.zernike_moments(im, area)

def moments(im):
    '''Compute image moments

    Parameters
    ----------
    im : 2D ndarray
        input image

    Returns
    -------
    fs : ndarray
        Four basic moments
    '''
    def mhmoments(y, x):
        return mh.features.moments.moments(im, y, x, normalize=1)
    if im.ndim != 2:
        raise ValueError('moments expected 2D array')
    return [
            mhmoments(1, 1),
            (mhmoments(1, 2) + mhmoments(2, 1))/2.,
            mhmoments(2, 2),
            ]

def _safe_haralick(im):
    fs = mh.features.haralick(im.astype(np.uint8), ignore_zeros=True).mean(0)
    if not np.any(np.isnan(fs)): return fs
    return np.zeros(13)
