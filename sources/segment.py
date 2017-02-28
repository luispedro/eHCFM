import numpy as np
import mahotas as mh
from labeled_filter import filter_labeled
from globals import VOXEL_SIZE_MU_CUBED

FL_CHANNELS = ['dna', 'auto', 'lysine', 'DiO6']
MIN_THRESHOLD=1.5
MIN_SIZE_SUBOBJECT_DIAMETER = 0.5 # mu microns
MIN_SIZE_SUBOBJECT_VOLUME = 4./3 * np.pi * MIN_SIZE_SUBOBJECT_DIAMETER**3
MIN_NR_VOXELS_SUBOBJECT = int(np.round(MIN_SIZE_SUBOBJECT_VOLUME/VOXEL_SIZE_MU_CUBED))

def almost_max(im, axis=0, in_place=False):
    if not in_place:
        im = im.copy()
    im.partition(im.shape[axis] - 2, axis=axis)
    sl = [slice(None, None, None) for _ in im.shape]
    sl[axis] = im.shape[axis] - 2
    return im[sl]


def _do_detection(ch, mean_var, min_threshold=0):
    if ch.ndim != 2:
        raise ValueError("This should be a 2D array")
    ch = mh.median_filter(ch, np.ones((9,9)))
    mean, var = mean_var
    t = mean + 1.5*np.sqrt(var)
    if min_threshold:
        t = max(t, min_threshold)
    return ch > t

def _do_morph(zt):
    zt = mh.open(zt)
    zt = mh.close(zt, mh.disk(2))
    zt = mh.close_holes(zt)
    return zt

def detect_objects(im, mean_vars):
    '''Detect objects

    Returns binary array'''
    zt = np.array([_do_detection(almost_max(im.get(ch)), mean_vars[ch], min_threshold=MIN_THRESHOLD)
                for ch in FL_CHANNELS])
    zt = zt.any(0)
    return _do_morph(zt)

def segment3d_labeled(im, labeled, mean_var):
    zs = [im.get(ch)
                for ch in FL_CHANNELS]
    zs = np.array([mh.median_filter(ch)
                for ch in zs])

    result = np.zeros(zs.shape, np.intc)
    n_objects = labeled.max()
    if n_objects == 0:
        return result
    for ch in range(result.shape[0]):
        objects = False
        for oi in range(n_objects):
            region = (labeled == (oi+1))
            ch_region = zs[ch] * region
            t = mh.otsu(ch_region.astype(np.uint16), ignore_zeros=True)
            objects = objects|(ch_region > t)
        objects = filter_labeled(objects, min_size=MIN_NR_VOXELS_SUBOBJECT)
        objects = (objects > 0)
        for stack in range(result.shape[1]):
            result[ch, stack] += labeled*objects[stack]
    return result

