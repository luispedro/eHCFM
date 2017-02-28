import mahotas as mh
import load
def remove_overlapped(objects, im, overlap):
    '''Removes objects which are in the overlap region'''
    x,y = load.field_coordinates(im)
    is_extreme0 = (x == 0)
    is_extreme1 = (y == 0)
    if is_extreme0 and is_extreme1:
        return objects
    labeled,nr = mh.label(objects)
    remove = []
    for oi in xrange(nr):
        box = mh.bbox(labeled == (oi+1))
        min0 = box[0]
        min1 = box[2]
        if min0 > 2048-overlap and not is_extreme0:
            remove.append(oi+1)
        if min1 > 2048-overlap and not is_extreme1:
            remove.append(oi+1)
    if len(remove):
        objects = mh.labeled.remove_regions(labeled, remove)
        objects = (objects > 0)
    return objects


def remove_overlapped20(objects):
    labeled, n = mh.label(objects)
    min0,_,min1,_ = mh.labeled.bbox(labeled).T
    valid = (min0 < (3072-3072//20)) & (min1 < (3072-3072//20))
    labeled = mh.labeled.remove_regions_where(labeled, ~valid)
    labeled = mh.labeled.remove_bordering(labeled)
    return (labeled > 0)
