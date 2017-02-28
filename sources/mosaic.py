DOWNSAMPLE_RATE = 4

def build_mosaic(images):
    '''Builds a mosaic image from all the images given as argument
    '''
    import numpy as np
    from segment import almost_max
    from composites import build_overlay
    import mahotas as mh
    from load import load_directory
    thumbs = {}
    for dirname in images:
        im = load_directory(dirname)
        zs = [im.get(ch)
                    for ch in ['dna', 'auto', 'lysine', 'DiO6']]
        zs_almost_max  = [mh.median_filter(almost_max(z)) for z in zs]
        overlay = build_overlay(zs_almost_max)
        overlay = overlay.astype(float)
        thumb = np.dstack([
                    mh.convolve(overlay[:,:,i].T, np.ones((DOWNSAMPLE_RATE,DOWNSAMPLE_RATE)))[::DOWNSAMPLE_RATE,::DOWNSAMPLE_RATE]
                        for i in range(3) ])

        x,y = dirname[-len('X00--Y00'):].split('--')
        x = int(x[1:], 10)
        y = int(y[1:], 10)
        thumbs[x,y] = thumb
        im.unload() # release memory

    xs = list(sorted(set([x for x,_ in thumbs.keys()])))
    ys = list(sorted(set([y for _,y in thumbs.keys()])))

    h,w,_ = thumb.shape
    canvas = np.zeros((len(ys)*h, len(xs)*w, 3))
    for (x,y),v in thumbs.items():
        x = len(xs) - xs.index(x) - 1
        y = len(ys) - ys.index(y) - 1

        canvas[y*h:(y+1)*h, x*w:(x+1)*w] = v
    return mh.stretch_rgb(canvas)
