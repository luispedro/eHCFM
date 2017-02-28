import subprocess
import numpy as np
import mahotas as mh
from segment import almost_max
from utils import mkdir_p, fsync_all
import globals

dna_index, auto_index, lysine_index, DiO6_index, light_index  = range(5)

def output_image(im, oname):
    from os import path
    outdir,_ = path.split(oname)
    mkdir_p(outdir)
    if not oname.endswith('.png'):
        oname += '.png'
    if im.dtype == bool:
        im = im.astype(np.uint8)
    elif im.max() < 256:
        im = im.astype(np.uint8)
    elif im.max() < 2**16:
        im = im.astype(np.uint16)
    else:
        raise ValueError('Too high')
    mh.imsave(oname, im)


def create_rotating(oname, s):
    s = mh.croptobbox(s, border=6)
    redc = (.9, .1, .1)
    greenc = (.3, .9, .3)
    cyanc = (.1, .9, .9)
    bluec = (.3, .3, .9)

    def smooth(t):
        # Add two slices of 0-padding on the top & bottom:
        t = np.concatenate([t[:2]*0, t, t[:2]*0])
        return mh.gaussian_filter(t, 1.2, mode='nearest')
    import mayavi
    from mayavi import mlab
    import shutil
    import tempfile
    import imageio
    import os

    mlab.options.offscreen = True
    # engine = mlab.get_engine()
    # engine.current_scene.scene.off_screen_rendering = True

    ch0, ch1, ch2, ch3 = None,None,None,None
    try:
        f = mlab.gcf()
        Z,X,Y = s[0].shape
        Z,X,Y = np.mgrid[:Z+4,:X,:Y]
        if globals.SIZE_FRACTION == 'H5':
            X = X * globals.PIXEL_SIDE_MU
            Y = Y * globals.PIXEL_SIDE_MU

        ch0 = mlab.contour3d(Z, X, Y, smooth(s[dna_index]), color=bluec)
        ch1 = mlab.contour3d(Z, X, Y, smooth(s[auto_index]), color=redc)
        ch2 = mlab.contour3d(Z, X, Y, smooth(s[DiO6_index]), color=greenc)
        ch3 = mlab.contour3d(Z, X, Y, smooth(s[lysine_index]), color=cyanc, opacity=.33)
        ch1.stop()
        ch2.stop()
        ch3.stop()
        #f.scene.camera.position = np.array([-58.59674338, -12.91657568,  73.07397018])
        try:
            tmp_ofile = tempfile.NamedTemporaryFile(suffix='.gif', delete=False)
            tmp_ofile.close()
            tmp_oname = tmp_ofile.name
            with imageio.get_writer(tmp_oname, mode='I') as writer:
                for i in range(210):
                    if i == 30:
                        ch1.start()
                    if i == 60:
                        ch2.start()
                    if i == 90:
                        ch3.start()
                    if i == 120:
                        ch3.stop()
                    if i == 150:
                        ch2.stop()
                    if i == 180:
                        ch1.stop()
                    f.scene.camera.azimuth(15)
                    f.scene.render()
                    im = mlab.screenshot()
                    im = im[:,:,:3]
                    is_bg = np.all(im == 128, 2)
                    im[is_bg] = (255, 255, 255)

                    writer.append_data(im)
            shutil.move(tmp_oname, oname)
        except:
            os.unlink(tmp_oname)
            raise
    finally:
        if ch0 is not None:
            ch0.remove()
        if ch1 is not None:
            ch1.remove()
        if ch2 is not None:
            ch2.remove()
        if ch3 is not None:
            ch3.remove()
        mlab.clf(f)


def purple(c, stretch=False):
    if stretch:
        c = mh.stretch(c)
    return np.dstack([255-c/2,255-c,255-c/4])

def dark_green(c, stretch=False):
    if stretch:
        c = mh.stretch(c)
    return np.dstack([255-c,255-c/4,255-c])


def in_pos_w(im, channels):
    canvas = 255+np.zeros((im.shape+(3,)), np.uint8)
    for c in range(3):
        if c not in channels:
            canvas[:,:,c] = 255-im
    return canvas

def in_pos(im, channels):
    canvas = np.zeros((im.shape+(3,)), np.uint8)
    for c in channels:
        canvas[:,:,c] = im
    return canvas


def build_overlay(to_show):
    h,w = to_show[0].shape
    composite = np.zeros((h,w,3), np.uint8)
    composite = np.maximum(composite, in_pos(to_show[auto_index], [0]))
    composite = np.maximum(composite, in_pos(to_show[DiO6_index], [1]))
    composite = np.maximum(composite, in_pos(to_show[lysine_index],     [1,2]))
    composite = np.maximum(composite, in_pos(to_show[dna_index], [2]))
    return composite

def build_composite(zs, region, find_best_light=False):
    '''Build composite from stack

    Parameters
    ----------
    zs : stack
    region : binary image or None
    '''
    import edf

    if region is not None:
        s = mh.bbox(region, border=6, as_slice=True)
        to_show = [z[s] for z in zs]
    else:
        to_show = zs[:]
    if find_best_light:
        to_show[-1] = edf.sobel_cv_best([st[s] for st in zs[-1]])

    h,w = to_show[0].shape
    border = 8
    composite = build_overlay(to_show)

    if region is not None:
        outline = mh.borders(region[s])
        outline = np.array([255,255,255])[None,None,:]*outline[:,:,None]
        outline = outline.astype(np.uint8)
        composite = np.maximum(composite, outline)

    sbar_height = 6
    sbar_border = 4
    sbar_width = 26 # Roughly 5 mu m (1pixel = 0.19 mu m)

    canvas = np.zeros((2*h + border + sbar_border*2 + sbar_height, w*3 + 2*border, 3), np.uint8)
    canvas.fill(192)


    top = slice(None,h, None)
    bottom = slice(h+border,2*h+border, None)

    left = slice(None, w, None)
    middle = slice(w+border, 2*w+border,None)
    rigth = slice(2*w+2*border, 3*w+2*border,None)

    canvas[   top,  left] = purple(to_show[lysine_index])
    canvas[bottom,  left] = dark_green(to_show[DiO6_index])

    canvas[   top, middle] = in_pos_w(to_show[dna_index], [2])
    canvas[bottom, middle] = in_pos_w(to_show[auto_index], [0])

    canvas[   top,   rigth] = np.dstack(( to_show[light_index],
                                            to_show[light_index],
                                            to_show[light_index]))
    canvas[bottom,   rigth] = composite
    canvas[-sbar_border-sbar_height:-sbar_border, -sbar_border-sbar_width:-sbar_border] = 0

    return canvas

def create_animation(ofile, s):
    '''Create animation onto ``ofile``'''
    import os
    import shutil
    import imageio
    import tempfile
    tmp_ofile = tempfile.NamedTemporaryFile(suffix='.gif', delete=False)
    tmp_ofile.close()
    tmp_ofile = tmp_ofile.name

    try:
        with imageio.get_writer(tmp_ofile, mode='I') as writer:
            n_slices = s.shape[1]
            for i in range(n_slices):
                c = build_composite(s[:-1][:,i,:,:], s[-1][0] > 0)
                writer.append_data(c)
    except:
        os.unlink(tmp_ofile)
        raise
    shutil.move(tmp_ofile, ofile)


def adaptative_mean(im):
    mean = im.sum(0)/(1e-4 + (im > 0).sum(0))
    return np.round(mean)

_imagej_metadata = """ImageJ=1.47a
images={images}
channels={channels}
slices={slices}
hyperstack=true
mode=color
loop=false"""
def output_hyperstack(zs, oname):
    '''
    Write out a hyperstack to ``oname``

    Parameters
    ----------
    zs : ndarray
    oname : str
    '''
    from imread import imsave_multi
    zs = np.asanyarray(zs)
    if zs.ndim != 4:
        raise ValueError('output_hyperstack only works with 4D images [received image with shape {}]'.format(zs.shape))

    slices = zs.shape[1]
    channels = zs.shape[0]
    metadata = _imagej_metadata.format(images=(channels*slices), channels=channels, slices=slices)
    ims = []
    for s1 in range(slices):
        for s0 in range(channels):
            ims.append(zs[s0, s1])
    imsave_multi(oname, ims, opts={'metadata': metadata})

def output_masks(bins, outdir):
    mkdir_p(outdir)
    outputs = []
    for i, bin in enumerate(bins):
        bin = bin.astype(np.uint8)*255
        oname = outdir + '/mask-{:03}.tiff'.format(i)
        output_hyperstack(bin, oname)
        outputs.append(oname)
    return outputs

class Visualizations(object):
    __slots__ = ['mean', 'hyper', 'max', 'rotation', 'animation', 'mask3d']
    def __init__(self, max=None, mean=None, hyper=None, rotation=None, animation=None, mask3d=None):
        self.max = max
        self.mean = mean
        self.hyper = hyper
        self.animation = animation
        self.rotation = rotation
        self.mask3d = mask3d

    def __str__(self):
        return "Visualizations(max={0.max}, mean={0.mean}, hyper={0.hyper}, rotation={0.rotation}, animation={0.animation}, mask3d={0.mask3d})".format(self)
    __repr__ = __str__

def output_visualizations(im, labeled, labeled_3d, outpat, save_hyperstack=True, save_animation=True, save_rotation=True, save_true_max=False):
    import mahotas as mh
    nr_objects = labeled.max()
    if nr_objects == 0:
        return []
    zs = [im.get(ch)
                for ch in ['dna', 'auto', 'lysine', 'DiO6', 'light']]
    zs_max  = [mh.median_filter(z.max(0)) for z in zs[:-1]] + [zs[-1]]
    zs_mean = [adaptative_mean(z) for z in zs[:-1]] + [zs[-1]]
    zs_almost_max  = [mh.median_filter(almost_max(z)) for z in zs[:-1]] + [zs[-1]]

    outdir_composite = outpat.format('composites')
    outdir_animation = outpat.format('animations')
    mkdir_p(outdir_composite)
    mkdir_p(outdir_animation)
    results = []
    for oid in range(nr_objects):
        region = (labeled == oid+1)
        if np.any(region):
            if save_true_max:
                max_fname = '{}/max_{:03}.png'.format(outdir_composite, oid)
                c = build_composite(zs_max, region, find_best_light=True)
                mh.imsave(max_fname, c)

            max_fname = '{}/almost_max_{:03}.png'.format(outdir_composite, oid)
            c = build_composite(zs_almost_max, region, find_best_light=True)
            mh.imsave(max_fname, c)

            mean_fname = '{}/mean_{:03}.png'.format(outdir_composite, oid)
            c = build_composite(zs_mean, region, find_best_light=True)
            mh.imsave(mean_fname, c)

            s = mh.bbox(region, as_slice=True)
            zoid = np.array([z[np.index_exp[:]+s] for z in zs] + [[region[s].astype(np.uint8) for _ in range(zs[0].shape[0])]])

            if labeled_3d is not None:
                object3d = (labeled_3d == (oid+1))
                mask_3dname = '{}/mask3d_{:03}.tiff'.format(outdir_animation, oid)
                cur3d = object3d[np.index_exp[:,:]  + s]
                output_hyperstack(cur3d.astype(np.uint8)*255, mask_3dname)
            else:
                mask_3dname = None

            cur_res = Visualizations(max=max_fname, mean=mean_fname, mask3d=mask_3dname)

            if save_hyperstack:
                hyper_fname = '{}/hyper_{:03}.tif'.format(outdir_composite, oid)
                output_hyperstack(zoid, hyper_fname)
                cur_res.hyper=hyper_fname

            if save_animation:
                bbox = mh.bbox(region, border=6, as_slice=True)
                s = np.array([z[np.index_exp[:] + bbox] for z in zs])

                mask = np.array([region[bbox] for _ in range(zs[0].shape[0])])
                anim_oname = '{}/animation_{:03}.gif'.format(outdir_animation, oid)
                create_animation(anim_oname, np.append(s, [mask], axis=0))
                cur_res.animation = anim_oname

            if save_rotation:
                if labeled_3d is None:
                    raise ValueError("When `save_rotation` is used, labeled_3d must not be None!")
                rot_oname = '{}/rotation_{:03}.gif'.format(outdir_animation, oid)
                create_rotating(rot_oname, object3d)
                cur_res.rotation = rot_oname


            results.append(cur_res)
    return results

