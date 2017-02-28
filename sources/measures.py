import mahotas as mh
import numpy as np
import features
from features import central_zernike
from segment import almost_max
from globals import PIXEL_SIDE_MU, PIXEL_SIZE_MU_SQUARED, VOXEL_SIZE_MU_CUBED


def select_region(im, s, region=None):
    r = im[np.index_exp[:] + s]
    if region is not None:
        r = r.copy()
        r *= region[s]
    return r

def _overlap(a, b):
    corr = np.corrcoef(a[0].ravel(), b[0].ravel())[0,1]
    a1 = a[1].ravel()
    b1 = b[1].ravel()
    c = (a1 & b1)
    if np.any(c):
        a1 = a1[c]
        b1 = b1[c]
        bcorr = (np.corrcoef(a1, b1)[0,1]
                if a1.var() > 0 and b1.var() > 0
                else 0.0)
    else:
        bcorr = -1.0
    return [corr, bcorr]

def _haralick(im):
    fs = mh.features.haralick(im.astype(np.uint8), ignore_zeros=True).mean(0)
    if not np.any(np.isnan(fs)): return fs
    return np.zeros(13)

def greymeasures(grey, binary):
    objects,nr_objects = mh.label(binary)
    if nr_objects == 0:
        return np.zeros(12)
    surface = mh.borders(objects)
    Z,X,Y = np.where(surface)
    cz,cx,cy = mh.center_of_mass(grey)
    Sx = PIXEL_SIDE_MU
    Sy = PIXEL_SIDE_MU
    Sz = 1.0
    surfdist = np.sqrt((X-cx)**2 * Sx**2 + (Y - cy)**2 * Sy**2 + (Z - cz)**2 * Sz ** 2)

    pixels = grey[binary].ravel()
    return [
        nr_objects,
        pixels.sum(),
        pixels.max(),
        pixels.min(),
        pixels.mean(),
        pixels.std(),
        np.median(pixels),
        np.sum(pixels**2.),
        np.mean(pixels**2.),

        np.sum(surface > 0),
        np.mean(surfdist),
        np.median(surfdist),
        ]

_greymeasures_names = [
        'nr_objects',
        'sum',
        'max',
        'min',
        'mean',
        'std',
        'median',
        'sum2',
        'mean2',
        'surfvoxels',
        'surfdist.mean',
        'surfdist.median',
        ]

def brightness(im, labeled):
    measures = []
    dna = im.get('dna')
    auto = im.get('auto')
    lysine = im.get('lysine')
    DiO6 = im.get('DiO6')
    n = labeled.max()
    for i in range(n):
        region = (labeled == (i+1))
        if not np.any(region):
            continue
        s = mh.bbox(region, as_slice=1)
        dnar = select_region(dna, s , region)
        autor = select_region(auto, s, region)
        lysiner = select_region(lysine, s, region)
        DiO6r = select_region(DiO6, s, region)
        measures.append(
            [region.sum()
            ,dnar.sum()
            ,autor.sum()
            ,lysiner.sum()
            ,DiO6.sum()


            ,dnar.sum()/float(region.sum())
            ,autor.sum()/float(region.sum())
            ,lysiner.sum()/float(region.sum())
            ,DiO6r.sum()/float(region.sum())

            ])
    return np.array(measures)


def measures1(region, region3d, dnar, autor, lysiner, DiO6r, lightr, dnart, autort, lysinert, DiO6rt):
    from edf import sobel_best
    lightr_max = lightr.max(0)
    lightr_best = sobel_best(lightr)
    assert lightr_max.shape == lightr_best.shape

    dnar_mean = dnar.mean(0)
    autor_mean = autor.mean(0)
    lysiner_mean = lysiner.mean(0)
    DiO6r_mean = DiO6r.mean(0)

    region_maxes = [almost_max(c) for c in (dnar, autor, lysiner, DiO6r)]

    n_pixels = region.sum()
    slice = mh.bbox(region, border=3, as_slice=True)
    region3d = region3d.any(0)
    n_voxels = sum([mh.close_holes(region3d[d0][slice]).sum() for d0 in range(region3d.shape[0])])
    perimeter = mh.bwperim(region).sum()
    roundness = mh.features.roundness(region)
    major_axis, minor_axis = mh.features.ellipse_axes(region)
    channels = [
                (dnar,dnart),
                (autor,autort),
                (lysiner,lysinert),
                (DiO6r,DiO6rt),
                ]
    centers = []
    for c,_ in channels:
        centers.append(mh.center_of_mass(c)/c.shape)
    centers = np.concatenate(centers)
    overlaps = []
    for i,ci in enumerate(channels):
        for cj in channels[i+1:]:
            overlaps.extend(_overlap(ci, cj))
    moments = np.concatenate([
        features.moments(m) for m in [dnar_mean, autor_mean, lysiner_mean, DiO6r_mean, lightr_best]])
    greys = np.concatenate([
        greymeasures(r, b) for r,b in channels])
    haralicks = np.concatenate([
        _haralick(m) for m in region_maxes])

    lbps = np.concatenate([
        mh.features.lbp(m, 8, 8)
            for m in region_maxes])

    return np.concatenate([
                [n_pixels * PIXEL_SIZE_MU_SQUARED
                ,n_voxels * VOXEL_SIZE_MU_CUBED
                ,perimeter * PIXEL_SIDE_MU
                ,roundness
                ,major_axis * PIXEL_SIDE_MU
                ,minor_axis * PIXEL_SIDE_MU
                ,dnart.sum() * PIXEL_SIZE_MU_SQUARED
                ,autort.sum() * PIXEL_SIZE_MU_SQUARED
                ,lysinert.sum() * PIXEL_SIZE_MU_SQUARED
                ,DiO6rt.sum() * PIXEL_SIZE_MU_SQUARED
                ],
                greys,
                centers,

                central_zernike(dnar_mean),
                central_zernike(autor_mean),
                central_zernike(lysiner_mean),
                central_zernike(DiO6r_mean),
                central_zernike(lightr_best),

                haralicks,
                _haralick(lightr_max),
                _haralick(lightr_best),

                lbps,
                mh.features.lbp(lightr_best, 8, 8),

                overlaps,
                moments,
                ])
names = [
        'field_pos0'
        ,'field_pos1'
        ,'pos_in_field0'
        ,'pos_in_field1'
        ,'size2d'
        ,'size'
        ,'perimeter'
        ,'roundness'
        ,'major_axis'
        ,'minor_axis'
        ,'dna.size'
        ,'auto.size'
        ,'lysine.size'
        ,'DiO6.size'
    ]
for c in ('dna','auto', 'lysine', 'DiO6'):
    names.extend(['{}.{}'.format(c, n) for n in _greymeasures_names])

for c in ('dna','auto', 'lysine', 'DiO6'):
    names.extend(['center_mass{}.{}'.format(n, c) for n in range(3)])


for c in ('dna','auto', 'lysine', 'DiO6', 'lightbest'):
    names.extend(['{}.central_zernike{}'.format(c,n) for n in xrange(25)])

for c in ('dna','auto', 'lysine', 'DiO6', 'lightmax', 'lightbest'):
    names.extend(['{}.haralick{}'.format(c, n) for n in xrange(13)])

for c in ('dna','auto', 'lysine', 'DiO6', 'lightbest'):
    names.extend(['{}.lbp_8_8_{}'.format(c, n) for n in xrange(36)])

names.extend(
        ['dna.auto.corr'
        ,'dna.auto.thresh.overlap'

        ,'dna.lysine.corr'
        ,'dna.lysine.thresh.overlap'

        ,'dna.DiO6.corr'
        ,'dna.DiO6.thresh.overlap'

        ,'auto.lysine.corr'
        ,'auto.lysine.thresh.overlap'

        ,'auto.DiO6.corr'
        ,'auto.DiO6.thresh.overlap'

        ,'lysine.DiO6.corr'
        ,'lysine.DiO6.thresh.overlap'
        ])
for c in ('dna','auto', 'lysine', 'DiO6', 'lightbest'):
    names.extend(['{}.moment{}'.format(c, m) for m in ('11', '12', '22')])


def measures(im, labeled, labeled3d):
    from load import field_coordinates
    f0,f1 = field_coordinates(im)
    measures = []
    dna = im.get('dna')
    auto = im.get('auto')
    lysine = im.get('lysine')
    DiO6 = im.get('DiO6')
    light = im.get('light')

    n = labeled3d.max()
    for i in range(n):
        regions3d = (labeled3d == (i+1))
        region = (labeled == (i+1))
        if not np.any(region):
            raise ValueError('Had an empty region')
        s = mh.bbox(region, border=6, as_slice=1)
        x0,x1 = mh.center_of_mass(region)

        dnar = select_region(dna, s, region)
        autor = select_region(auto, s, region)
        lysiner = select_region(lysine, s, region)
        DiO6r = select_region(DiO6, s, region)
        lightr = select_region(light, s, region)

        dnart = select_region(regions3d[0], s)
        autort = select_region(regions3d[1], s)
        lysinert = select_region(regions3d[2], s)
        DiO6rt = select_region(regions3d[3], s)

        measures.append(np.concatenate([
                [f0, f1, x0, x1],
                measures1(region, regions3d, dnar, autor, lysiner, DiO6r, lightr, dnart, autort, lysinert, DiO6rt)]))

    return np.array(measures)


