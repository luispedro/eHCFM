import macol.image
import re

def extract_z_position(fname):
    '''Extract z stack number from filename'''
    m = re.match(r'^image--L\d\d--S\d\d--U0[012]--V0[012]--J\d\d--E00--O\d\d--X\d\d--Y\d\d--T00--Z(\d\d)--C\d\d\.ome\.tif(?:\.gz)?', fname)
    if m is None:
        return m
    z, = m.groups()
    return int(z, 10)

class MultiFileImageGZ(macol.image.MultiFileImage):
    def open_file(self, fname):
        import imread
        if fname.endswith('.gz'):
            import gzip
            data = gzip.open(fname).read()
            return imread.imread_from_blob(data, 'tif')
        return imread.imread(fname)

def load_directory(basedir):
    import os
    suffix = ''
    max_z = -1
    for fname in os.listdir(basedir):
        if fname.endswith('.ome.tif') or fname.endswith('.ome.tif.gz'):
            basename = os.path.join(basedir, fname)
            basename = basename[:basename.find('--Z')]
            if fname.endswith('.gz'):
                suffix = '.gz'
            break
    else:
        raise IOError('load_directory: {} does not seem like an image directory'.format(basedir))
    for fname in os.listdir(basedir):
        this_z = extract_z_position(fname)
        if this_z is not None:
            max_z = max(max_z, this_z)
    channels = ['DiO6', 'auto', 'light', 'dna', 'lysine']
    files = {ch:['{}--Z{:02}--C{:02}.ome.tif{}'.format(basename, zi,ci, suffix) for zi in xrange(max_z + 1)]
                    for ci,ch in enumerate(channels)}
    im = MultiFileImageGZ(files)
    im.basedir = basedir
    return im


def get_datadirs(basedir):
    '''
    directories = get_datadirs(basedir)

    Returns a list of directories that appear to contain microscope data

    Returns
    -------
    directories : list of str
    '''
    from os import walk
    datadirs = []
    for root, dirs, files in walk(basedir, followlinks=True):
        if root.endswith('AF'): continue
        for f in files:
            if f.endswith('ome.tif') or f.endswith('.ome.tif.gz'):
                datadirs.append(root)
                break
    datadirs.sort()
    return datadirs

def is_control(im):
    '''
    c = is_control(im)

    Returns whether this is a control image
    '''
    u,v = get_chamber(im)
    return v in (0,2)

def get_samples(basedir):
    if basedir.startswith('../data/'):
        basedir = basedir[len('../data/'):]
    basedir = basedir.split('/')[0]
    tokens = basedir.split('_')
    sampleA = tokens[3]
    sampleB = tokens[4].split('-')[0]
    return sampleA, sampleB

def get_chamber(im):
    '''
    u,v = get_chamber(im)
    '''
    import re
    if type(im) == str:
        path = im
    else:
        path = im.files['dna'][0]
    m = re.search(r'chamber--U([0-9]{2})--V([0-9]{2})', path)
    if not m:
        raise ValueError('path does not match expected pattern: "{}"'.format(path))
    u,v = m.groups()
    return int(u),int(v)

def field_coordinates(im):
    '''x,y = field_coordinates(im)
    '''
    import re
    if type(im) == str:
        path = im
    else:
        try:
            path = im.files['dna'][0]
        except:
            path = im.iA.files['dna'][0]
    m = re.search('field--X([0-9]{2})--Y([0-9]{2})', path)
    if not m:
        raise ValueError('path does not match expected pattern: "{}"'.format(path))
    x,y = m.groups()
    return int(x) , int(y)
