from os import path
import re

def sample_info(basedir, fraction='auto', respect_K=True):
    if fraction == 'auto':
        fraction = detect_fraction(basedir)
    basedir = path.basename(basedir.strip())
    fname,index_Run_Label = {
            '5': ('screen-data/sample-registry.tsv',21),
            '20': ('screen-data/sample-registry20_180.tsv',20),
            '02': ('screen-data/sample-registry.02.tsv',20),
            }[fraction]
    m = re.match(r'^TARA_HCS1_H(?:5|20|02)_([AG][0-9]+)_([AG][0-9]+)--\d{4}_\d{2}_\d{2}_\d\d_\d\d_\d\d$', basedir)
    if m is None:
        raise ValueError("Could not match {} to TARA regex".format(basedir))
    first, second = m.groups()
    firstr = None
    secondr = None
    for i,line in enumerate(open(fname)):
        tokens = line.rstrip().split('\t')
        if i == 0:
            if tokens[13] != 'HTM_ACQUISITION_LABEL_aggregated':
                print("File format has changed (file: {}; index 13)! check.".format(fname))
            if tokens[index_Run_Label] != 'HTM_ACQUISITION_Run_Label':
                print("File format has changed (file: {}; index {})! check.".format(fname, index_Run_Label))
        if respect_K and line[0] != 'K':
            continue
        label = tokens[13]
        acquisition = tokens[index_Run_Label]
        if basedir[:-8] == acquisition[:-8]:
            if first in label:
                firstr = label
            elif second in label:
                secondr = label
            else:
                raise ValueError("Something is wrong [basedir is {}, acquisition is {}, label is {} ({}/{})]".format(basedir, acquisition, label, first, second))
    return firstr, secondr


def detect_fraction(s):
    if s.startswith('TARA_HCS1_H5_'): return'5'
    elif s.startswith('TARA_HCS1_H20_'): return '20'
    elif s.startswith('TARA_HCS1_H02_'): return '02'
    raise ValueError("detect_fraction: cannot parse {}".format(s))

if __name__ == '__main__':
    n_Nones = 0
    for f in [
        'screen-data/tara-data-directories.txt',
        'screen-data/tara-data-directories.20_200.txt',
        'screen-data/tara-data-directories.02.txt']:
        for line in open(f):
            f1, f2 = sample_info(line)

            n_Nones += (f1 is None)
            n_Nones += (f2 is None)
            if f1 is None:
                print 1, line.strip()
            if f2 is None:
                print 2, line.strip()
