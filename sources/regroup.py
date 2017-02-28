from utils import mkdir_p
from os import path
import datetime
import shutil

OUTPUT_DIRECTORY = "/g/taramicro/outputs/OUTPUT/"
START_DATE = datetime.datetime(2014,01,01).toordinal()
def processing_code():
    from time import strftime
    now = datetime.datetime.now()
    pcode = (now.toordinal()-START_DATE)*100 + now.hour
    sdate = strftime('%Y-%m')
    return (sdate, pcode)


def get_git_head():
    from subprocess import Popen, PIPE
    p = Popen(["git", 'rev-parse', '--verify', 'HEAD'], stdout=PIPE)
    return p.stdout.read().rstrip('\n')

def write_metadata(ofname, basedir, label, chamber):
    from time import strftime
    date = strftime("%Y-%m-%d")
    with open(ofname, 'w') as output:
        output.write("""\
original\t{}
label\t{}
chamber\t{}
processing-date\t{}
git-head\t{}
""".format(basedir, label, chamber, date, get_git_head()))
    return ofname


def find_metadata_file():
    return path.join(path.abspath(path.dirname(__file__)), 'screen-data/Features_metadata.xlsx')

def regroup_data(sdate_pcode, label, visualizations, ms, labeleds, logfile):
    if label is None:
        return
    from itertools import chain
    sdate,pcode = sdate_pcode
    outdir = '{}/{}/{}--P{:05}'.format(OUTPUT_DIRECTORY, sdate, label, pcode)
    if path.exists(outdir):
        raise ValueError('Output directory {} already exists'.format(outdir))

    tmp_outdir = outdir + '.tmp'
    mkdir_p(tmp_outdir + '/PICTURES/COMPOSITES/')
    mkdir_p(tmp_outdir + '/OBJECT_LOCATIONS/')
    mkdir_p(tmp_outdir + '/PICTURES/HYPERSTACKS/')
    mkdir_p(tmp_outdir + '/PICTURES/ANIMATIONS/')
    mkdir_p(tmp_outdir + '/PICTURES/MASKS_3D/')

    if ms is not None:
        ms.to_csv('{}/objects.tsv'.format(tmp_outdir), sep='\t')
        shutil.copyfile(find_metadata_file(), '{}/Metadata.xlsx'.format(tmp_outdir))
    shutil.copyfile(logfile, '{}/processing-logfile.tsv'.format(tmp_outdir))


    for ell in labeleds:
        shutil.copyfile(ell, path.join(tmp_outdir, 'OBJECT_LOCATIONS', path.basename(ell)))

    for n,v in enumerate(chain.from_iterable(visualizations)):
        shutil.copyfile(v.max, "{}/PICTURES/COMPOSITES/{}--L{:05}--max.png".format(tmp_outdir, label, n))
        shutil.copyfile(v.mean, "{}/PICTURES/COMPOSITES/{}--L{:05}--mean.png".format(tmp_outdir, label, n))
        if v.mask3d is not None:
            shutil.copyfile(v.mask3d, "{}/PICTURES/MASKS_3D/{}--L{:05}--mask3d.tiff".format(tmp_outdir, label, n))
        if v.hyper is not None:
            shutil.copyfile(v.hyper, "{}/PICTURES/HYPERSTACKS/{}--L{:05}--hyperstack.tiff".format(tmp_outdir, label, n))
        if v.animation is not None:
            shutil.copyfile(v.animation, "{}/PICTURES/ANIMATIONS/{}--L{:05}--animation.gif".format(tmp_outdir, label, n))
        if v.rotation is not None:
            shutil.copyfile(v.rotation, "{}/PICTURES/ANIMATIONS/{}--L{:05}--rotation.gif".format(tmp_outdir, label, n))
    shutil.move(tmp_outdir, outdir)
    return outdir


def paste_measures(sdate_pcode, label, measures, classifications):
    from measures import names
    from itertools import chain
    import numpy as np
    import pandas as pd
    sdate,pcode = sdate_pcode
    lat = '-'
    long = '-'
    sample_time = '-'
    device = '-'

    postfix_names = [
            'HTM_OBJECT_OutputFileName_composite_max',
            'HTM_OBJECT_OutputFileName_composite_mean',
            'HTM_OBJECT_OutputFileName_AnimatedZstack',
            'HTM_OBJECT_OutputFileName_Hyperstack',
            'HTM_OBJECT_OutputFileName_mask3D_Hyperstack',
            'HTM_OBJECT_URI_composite_max',
            'HTM_OBJECT_URI_composite_mean',
            'HTM_OBJECT_URI_AnimatedZstack',
            'HTM_OBJECT_URI_Hyperstack',
            'HTM_OBJECT_URI_mask3D_Hyperstack',
            'HTM_OBJECT_CLASSIFICATION_Pattern',
            'HTM_OBJECT_URI_ClassificationPattern',
            'HTM_OBJECT_CLASSIFICATION_Method',
            'HTM_OBJECT_CLASSIFICATION_TrainingSet',
            'HTM_OBJECT_CLASSIFICATION_Prediction',
            'HTM_OBJECT_CLASSIFICATION_ConfidencePmax',
            'HTM_OBJECT_CLASSIFICATION_ConfidencePdelta',
            ]
    prefix_names = [
            'HTM_OBJECT_LABEL',
            'HTM_OBJECT_LABEL_aggregated',
            'HTM_ACQUISITION_LABEL_aggregated',
            'HTM_PROCESSING_LABEL_aggregated',
            'HTM_SAMPLE_LABEL_aggregated',
            'HTM_STATION_LABEL',
            'HTM_SAMPLE_DEPTH_Intented_Nominal',
            'HTM_SAMPLE_TYPE',
            'HTM_SAMPLE_BARCODE',
            'EVENT_DATETIME_End',
            'EVENT_LATITUDE_End',
            'EVENT_LONGITUDE_End',
            'EVENT_DEPTH_Intended_(m)',
            'EVENT_DEVICE_LABEL',
        ]


    pasted = []
    for nr, (meas,classification) in enumerate(zip(chain(*measures),chain(*classifications))):
        olabel = 'L{:05}'.format(nr)
        processing  = '{}--P{:5}'.format(label, pcode)
        full_label = '{}--{}'.format(label, olabel)
        station, depth, sampletype, barcode, acquisition_code = label.split('--')
        sample = '--'.join([station, depth, sampletype, barcode])

        prefix  = [olabel, full_label, label, processing, sample, station, depth, sampletype, barcode, sample_time, lat, long, depth, device]

        prediction,pmax,confidence = classification
        postfix = [
            '{}--max.png'.format(full_label),
            '{}--mean.png'.format(full_label),
            '{}--animation.gif'.format(full_label),
            '{}--hyperstack.tiff'.format(full_label),
            '{}--mask3d.tiff'.format(full_label),
            'http://composite-max',
            'http://composite-mean',
            'http://animated',
            'http://hyper',
            'http://mask',
            'TAXONOMY_RANK2_2014-08.csv',
            'http://',
            'RandomForest500',
            'ClassificationSetV5.1',
            prediction,
            pmax,
            confidence,
            ]
        pasted.append(np.concatenate( (prefix, meas, postfix) ))
    data = np.array(pasted)
    return pd.DataFrame(data, columns=(prefix_names + names + postfix_names))




