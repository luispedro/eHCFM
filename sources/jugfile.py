from collections import defaultdict
from sys import argv
from os import path
from jug import Task, TaskGenerator, bvalue, set_jugdir
from jug.hooks.exit_checks import exit_if_file_exists
from jug.backends.file_store import file_store

import metadata
import load
from segment import detect_objects, segment3d_labeled
from composites import output_visualizations
from labeled_filter import filter_labeled
from load import load_directory
from qc import measure_saturation
from overlaps import remove_overlapped
from beads import save_beads, save_bead_brightness
from measures import measures, brightness
from regroup import regroup_data, processing_code, paste_measures, write_metadata
from illumination import mean_var, compute_median_top, fit_smooth
from utils import mkdir_p
from mahotas.labeled import labeled_size
from globals import PIXEL_SIDE_MU, PIXEL_SIZE_MU_SQUARED
from mosaic import build_mosaic

MIN_DIAMETER = 5
MIN_SQ_MUS = (MIN_DIAMETER/2)**2 * 3.1459
MIN_NR_PIXELS = int(MIN_SQ_MUS/PIXEL_SIZE_MU_SQUARED)

exit_if_file_exists('jug.stop.sign')
p = Task(processing_code)

get_datadirs = TaskGenerator(load.get_datadirs)
detect_objects = TaskGenerator(detect_objects)
segment3d_labeled = TaskGenerator(segment3d_labeled)
output_visualizations = TaskGenerator(output_visualizations)
load_directory = TaskGenerator(load_directory)
filter_labeled = TaskGenerator(filter_labeled)
measure_saturation = TaskGenerator(measure_saturation)
measures = TaskGenerator(measures)
brightness = TaskGenerator(brightness)
remove_overlapped = TaskGenerator(remove_overlapped)
write_metadata = TaskGenerator(write_metadata)
regroup_data = TaskGenerator(regroup_data)
paste_measures = TaskGenerator(paste_measures)
mean_var = TaskGenerator(mean_var)
compute_median_top = TaskGenerator(compute_median_top)
fit_smooth = TaskGenerator(fit_smooth)
labeled_size = TaskGenerator(labeled_size)
build_mosaic = TaskGenerator(build_mosaic)


@Task
def make_outputdirs():
    mkdir_p('outputs/')
    for d in ['animations/',
                'bead_stats/',
                'predictions.all/',
                'composites/',
                'labels/',
                'metadata/',
                'mosaics/',
                'saturated/']:
        mkdir_p('outputs/' + d)

@TaskGenerator
def save_saturation(saturated):
    import numpy as np
    saturated = np.concatenate(saturated, axis=1)
    np.save('outputs/saturated/saturated_{}'.format(basedir), saturated)


@TaskGenerator
def output_accepted_summary(outdir, name, accepted):
    mkdir_p(outdir)
    output = open(outdir + '/' + name + '.accepted.txt', 'w')
    pos = 0
    total = 0.
    for _,acc in accepted:
        pos += sum(acc)
        total += len(acc)
    if total:
        output.write('Accepted: {:%} ({}/{})\n'.format(pos/total, pos, total))
    else:
        output.write('Accepted: 0 (0/0)\n')
    output.close()

@TaskGenerator
def output_image(objects, oname):
    import composites
    return composites.output_image(objects, oname)

@TaskGenerator
def output_summary(basedir, predictions):
    def aggregate(predictions):
        from collections import Counter
        from itertools import chain
        c = Counter()
        if predictions is None:
            return c
        for p in chain.from_iterable(predictions):
            c[p] += 1
        return c

    with open('outputs/{}.summary.txt'.format(basedir), 'w') as output:
        output.write("Chamber V00-U01\n")
        output.write("---------------\n")
        for k,v in  aggregate(predictions.get((0,1))).iteritems():
            output.write("     {:3} {:20}\n".format(v,k))

        output.write("Chamber V01-U01\n")
        output.write("---------------\n")
        for k,v in  aggregate(predictions.get((1,1))).iteritems():
            output.write("     {:3} {:20}\n".format(v,k))


@TaskGenerator
def predict_class(features, return_probs=False):
    import cPickle as pickle
    import numpy as np
    rf4 = pickle.load(open('classifiers/labeled.classifier.pkl', 'rb'))
    rf4.n_jobs = 1
    res = []
    for fs in features:
        cur = []
        for f in fs:
            preds = rf4.predict_proba(f)
            assert len(preds) == 1
            preds = preds[0]
            if return_probs:
                cur.append(preds)
            else:
                preds.sort()
                # np.partition(preds, kth=-2)[:,-2:].ptp(1)
                pmax = preds[-1]
                conf = preds[-1]-preds[-2]
                kl, = rf4.predict(f)
                cur.append( (kl, pmax, conf) )
        res.append(cur)
    return np.array(res)

@TaskGenerator
def paste_save_all_predictions(label, preds):
    from itertools import chain
    import pandas as pd
    import cPickle as pickle
    rf4 = pickle.load(open('classifiers/labeled.classifier.pkl', 'rb'))

    all_preds = {}
    for i,p in enumerate(chain.from_iterable(preds)):
        all_preds["{}--L{:05}".format(label, i)] = p
    all_preds = pd.DataFrame.from_dict(all_preds, orient='index')
    all_preds.sort_index(inplace=True)
    all_preds.columns = rf4.classes_
    all_preds.to_csv('outputs/predictions.all/{}.tsv'.format(label), sep='\t')
    return all_preds

@TaskGenerator
def renumber_labeled(labeleds):
    next_id = 0
    labeleds.sort(key=lambda ix:ix[0])
    filenames = []
    for name,labeled in labeleds:
        labeled += (labeled > 0) * next_id
        next_id = labeled.max() + 1

        name = name.replace('/g/taramicro/', 'outputs/labeled.renumbered/')
        name += '.png'

        output_image.f(labeled, name)
        filenames.append(name)

    return filenames


@TaskGenerator
def write_labels(basedir, label1, label2):
    with open('outputs/labels/{}_labels.txt'.format(basedir), 'w') as olabels:
       olabels.write("{}\n{}\n".format(label1, label2))

if argv[1].endswith('/'): argv[1] = argv[1][:-1]
basedir = path.basename(argv[1])

set_jugdir(file_store('jugdirs/jugdir_{}'.format(basedir), compress_numpy=True))

images = get_datadirs('/g/taramicro/'+basedir)
images = bvalue(images)
images.sort()

illumedian = {}
wells = frozenset(map(load.get_chamber, images))
mean_vars = {well:{} for well in wells}
for ch in ['dna', 'auto', 'lysine', 'DiO6', 'light']:
    illumedian[ch] = fit_smooth(compute_median_top(images, ch))
    for well in wells:
        mean_vars[well][ch] = mean_var([im for im in images if load.get_chamber(im) == well], ch, project=False)

sizes = []
bead_sizes = {}
saturated = []
accepted = []
bead_brightness = defaultdict(list)
visualizations = defaultdict(list)
features = defaultdict(list)
labeleds = defaultdict(list)

for d in images:
    outpat = 'outputs/{}/' + d[len('/g/taramicro/'):]
    im = load_directory(d)
    is_control = load.is_control(d)
    chamber = load.get_chamber(d)

    objects = detect_objects(im, mean_vars[chamber])
    objects = remove_overlapped(objects, im, 2048//10)

    min_size = (MIN_NR_PIXELS if not is_control else None)
    labeled = filter_labeled(objects, remove_bordering=True, min_size=min_size)
    if is_control:
        csize = labeled_size(labeled)
        bead_sizes[d] = csize
        bead_brightness[chamber].append(brightness(im, labeled))
    else:
        labeleds[chamber].append((d, labeled))
        labeled_3d = segment3d_labeled(im, labeled, mean_vars[chamber])
        visualizations[chamber].append(
            output_visualizations(im, labeled, labeled_3d, outpat, save_rotation=False))
        fs = measures(im, labeled, labeled_3d)
        features[chamber].append(fs)
        saturated.append(measure_saturation(im))

mosaic1 = build_mosaic([d for d in images if load.get_chamber(d) == (0,1)])
mosaic2 = build_mosaic([d for d in images if load.get_chamber(d) == (1,1)])
output_image(mosaic1, 'outputs/mosaics/{}-mosaic1'.format(basedir))
output_image(mosaic2, 'outputs/mosaics/{}-mosaic2'.format(basedir))

for (v,u) in [(0,0), (0,2), (1,0), (1,2)]:
    ims = [d for d in images if load.get_chamber(d) == (v,u)]
    if len(ims):
        m = build_mosaic(ims)
        output_image(m, 'outputs/mosaics/{}--beads{}{}'.format(basedir, u, v))

label1,label2 = metadata.sample_info(basedir , respect_K=False)
write_labels(basedir, label1, label2)

predictions = {}
for ch, label in enumerate([label1, label2]):
    ch = (ch,1) # 0 -> (0,1) ; 1 -> (1,1)
    if len(features[ch]) > 0 and label is not None:
        predictions[ch] = predict_class(features[ch])
        log = write_metadata('outputs/metadata/{}.txt'.format(basedir), basedir, label, ch[0])
        all_preds = predict_class(features[ch], return_probs=True)
        paste_save_all_predictions(label, all_preds)
        measures1 = paste_measures(p, label, features[ch], predictions[ch])
        labeleds1 = renumber_labeled(labeleds[ch])
        outdir = regroup_data(p, label, visualizations[ch], measures1, labeleds1, log)

save_bead_brightness(dict(bead_brightness), basedir)
save_saturation(saturated)
save_beads(bead_sizes, basedir)

output_accepted_summary('outputs/label_keys/', basedir, accepted)
output_summary(basedir, predictions)

