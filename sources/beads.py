from jug import TaskGenerator
from utils import mkdir_p as maybe_mkdir

def save_beads(bead_sizes, basedir):
    import load
    from collections import defaultdict
    grouped = defaultdict(list)
    for im,cs in bead_sizes.iteritems():
        u_v = load.get_chamber(im)
        grouped[u_v].append(cs)
    for (u,v),beads in grouped.iteritems():
        if len(beads):
            maybe_mkdir('outputs/bead_stats')
            do_save_beads('outputs/bead_stats/{}{}_{}.npy'.format(u, v, basedir), beads)


@TaskGenerator
def save_bead_brightness(bead_brightness, basedir):
    import numpy as np
    maybe_mkdir('outputs/bead_brightness')
    for k,val in bead_brightness.iteritems():
        val = [v for v in val if len(v)]
        u,v = k
        if len(val):
            val = np.concatenate(val)
            np.save('outputs/bead_brightness/{}_{}_{}.npy'.format(u,v,basedir), val)


@TaskGenerator
def do_save_beads(output, beads):
    import numpy as np
    beads = [b[1:] for b in beads]
    beads = np.concatenate(beads)
    np.save(output, beads)
