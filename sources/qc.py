def measure_saturation(im):
    import numpy as np
    channels = ['dna', 'auto', 'DiO6', 'lysine']
    saturated = []
    for c in channels:
        c = im.get(c)
        saturated.append( (c == 255).sum(1).sum(1) )
    return np.array(saturated)
