import numpy as np
def mean_var(images, channel, project=True):
    import load
    from segment import almost_max
    means = []
    vars = []
    for im in images[::3]:
        im = load.load_directory(im)
        im = im.get(channel)
        if project:
            im = almost_max(im)
        means.append(im.mean())
        vars.append(im.var())
    return np.array([np.mean(means), np.mean(vars)])

def compute_median_top(images, channel):
    import load
    tops = []
    for im in images[::7]:
        im = load.load_directory(im)
        x,y = load.field_coordinates(im)
        if x in (0,11) or y in (0,14):
            continue
        try:
            tops.append(im.get(channel, 31))
        except:
            tops.append(im.get(channel, 4))
    return np.median(tops, axis=0)


def preprocess_median(t_median):
    import mahotas as mh
    t33 = mh.median_filter(t_median, np.ones((33,33)))
    t33_24 = mh.gaussian_filter(t33, 24)
    return t33_24

def fit_smooth(t_median):
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    import numpy as np
    s0,s1 = t_median.shape
    X,Y = np.mgrid[:s0, :s1]
    D = (X-s0/2.)**2 + (Y-s1/2.)**2
    X = np.dstack([D, X,Y]).reshape((-1, 3))
    Xp = PolynomialFeatures(2).fit_transform(X)

    lr = LinearRegression()
    lr.fit(Xp, t_median.ravel())
    return lr.predict(Xp).reshape((s0,s1))


def normalize_by_mean(im, t_median):
    if t_median is None:
        return im
    min_val = t_median.min()
    t_median = np.maximum(t_median - min_val/10., 1.0)
    return (im-min_val)/t_median



