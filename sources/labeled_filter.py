def filter_labeled(objects, remove_bordering=False, min_size=None, max_size=None):
    '''
    Remove labeled regions that should not be analyzed further
    '''
    import mahotas as mh
    from mahotas.labeled import filter_labeled
    labeled,_ = mh.label(objects)
    labeled,_ = filter_labeled(labeled, remove_bordering=remove_bordering,
            min_size=min_size, max_size=max_size)
    return labeled


