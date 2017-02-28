def mkdir_p(dirname):
    from os import makedirs
    try:
        makedirs(dirname)
    except:
        pass

def fsync_all(paths):
    import os
    for fname in paths:
        fd = os.open(fname, os.O_RDONLY)
        try:
            os.fsync(fd)
        except OSError as err:
            if err.errno != 22:
                raise
        finally:
            os.close(fd)

def sync_move(src, dest):
    from os import path
    from shutil import move
    fsync_all([src, path.dirname(src)])
    move(src, dest + '.tmp')

    fsync_all([dest + '.tmp'])
    move(dest + '.tmp', dest)
