def get_lsb_cpus():
    from os import environ
    if 'LSB_MCPU_HOSTS' not in environ:
        return 1
    tokens = environ['LSB_MCPU_HOSTS'].strip().split()
    if len(tokens) > 2:
        raise SystemError("Cannot handle this type of environment (LSB_MCPU_HOSTS='{}')".format(environment['LSB_MCPU_HOSTS']))
    return int(tokens[1])

if __name__ == '__main__':
    print('Running with {} CPUS.'.format(get_lsb_cpus()))
