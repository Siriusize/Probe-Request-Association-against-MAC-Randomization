import os, sys
import pickle

def pickle_object(obj, pkl_filepath, override=True):
    """
    Dump the object by pickle. The container directory will be automatically created if it does not exist.
    :param obj:
    :param pkl_filepath:
    :param override:
    """
    if os.path.exists(pkl_filepath):
        if not override:
            print('Skip writing to %s. The pickle file exists.' % pkl_filepath)
            return
    pkl_dirpath = os.path.dirname(pkl_filepath)
    if not os.path.exists(pkl_dirpath):
        print('Create folder %s.' % pkl_dirpath)
        os.makedirs(pkl_dirpath)
    with open(pkl_filepath, 'wb') as pkl_file:
        pickle.dump(obj, pkl_file)


def load_pickle(pkl_filepath):
    """
    Load the object by pickle.
    :param pkl_filepath:
    :return: The loaded object
    """
    if not os.path.exists(pkl_filepath):
        raise ValueError('Cannot load pickle from %s because the file does not exist.' % pkl_filepath)
    with open(pkl_filepath, 'rb') as pkl_file:
        return pickle.load(pkl_file)


def construct_or_load(choice, cache_name=None, cache_dirpath=None):

    if choice not in ['auto', 'construct', 'auto']:
        raise ValueError('"%s" is not a valid value for parameter "choice".' % choice)

    def decorator(func):

        def wrapper(*args, **kwargs):
            if choice == 'construct':
                will_load, will_construct = False, True
            elif choice == 'load':
                will_load, will_construct = True, False
            else:
                will_load, will_construct = True, True

            r = None
            if will_load:
                if cache_name is None:
                    raise ValueError('"cache_name" is not specified.')
                if cache_dirpath is None:
                    raise ValueError('"cache_dirpath" is not specified.')
                try:
                    r = load_pickle(os.path.join(cache_dirpath, cache_name + '.pkl'))
                except ValueError as err:
                    if not will_construct:
                        raise err
            if r is None and will_construct:
                r = func(*args, **kwargs)
            return r

        return wrapper

    return decorator
