import os
import socket

# path configuration

hostname = socket.gethostname()
if hostname == '': # Adjust to your path
    MAIN_DIR = ''
else:
    MAIN_DIR = os.getcwd()
ENV = ''

# site information
MAP_SCALE = 7 / 141     # 7m : 141px

SENSOR_BSSIDS = None

def get_working_dir():
    work_path = os.path.join(MAIN_DIR, ENV)
    if not os.path.exists(work_path):
        raise EnvironmentError('Cannot find the working directory %s.' % work_path)
    return work_path


def get_cache_dir():
    cache_path = os.path.join(MAIN_DIR, ENV, 'cache')
    if not os.path.exists(cache_path):
        os.makedirs(cache_path)
        print('Directory %s is created.' % cache_path)
    return cache_path


def meter_to_pixel(meter):
    return meter / MAP_SCALE
