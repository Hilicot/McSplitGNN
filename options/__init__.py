import os
import sys
import logging
from colorlog import ColoredFormatter

# get name of caller (test, train or debug)
caller = sys.argv[0].split(os.sep)[-1].split(".")[0]


# depending on caller, import the correct options
opt = None
if caller == "train":
    from .train_options import TrainOptions
    to = TrainOptions()
    to.initialize()
    opt = to.parse()
elif caller == "test":
    from .test_options import TestOptions
    to = TestOptions()
    to.initialize()
    opt = to.parse()
elif caller == "debug":
    from .debug_options import DebugOptions
    do = DebugOptions()
    do.initialize()
    opt = do.parse()

if opt is None:
    raise ValueError("opt is None. you should call this script from train.py, test.py or debug.py.")

# initialize logger
log_format = '%(levelname)s | %(asctime)s | %(filename)s line:%(lineno)d | %(message)s'
dateformat = '%H:%M:%S'
colored_format = ColoredFormatter("%(log_color)s" + log_format + "%(reset)s",datefmt=dateformat,log_colors={
		'DEBUG':    'cyan',
		'INFO':     'light_white',
		'WARNING':  'yellow',
		'ERROR':    'red',
		'CRITICAL': 'red,bg_white',
	},)


handlers = []
if opt.log_file is not None:
    handlers.append(logging.FileHandler(opt.log_file))
if opt.log_stdout:
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(colored_format)
    handlers.append(stream)
logging.basicConfig(
    level=opt.log_level.upper(),
    format=log_format,
    datefmt=dateformat,
    handlers=handlers)

