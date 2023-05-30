import os
import sys
import logging
from colorlog import ColoredFormatter
from datetime import datetime

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

opt.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# initialize logger
log_format = '%(levelname)s | %(asctime)s | %(filename)s line:%(lineno)d | %(message)s'
dateformat = '%H:%M:%S'
colored_format = ColoredFormatter("%(log_color)s" + log_format + "%(reset)s",datefmt=dateformat,log_colors={
		'DEBUG':    'light_cyan',
		'INFO':     'light_white',
		'WARNING':  'light_yellow',
		'ERROR':    'light_red',
		'CRITICAL': 'light_red,bg_white',
	},)


handlers = []
if opt.log_folder is not None:
    handlers.append(logging.FileHandler(os.path.join(opt.log_folder,f"{opt.timestamp}.txt")))
if opt.log_stdout:
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(colored_format)
    handlers.append(stream)
logging.basicConfig(
    level=opt.log_level.upper(),
    format=log_format,
    datefmt=dateformat,
    handlers=handlers)

