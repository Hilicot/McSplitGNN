from options import opt
from src.mcsplit import mcsplit
import logging

def debug():
    logging.info(opt)

    mcsplit()


if __name__ == "__main__":
    debug()