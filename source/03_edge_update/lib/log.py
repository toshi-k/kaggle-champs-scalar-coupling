import os
import logging
from pathlib import Path


def init_logger(path, name='root'):

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s %(message)s")

    handler1 = logging.StreamHandler()
    handler1.setFormatter(formatter)

    dir_log = Path(path).parent

    dir_log.mkdir(exist_ok=True)

    os.makedirs(dir_log, exist_ok=True)
    handler2 = logging.FileHandler(filename=path)
    handler2.setFormatter(formatter)

    logger.addHandler(handler1)
    logger.addHandler(handler2)

    return logger
