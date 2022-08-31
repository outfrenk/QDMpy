__version__ = "0.1.0a"

import logging
import os
import sys
from logging.config import fileConfig
from pathlib import Path

import matplotlib as mpl

mpl.rcParams["figure.facecolor"] = "white"

projectdir = os.path.dirname(os.path.abspath(__file__))
src_directory = os.path.join(projectdir, "..")
sys.path.append(projectdir)

from utils import load_config

logging_conf = Path(projectdir, "logging.conf")

fileConfig(logging_conf)

logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("h5py").setLevel(logging.WARNING)

LOG = logging.getLogger(f"QDMpy")

import coloredlogs

coloredlogs.install(
    level="DEBUG",
    fmt="%(asctime)s %(levelname)8s %(name)s.%(funcName)s >> %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
    logger=LOG,
    isatty=True,
)

LOG.info("WELCOME TO pyQDM")

settings = load_config()

desktop = os.path.join(os.path.expanduser("~"), "Desktop")

### CHECK IF pygpufit IS INSTALLED ###
import importlib.util

package = "pygpufit"
pygpufit_present = importlib.util.find_spec(package)  # find_spec will look for the package

if pygpufit_present is None:
    LOG.error(
        "Can't import pyGpufit. The package is necessary for most of the calculations. Functionality of QDMpy "
        "will be greatly diminished."
    )
    LOG.error(
        f"try running:\n"
        f">>> pip install --no-index --find-links={os.path.join(src_directory, 'pyGpufit', 'win', 'pyGpufit-1.2.0-py2.py3-none-any.whl')} pyGpufit"
    )
else:
    import pygpufit.gpufit as gf

    LOG.info(f"CUDA available: {gf.cuda_available()}")
    LOG.info("CUDA versions runtime: {}, driver: {}".format(*gf.get_cuda_version()))


from QDMpy.core.qdm import QDM
from QDMpy.core.odmr import ODMR
from QDMpy.core.fit import Fit

if __name__ == "__main__":
    LOG.info("This is a module. It is not meant to be run as a script.")
    sys.exit(0)