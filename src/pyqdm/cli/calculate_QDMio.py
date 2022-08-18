#!/usr/bin/python

import sys, getopt
import argparse
import time
import logging

import pyqdm
from pyqdm.qdm import QDM

def main(argv):
    inpath = None
    outpath = None
    tstart = time.process_time()

    parser = argparse.ArgumentParser(description='Calculate the B111 field from ODMR data recorded with QDMio made QDM')
    parser.add_argument('-i', '--input',
                        help='input path, location of the QDM data files and LED/laser images.',
                        required=True)
    parser.add_argument('-b', '--binfactor', type=int,
                        help='Binning factor of the ODMR data. Default: 1',
                        default=1,
                        required=False)
    parser.add_argument('-d', '--diamond', type=str,
                        help='Type of diamond used in the experiment. Default: \'auto\'',
                        default=None,
                        required=False)
    parser.add_argument('-gf', '--globalfluorescence', type=float,
                        help='Global fluorescence of the sample. Default: 0.2',
                        default=0.2,
                        required=False)
    parser.add_argument('--debug',
                        help='sets logging to DEBUG level',
                        action='store_true',
                        default=False,
                        required=False)

    args = parser.parse_args()

    if args.debug:
        pyqdm.LOG.setLevel("DEBUG")
    else:
        pyqdm.LOG.setLevel("INFO")

    QDMobj = QDM.from_QDMio(args.input, diamond_type=args.diamond)
    QDMobj.bin_data(bin_factor=args.binfactor)
    QDMobj.correct_glob_fluorescecne(glob_fluorescence=args.globalfluorescence)
    QDMobj.fit_ODMR()
    QDMobj.export_QDMio()
    pyqdm.LOG.info("pyqdm finished in {:.2f} seconds".format(time.process_time()-tstart))

if __name__ == "__main__":
    main(sys.argv[1:])
