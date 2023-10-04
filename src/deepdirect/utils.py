# -*- coding: utf-8 -*-
# @Author: Pengyao Ping
# @Date:   2023-02-16 11:01:06
# @Last Modified by:   Pengyao Ping
# @Last Modified time: 2023-04-07 17:22:18

import logging
from colorlog import ColoredFormatter
import datetime

def custom_logger(root_name, debug_mode) -> logging.Logger: 
    logger = logging.getLogger(root_name)    
    
    formatter = ColoredFormatter(
        "%(green)s[%(asctime)s] %(blue)s%(name)s %(log_color)s%(levelname)-8s%(reset)s %(message)s",
        datefmt=None,
        reset=True,
        log_colors={
            'DEBUG':    'cyan',
            'INFO':     'green',
            'WARNING':  'yellow',
            'ERROR':    'red',
            'CRITICAL': 'red, bg_white',
        },
        secondary_log_colors={},
        style='%'
    )
    if debug_mode:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    
    # Output full log
    file_handler = logging.FileHandler( datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_") + 'deepdirect.log')

    file_handler.setLevel(logging.INFO)

    # formatter = logging.Formatter(log_format)
    logger.addHandler(file_handler)

    # # Output warning log
    # file_handler = logging.FileHandler('deepdirect.Warning.log')
    # file_handler.setLevel(logging.WARNING)
    # # formatter = logging.Formatter(log_format)
    # file_handler.setFormatter(formatter)
    # logger.addHandler(file_handler)

    # # Output error log
    # file_handler = logging.FileHandler('deepdirect.Error.log')
    # file_handler.setLevel(logging.ERROR)
    # # formatter = logging.Formatter(log_format)
    # file_handler.setFormatter(formatter)
    # logger.addHandler(file_handler)

    return logger

def usage():
    # print(" ")
    print("deepdirect mutator required parameters:")
    # print("   Mandatory:")
    print("     pre                   sequence before mutation")
    print("     rbd                   receptor binding domian")
    print("     same                  chain indicator")
    print("     xx                    x axis of protein complex (x axis, yaxis, zaxis)")
    print("     yy                    y axis of protein complex (x axis, yaxis, zaxis)")
    print("     zz                    z axis of protein complex (x axis, yaxis, zaxis)")
    print("     noise                 nosie")


    # print("   Mandatory:")
    # print("     -p|--pre                   sequence before mutation")
    # print("     -r|--rbd                   receptor binding domian")
    # print("     -s|--same                  chain indicator")
    # print("     -x|--xx                    x axis of protein complex (x axis, yaxis, zaxis)")
    # print("     -y|--yy                    y axis of protein complex (x axis, yaxis, zaxis)")
    # print("     -z|--zz                    z axis of protein complex (x axis, yaxis, zaxis)")
    # print("     -i|--input_noi             nosie")

    # print("   Modules: [mutator]")
    # # print(" ")
    # print("1. Using config file")
    # print("     deepdirect -m|--module <module_name> -c <path_configuration_file>")
    # print("   Mandatory:")
    # print("     -c|--config                   input configuration file")
    # # print(" ")
    # print("2. Using command line with the default parameters")
    # print("     deepdirect -m|--module <module_name> -i <path_raw_data.fastq|fasta|fa|fq>")
    # print("   Mandatory:")
    # print("     -i|--input                    input raw data to be corrected")
    # print("   Options:")
    # print("     -d|--directory                set output directory")
    # print("     -a|--high_ambiguous           predict high ambiguous errors using machine learning when set true, defaut true")
    # print("     -t|--true                     input ground truth data if you have")
    # print("     -r|--rectification            input corrected data when using module evaluation")
    # print("     -p|--parallel                 use multiple cpu cores, default total cpu cores - 2")
    # print("     -g|--tree_method              use gpu for training and prediction, default auto, (options gpu_hist, hist, auto)")
    # # print("     -o|--over_sampling            use over sampling or downsampling for negative samples, default True")
    # print("     -h|--help                     show this help")