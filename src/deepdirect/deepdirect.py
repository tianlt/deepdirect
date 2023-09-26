
from deepdirect.model import build_model, build_aa_mutator
from deepdirect.utils import custom_logger, usage
import sys, getopt
import deepdirect
import os
import pkg_resources

def main():
    argv = sys.argv[1:]
    # create logger
    logger = custom_logger("deepdirect", debug_mode=False)
    try:
        opts, args = getopt.getopt(argv, "hvo:", ["help", "version", "output="]) 

        # aa_mutator.predict([pre, rbd, same, x, y, z, input_noi])        
    except getopt.GetoptError as err:
        logger.error(err)
        usage()
        sys.exit(2)

    logger.info("Version: " + "".join(deepdirect.__version__))
    # logger.info("Commands: " + " ".join(input_commands))

    for opt, a in opts:
        if opt in ("-v", "--version"):
            logger.info("Version: " + "".join(deepdirect.__version__))
        elif opt in ("-h", "--help"):
            usage()
            sys.exit()
        elif opt in ("-o", "--output"):
            output = a
        else:
            assert False, "unhandled option"

    relative_path_to_weight = 'weights/model_i_weights.h5'
    weight_file_path = pkg_resources.resource_filename('deepdirect', relative_path_to_weight)

    aa_mutator = build_aa_mutator()

    aa_mutator.load_weights(weight_file_path)

if __name__=="__main__":
    main()