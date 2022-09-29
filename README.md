
<!-- README.md is generated from README.Rmd. Please edit that file -->

# deepdirect

<!-- badges: start -->

<!-- badges: end -->

Deepdirect is an in silico approach to generate mutations for protein
complexes towards a specified direction (increase/decrease) in binding
affinity.

## System requirements and dependencies

### Hardware Requirements

`deepdirect` model is able to be trained and perform its operations on a
standard computer.

### OS Requirements

The `deepdirect` model should be compatible with Windows, Mac, and Linux
operating systems. The package has been tested on the following systems:

  - Linux 3.10.0
  - Windows 10

### Dependencies

`deepdirect` framework is built and trained on the `Tensorflow 2.4.0`
and `Keras 2.4.0`.

## Framework construction

The python file including all required deepdirect framework built
function is able to be downloaded from GitHub:
`deepdirect_framework/model_function.py`

## Data structure

  - `deepdirect_framework` folder contains the trained model weights and
    the model constructing functions.
  - `deepdirect_paper` folder contains codes for building and training
    models, and performing analysis in the deepdirect manuscript. The
    file `ab_bind_data_extract.py` and `skempi_data_extract.py` contains
    code for constructing training datasets for Deepdirect framework.
    `train_step_1.py` contains code for training step 1 for the mutation
    mutator. `final_model.py` contains code for training step 2 (final)
    for the mutation mutator. `model_function.py` contains code for
    constructing the Deepdirect framework.
    `model_evaluation_application.py` contains code for model
    evaluation, and teh application on Novavax-vaccine.
    `evolution_analysis.py` contains code for performing evolution
    analysis.

## File source

For files that are required as input in the code but not generated from
other codes, please refer to the data availability section in the
original paper.

## Issues and bug reports

Please use <https://github.com/tianlt/deepdirect/issues> to submit
issues, bug reports, and comments.

## License

deepdirect is distributed under the [GNU General Public License
version 2
(GPLv2)](https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html).
