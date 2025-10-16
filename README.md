Some code from John for training neural networks for the ttH(cc) analysis.
This has been used so far for studies on the impact flavor tagging granularity would have on the analysis.

The NN's are trained using TensorFlow, interfaced to Keras. Prior to training, the data must be converted from ROOT to HDF5 format. This is done with the script HMaker.py.

In order to access all of the packages, there are singularity images provided by the top team. For running on lxplus, do:

export SINGULARITY_CACHEDIR="/tmp/$(whoami)/singularity"
mkdir -p $SINGULARITY_CACHEDIR

And then for the HDF5 conversion:

singularity shell -B /eos /eos/home-k/kzoch/top-ml-tutorial/singularity-images/conversion.sif 

or for the training:

/eos/home-k/kzoch/top-ml-tutorial/singularity-images/training.sif

To run the HDF5 maker:

python HMaker.py -i (input)  -f (fraction=1.0)

input can be ttH_cc, etc. Setting a fraction less than 1 saves a random sub-sample of the events, if you are limited by disk space or memory.

To  run the training:

python TrainNN.py (options)

options are:
-f to run all 5 folds (only run the first fold if it is not set)
-s for the flavor tagging scheme:
   1: default PCFT bins
   2: extra tight b-tag bins
   3: extra medium b-tag bins
   4: extra tight c-tag bins
   5: extra medium c-tag bins
   6: extra loose c-tag bins
   7: all extra PCFT bins
   8: discriminant values Db and Dc
   9: GN2 probability values
-p if you have already done the training previously and just want to evaluate

there are also options for studying parameters or variable list.

The output models can be converted into ONNX format using the python script:

python -m tf2onnx.convert --saved-model THC_NN01scheme1 --output thc_nn01scheme1.onnx

The suffixes 01, 23, etc are the last digits of the event numbers to be used for testing.
