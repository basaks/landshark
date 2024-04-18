# Landshark regression stages.sh 

This workflow has the following steps in this location: 
/g/data/ge3/sudipta/jobs/landshark_models/john_test/cnn_binary_class

## extract covariates (creating HDF5 files)
## extract targets (as above)
## extract folds for x-val --this example uses 5 jobs for folds
## train fold -- as above 5 jobs  # you will have 5 models already
## do final training with all data (use the submit.py script for this)   # optional - we used the same data to train and validate
## predict (this example submits jobs in 10 nodes in parallel - each node run 1/10th of the prediction with 48 jobs each node - making a total of 480 prediction jobs)
## finally merge the predictions using the merge*.txt files via gdal_calc


There are a few other features that could be useful:

## how to use a validation shapefile - this provides functionality to supply a different validation set during model training 
## how to use a oos shapefile (for regression only atm)
