set -e
export name="sirsam"
# extract features
landshark-import --nworkers 0 --batch-mb 0.001 tifs \
    --name ${name} \
    --ignore-crs  \
    --continuous ../integration/data/continuous \
#    --continuous_list ../integration/data/continuous/continuous_list.txt




# extract targets
landshark-import --batch-mb 0.001 targets \
  --shapefile ../integration/data/targets/geochem_sites.shp \
  --name ${name} \
  --record Na_ppm_i_1 \
  --dtype continuous \
  --record Zr_ppm_i_1

# extract query for prediction
landshark-extract query \
    --features features_${name}.hdf5 \
    --strip 5 10 \
    --name ${name} \
    --halfwidth 1


echo "===========entering train test block ============================"
echo "===========entering train test block ============================"
echo "===========entering train test block ============================"
echo "===========entering train test block ============================"

# create train/test data split
landshark-extract --nworkers 0 --batch-mb 0.001 traintest \
  --features features_${name}.hdf5 \
  --split 1 10 \
  --targets targets_${name}.hdf5 \
  --name ${name} \
  --halfwidth 1


# train using train test split
landshark --keras-model train \
  --trainvalidation false \
  --data traintest_${name}_fold1of10/ \
  --config nn_regression_keras.py  \
  --epochs 20 \
  --iterations 5


# predict using train/test data based training
landshark --keras-model --batch-mb 0.001 predict \
    --config nn_regression_keras.py \
    --checkpoint nn_regression_keras_model_1of10 \
    --data query_${name}_strip5of10

# use validation shapefile to extract a validation

# extract targets

echo "===========entering validation training block ============================"
echo "===========entering validation training block ============================"
echo "===========entering validation training block ============================"
echo "===========entering validation training block ============================"

landshark-import --batch-mb 0.001 targets \
  --shapefile ../integration/data/targets/geochem_sites.shp \
  --name validation_${name} \
  --record Na_ppm_i_1 \
  --record Zr_ppm_i_1 \
  --dtype continuous

# extract validation data - note I am using the same targets as targets and validation here
# targets and validation hdf5 files could be different with a different validation shapefile/hdf5
landshark-extract --nworkers 0 --batch-mb 0.001 trainvalidate \
  --features features_${name}.hdf5 \
  --targets targets_${name}.hdf5 \
  --validation targets_validation_${name}.hdf5 \
  --name ${name} \
  --halfwidth 1

# train using validation data
landshark --keras-model train \
  --trainvalidation true \
  --data trainvalidate_${name}/   \
  --config nn_regression_keras.py   \
  --epochs 20 \
  --iterations 5

# predict using train/validation data based training
landshark --keras-model --batch-mb 0.001 predict \
    --config nn_regression_keras.py \
    --checkpoint nn_regression_keras_model_train_validation/ \
    --data query_${name}_strip5of10


# ======================oos validation block =============================
# import oos data from oos shapefile into target oos hdf5 file
# note I am using the same shapefile even as oos shapefile here
echo "===========entering oos validation block ============================"
echo "===========entering oos validation block ============================"
echo "===========entering oos validation block ============================"
echo "===========entering oos validation block ============================"

landshark-import --batch-mb 0.001 targets \
  --shapefile ../integration/data/targets/geochem_sites.shp \
  --name ${name}_oos \
  --record Na_ppm_i_1 \
  --record Zr_ppm_i_1 \
  --dtype continuous

# extract oos targets into tfrecord
# TODO: can avoid saving one of train.xxx.tfrecord and test.xxx.tfrecord
landshark-extract --nworkers 0 --batch-mb 0.01 traintest \
  --features features_${name}.hdf5 \
  --split 1 1 \
  --targets targets_${name}_oos.hdf5 \
  --name ${name}_oos \
  --halfwidth 1

# predict oos score
landshark -v DEBUG --keras-model --batch-mb 0.001 predict_oos \
    --proba false \
    --config nn_regression_keras.py \
    --checkpoint nn_regression_keras_model_1of10 \
    --data traintest_${name}_oos_fold1of1 \
    --pred_ensemble_size 12000000  # not used

exit 0