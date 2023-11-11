EXP_NAME="dinov2_feats_mlp2_20k"
EXP_DIR="./exps/${EXP_NAME}"
mkdir -p $EXP_DIR

OUTPUT_FILE="${EXP_DIR}/$(date +"%Y-%m-%d-%T").txt"
python -u run.py train_mlp2 --exp_dir $EXP_DIR --exp_name $EXP_NAME 2>&1 | tee $OUTPUT_FILE