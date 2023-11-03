EXP_NAME="dinov2_giant_freeze_2stacked_fulluk"
EXP_DIR="./exps/${EXP_NAME}"
MODE='giant'
TRAIN_MODE="freeze"
mkdir -p $EXP_DIR

OUTPUT_FILE="${EXP_DIR}/$(date +"%Y-%m-%d-%T").txt"
python -u run.py train_classifier --exp_dir $EXP_DIR --mode $MODE --train_mode $TRAIN_MODE --exp_name $EXP_NAME 2>&1 | tee $OUTPUT_FILE