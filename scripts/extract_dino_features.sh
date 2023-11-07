EXP_NAME="feats_uk1M"
EXP_DIR="./exps/${EXP_NAME}"
MODE='giant'
TRAIN_MODE="freeze"
mkdir -p $EXP_DIR

OUTPUT_FILE="${EXP_DIR}/$(date +"%Y-%m-%d-%T").txt"
python -u run.py extract_dino_features --mode $MODE 2>&1 | tee $OUTPUT_FILE