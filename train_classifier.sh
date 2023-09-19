EXP_NAME="resnet50"
EXP_DIR="./exps/${EXP_NAME}"
mkdir -p $EXP_DIR

OUTPUT_FILE="${EXP_DIR}/$(date +"%Y-%m-%d-%T").txt"
python -u run.py train_classifier --exp_dir $EXP_DIR 2>&1 | tee $OUTPUT_FILE