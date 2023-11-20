EXP_NAME="dinov2_feats_mlp1_20k"
EXP_DIR="./exps/${EXP_NAME}"
mkdir -p $EXP_DIR

LR=0.0001
EPOCHS=100
PER_CLASS_SIZE=10_000

OUTPUT_FILE="${EXP_DIR}/$(date +"%Y-%m-%d-%T").txt"

python -u \
    run.py train_mlp1 \
    --exp_dir $EXP_DIR --exp_name $EXP_NAME \
    --lr $LR --max_epochs $EPOCHS \
    --per_class_sample_size $PER_CLASS_SIZE \
     2>&1 | tee $OUTPUT_FILE