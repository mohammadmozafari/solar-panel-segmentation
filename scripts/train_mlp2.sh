EXP_NAME="dinov2_feats_mlp2_160k"
EXP_DIR="./exps/${EXP_NAME}"
mkdir -p $EXP_DIR

LR=0.000001
H=700
EPOCHS=100
PER_CLASS_SIZE=80_000

OUTPUT_FILE="${EXP_DIR}/$(date +"%Y-%m-%d-%T").txt"
python -u \
    run.py train_mlp2 \
    --exp_dir $EXP_DIR --exp_name $EXP_NAME \
    --hidden_size $H --lr $LR \
    --max_epochs $EPOCHS \
    --per_class_sample_size $PER_CLASS_SIZE \
     2>&1 | tee $OUTPUT_FILE