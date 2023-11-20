EXP_NAME="dinov2_feats_mlp3_160k"
EXP_DIR="./exps/${EXP_NAME}"
mkdir -p $EXP_DIR

LR=0.00001
H1=500
H2=50
EPOCHS=20
PER_CLASS_SIZE=80_000

OUTPUT_FILE="${EXP_DIR}/$(date +"%Y-%m-%d-%T").txt"
python -u \
    run.py train_mlp3 \
    --exp_dir $EXP_DIR --exp_name $EXP_NAME \
    --hidden_size1 $H1 --hidden_size2 $H2 --lr $LR \
    --max_epochs $EPOCHS \
    --per_class_sample_size $PER_CLASS_SIZE \
     2>&1 | tee $OUTPUT_FILE