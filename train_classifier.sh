OUTPUT_FILE="./logs/$(date +"%Y-%m-%d-%T").txt"
python -u run.py train_classifier 2>&1 | tee $OUTPUT_FILE