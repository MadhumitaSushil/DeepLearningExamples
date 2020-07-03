export BERT_PREP_WORKING_DIR="${BERT_PREP_WORKING_DIR}"

# Shard the text files
python3 ${BERT_PREP_WORKING_DIR}/bertPrep.py --action sharding --dataset medbooks --n_training_shards 490 --n_test_shards 4

### BERT BASE

## UNCASED

# Create TFRecord files Phase 1
python3 ${BERT_PREP_WORKING_DIR}/bertPrep.py --action create_tfrecord_files --dataset medbooks --max_seq_length 128 \
 --max_predictions_per_seq 20 --vocab_file ${BERT_PREP_WORKING_DIR}/download/google_pretrained_weights/uncased_L-12_H-768_A-12/vocab.txt --n_training_shards 490 --n_test_shards 4


# Create TFRecord files Phase 2
python3 ${BERT_PREP_WORKING_DIR}/bertPrep.py --action create_tfrecord_files --dataset medbooks --max_seq_length 512 \
 --max_predictions_per_seq 80 --vocab_file ${BERT_PREP_WORKING_DIR}/download/google_pretrained_weights/uncased_L-12_H-768_A-12/vocab.txt --n_training_shards 490 --n_test_shards 4


## CASED

# Create TFRecord files Phase 1
python3 ${BERT_PREP_WORKING_DIR}/bertPrep.py --action create_tfrecord_files --dataset medbooks --max_seq_length 128 \
 --max_predictions_per_seq 20 --vocab_file ${BERT_PREP_WORKING_DIR}/download/google_pretrained_weights/cased_L-12_H-768_A-12/vocab.txt \
 --do_lower_case=0 --n_training_shards 490 --n_test_shards 4


# Create TFRecord files Phase 2
python3 ${BERT_PREP_WORKING_DIR}/bertPrep.py --action create_tfrecord_files --dataset medbooks --max_seq_length 512 \
 --max_predictions_per_seq 80 --vocab_file ${BERT_PREP_WORKING_DIR}/download/google_pretrained_weights/cased_L-12_H-768_A-12/vocab.txt \
 --do_lower_case=0 --n_training_shards 490 --n_test_shards 4
