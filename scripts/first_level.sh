#!/usr/bin/env bash

if [ -z $3 ]; then
    echo "usage: first_level.sh <train folder> <test folder> <work folder>"
    exit
fi

TRAIN_DIR=$1
TEST_DIR=$2
WORK_DIR=$3

python dsl/features/tfidf.py --train $TRAIN_DIR --test $TEST_DIR --topn 100000 --tokenize word --tf lognorm --idf smooth --rare 0 --qf smooth --filter-punct --replace-digits > $WORK_DIR/tfidf_test.feat
python dsl/features/tfidf.py --train $TRAIN_DIR --test $TRAIN_DIR --topn 100000 --tokenize word --tf lognorm --idf smooth --rare 0 --qf smooth --filter-punct --replace-digits > $WORK_DIR/tfidf_train.feat

mkdir -p $WORK_DIR/dat/groups
python dsl/misc/create_subgroups.py $TEST_DIR $WORK_DIR/tfidf_test.feat $WORK_DIR/dat/groups
