#!/usr/bin/env bash

if [ -z $3 ]; then
    echo "usage: subgroup_tfidf.sh <train folder> <test folder> <work folder>"
    exit
fi

TRAIN_DIR=$1
TEST_DIR=$2
WORK_DIR=$3

for group in $(ls $TRAIN_DIR); do
    echo $group
    mkdir -p $WORK_DIR/$group
    python dsl/features/tfidf.py --train $TRAIN_DIR/$group --test $TRAIN_DIR/$group --topn 100000 --tokenize word --tf lognorm --idf smooth --rare 0 --qf smooth --filter-punct --replace-digits > $WORK_DIR/$group/group_tfidf_train.feat
    python dsl/features/tfidf.py --train $TRAIN_DIR/$group --test $TEST_DIR/$group --topn 100000 --tokenize word --tf lognorm --idf smooth --rare 0 --qf smooth --filter-punct --replace-digits > $WORK_DIR/$group/group_tfidf_test.feat
done
