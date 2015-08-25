#!/usr/bin/env bash

if [ -z $4 ]; then
    echo "usage: word_level.sh <train folder> <test folder> <work folder>"
    exit
fi

TRAIN_DIR=$1
TEST_DIR=$2
WORK_DIR=$3
TOP=$4

for group in $(ls $TRAIN_DIR ); do
    echo $group
    mkdir -p $WORK_DIR/$group/word_level/raw
    mkdir -p $WORK_DIR/$group/word_level/workdir
    #nice -n 15 python dsl/features/word_level.py --train $TRAIN_DIR/$group --raw $WORK_DIR/$group/word_level/raw --workdir $WORK_DIR/$group/word_level/workdir --test  $TEST_DIR/$group --topn $TOP --threshold 5 &

    mkdir -p $WORK_DIR/$group/stopword/train
    mkdir -p $WORK_DIR/$group/stopword/test
    mkdir -p $WORK_DIR/$group/stopword/raw
    mkdir -p $WORK_DIR/$group/stopword/workdir

    #nice -n 15 python dsl/utils/filter_freq.py --train $TRAIN_DIR/$group --topn 200 --test $TEST_DIR/$group --train-out $WORK_DIR/$group/stopword/train --test-out $WORK_DIR/$group/stopword/test &
    nice -n 15 python dsl/features/word_level.py --train $WORK_DIR/$group/stopword/train --raw $WORK_DIR/$group/stopword/raw --workdir $WORK_DIR/$group/stopword/workdir --test $WORK_DIR/$group/stopword/test --topn $TOP --threshold 5 &
done
