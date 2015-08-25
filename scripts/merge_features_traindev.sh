#!/usr/bin/env bash

if [ -z $3 ]; then
    echo "usage: merge_features.sh <train folder> <test folder> <work folder>"
    exit
fi

TRAIN_DIR=$1
TEST_DIR=$2
WORK_DIR=$3

if [ -z $4 ]; then
    TOPN=1000
else
    TOPN=$4
fi

for group in south_slavic indonesian spanish ; do
    echo $group
    mkdir -p final/closed/matrix/$group
    mkdir -p final/closed/features/$group
    nice -n 15 python dsl/features/featurize_pearson.py \
        -N 3 \
        $TRAIN_DIR/$group/ \
        $TEST_DIR/$group/ \
        -t 5 \
        --train-out final/closed/features/$group/pearson_train.feat \
        --test-out final/closed/features/$group/pearson_test.feat \
        --topn $TOPN \
        --train-mtx final/closed/matrix/$group/pearson_train.mtx \
        --test-mtx final/closed/matrix/$group/pearson_test.mtx \

        python dsl/misc/merge_feature_tables.py \
        --filter <(cat $TEST_DIR/$group/*) \
        --output final/closed/matrix/$group/test \
        --labels final/test.labels \
        --mapping final/closed/matrix/$group/test.mapping \
        final/closed/features/chngram.test.mtx \
        final/closed/v1/tfidf/tfidf_test.feat \
        final/closed/v1/level2/groups/$group/group_tfidf_test.feat \
        final/closed/features/$group/pearson_test.feat \
        <(paste <(cat $TEST_DIR/$group/*) <(cut -f2- -d" " $WORK_DIR/$group/word_level/workdir/test_dense.mtx ) ) \
        <(paste <(cat $TEST_DIR/$group/*) <(cut -f2- -d" " $WORK_DIR/$group/stopword/workdir/test_dense.mtx ) )

        python dsl/misc/merge_feature_tables.py \
        --filter <(cat $TRAIN_DIR/$group/*) \
        --output final/closed/matrix/$group/train \
        --labels dat/train_dev/labels \
        --mapping final/closed/matrix/$group/train.mapping \
        final/closed/features/chngram.train.mtx \
        final/closed/v1/tfidf/tfidf_train.feat \
        final/closed/v1/level2/groups/$group/group_tfidf_train.feat \
        final/closed/features/$group/pearson_train.feat \
        <(paste <(cat $TRAIN_DIR/$group/*) <(cut -f2- -d" " $WORK_DIR/$group/word_level/workdir/train_dense.mtx ) ) \
        <(paste <(cat $TRAIN_DIR/$group/*) <(cut -f2- -d" " $WORK_DIR/$group/stopword/workdir/train_dense.mtx ) )
    done
