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
    mkdir -p final/matrix/$group
    mkdir -p final/features/$group
    nice -n 15 python dsl/features/featurize_pearson.py \
        -N 3 \
        $TRAIN_DIR/$group/ \
        $TEST_DIR/$group/ \
        -t 5 \
        --train-out final/features/$group/pearson_train.feat \
        --test-out final/features/$group/pearson_test.feat \
        --topn $TOPN \
        --train-mtx final/matrix/$group/pearson_train.mtx \
        --test-mtx final/matrix/$group/pearson_test.mtx \

        python dsl/misc/merge_feature_tables.py \
        --filter <(cat $TEST_DIR/$group/*) \
        --output final/matrix/$group/test_3k \
        --labels final/features/dev.labels \
        --mapping final/matrix/$group/test.mapping \
        final/features/chngram.with_sentences.dev \
        final/experiments/tfidf/tfidf_test.feat \
        final/experiments/tfidf/dat/level2/groups/$group/group_tfidf_test.feat \
        final/features/$group/pearson_test.feat \
        <(paste <(cat $TEST_DIR/$group/*) <(cut -f2- -d" " $WORK_DIR/$group/word_level/workdir/test_dense.mtx ) ) \
        <(paste <(cat $TEST_DIR/$group/*) <(cut -f2- -d" " $WORK_DIR/$group/stopword/workdir/test_dense.mtx ) )

        python dsl/misc/merge_feature_tables.py \
        --filter <(cat $TRAIN_DIR/$group/*) \
        --output final/matrix/$group/train_3k \
        --labels final/features/train.labels \
        --mapping final/matrix/$group/train.mapping \
        final/features/chngram.with_sentences.train \
        final/experiments/tfidf/tfidf_train.feat \
        final/experiments/tfidf/dat/level2/groups/$group/group_tfidf_train.feat \
        final/features/$group/pearson_train.feat \
        <(paste <(cat $TRAIN_DIR/$group/*) <(cut -f2- -d" " $WORK_DIR/$group/word_level/workdir/train_dense.mtx ) ) \
        <(paste <(cat $TRAIN_DIR/$group/*) <(cut -f2- -d" " $WORK_DIR/$group/stopword/workdir/train_dense.mtx ) )
    done
