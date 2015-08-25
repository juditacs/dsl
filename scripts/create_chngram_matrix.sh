#!/usr/bin/env bash

if [ -z $3 ]; then
    echo "usage: subgroup_tfidf.sh <train folder> <test folder> <work folder>"
    exit
fi

TRAIN_DIR=$1
TEST_DIR=$2
WORK_DIR=$3

LANGID_DIR=/home/judit/projects/aacs15/langid
NORM_RARE=5
KATZ_RARE=30


#nice -n 15 python $LANGID_DIR/langid/langid.py --lower --train-dir $TRAIN_DIR --test-dir $TRAIN_DIR -N 1 --rare $NORM_RARE --filter-punct --train-mode normal --features $WORK_DIR/features_train_normal1 > /dev/null &
#nice -n 15 python $LANGID_DIR/langid/langid.py --lower --train-dir $TRAIN_DIR --test-dir $TEST_DIR -N 1 --rare $NORM_RARE --filter-punct --train-mode normal --features $WORK_DIR/features_test_normal1 > /dev/null &
#
#nice -n 15 python $LANGID_DIR/langid/langid.py --lower --train-dir $TRAIN_DIR --test-dir $TRAIN_DIR -N 2 --rare $NORM_RARE --filter-punct --train-mode normal --features $WORK_DIR/features_train_normal2 > /dev/null &
#nice -n 15 python $LANGID_DIR/langid/langid.py --lower --train-dir $TRAIN_DIR --test-dir $TEST_DIR -N 2 --rare $NORM_RARE --filter-punct --train-mode normal --features $WORK_DIR/features_test_normal2 > /dev/null &
#
#
#nice -n 15 python $LANGID_DIR/langid/langid.py --lower --train-dir $TRAIN_DIR --test-dir $TRAIN_DIR -N 3 --rare $NORM_RARE --filter-punct --train-mode normal --features $WORK_DIR/features_train_normal3 > /dev/null &
#nice -n 15 python $LANGID_DIR/langid/langid.py --lower --train-dir $TRAIN_DIR --test-dir $TEST_DIR -N 3 --rare $NORM_RARE --filter-punct --train-mode normal --features $WORK_DIR/features_test_normal3 > /dev/null &
#
#
#nice -n 15 python $LANGID_DIR/langid/langid.py --lower --train-dir $TRAIN_DIR --test-dir $TRAIN_DIR -N 2 --rare $KATZ_RARE --filter-punct --train-mode katz --features $WORK_DIR/features_train_katz2 > /dev/null &
#nice -n 15 python $LANGID_DIR/langid/langid.py --lower --train-dir $TRAIN_DIR --test-dir $TEST_DIR -N 2 --rare $KATZ_RARE --filter-punct --train-mode katz --features $WORK_DIR/features_test_katz2 > /dev/null &
#
#
#nice -n 15 python $LANGID_DIR/langid/langid.py --lower --train-dir $TRAIN_DIR --test-dir $TRAIN_DIR -N 3 --rare $KATZ_RARE --filter-punct --train-mode katz --features $WORK_DIR/features_train_katz3 > /dev/null &
#nice -n 15 python $LANGID_DIR/langid/langid.py --lower --train-dir $TRAIN_DIR --test-dir $TEST_DIR -N 3 --rare $KATZ_RARE --filter-punct --train-mode katz --features $WORK_DIR/features_test_katz3 > /dev/null &
#
#nice -n 15 python $LANGID_DIR/langid/langid.py --lower --train-dir $TRAIN_DIR --test-dir $TRAIN_DIR -N 4 --rare $KATZ_RARE --filter-punct --train-mode katz --features $WORK_DIR/features_train_katz4 > /dev/null &
#nice -n 15 python $LANGID_DIR/langid/langid.py --lower --train-dir $TRAIN_DIR --test-dir $TEST_DIR -N 4 --rare $KATZ_RARE --filter-punct --train-mode katz --features $WORK_DIR/features_test_katz4 > /dev/null &
#
#

paste \
    <(cut -f1- $WORK_DIR/features_train_katz2) \
    <(cut -f2- $WORK_DIR/features_train_katz3) \
    <(cut -f2- $WORK_DIR/features_train_katz4) \
    <(cut -f2- $WORK_DIR/features_train_normal1) \
    <(cut -f2- $WORK_DIR/features_train_normal2) \
    <(cut -f2- $WORK_DIR/features_train_normal3) \
    > $WORK_DIR/chngram.train.mtx


paste \
    <(cut -f1- $WORK_DIR/features_test_katz2) \
    <(cut -f2- $WORK_DIR/features_test_katz3) \
    <(cut -f2- $WORK_DIR/features_test_katz4) \
    <(cut -f2- $WORK_DIR/features_test_normal1) \
    <(cut -f2- $WORK_DIR/features_test_normal2) \
    <(cut -f2- $WORK_DIR/features_test_normal3) \
    > $WORK_DIR/chngram.test.mtx

