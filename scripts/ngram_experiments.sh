#!/usr/bin/env bash

langid_dir=/home/judit/projects/aacs15/langid
normal_rare=0
katz_rare=30

feat_out=/home/judit/projects/dsl/features
data_dir=/home/judit/projects/dsl/dat/t1_noxx

N=( 1 2 3 4 )

for n in ${N[@]}; do
    # normal
    nice -n 15 python $langid_dir/langid/langid.py --lower --train-dir $data_dir/train/ --test-dir $data_dir/train -N $n --rare $normal_rare --filter-punct --train-mode normal --features $feat_dir/features_train_normal${n}
    nice -n 15 python $langid_dir/langid/langid.py --lower --train-dir $data_dir/train/ --test-dir $data_dir/dev -N $n --rare $normal_rare --filter-punct --train-mode normal --features $feat_dir/features_dev_normal${n}
    # katz
    nice -n 15 python $langid_dir/langid/langid.py --lower --train-dir $data_dir/train/ --test-dir $data_dir/train -N $n --rare $katz_rare --filter-punct --train-mode katz --features $feat_dir/features_train_katz${n}
    nice -n 15 python $langid_dir/langid/langid.py --lower --train-dir $data_dir/train/ --test-dir $data_dir/dev -N $n --rare $katz_rare --filter-punct --train-mode katz --features $feat_dir/features_dev_katz${n}
done
