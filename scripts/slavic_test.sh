#!/usr/bin/env bash

feat_basedir=sandbox/slavic_pairwise/feat
data_basedir=dat/confused/slavic_pairwise

# setting up tfidf features
cat $feat_basedir/features_slavic_train_katz4 | grep -v ^2 > $feat_basedir/bypair/bs_hr_katz4_train.feat
cat $feat_basedir/features_slavic_train_katz3 | grep -v ^2 > $feat_basedir/bypair/bs_hr_katz3_train.feat

cat $feat_basedir/features_slavic_train_katz4 | grep -v ^1 > $feat_basedir/bypair/bs_sr_katz4_train.feat
cat $feat_basedir/features_slavic_train_katz3 | grep -v ^1 > $feat_basedir/bypair/bs_sr_katz3_train.feat

cat $feat_basedir/features_slavic_train_katz4 | grep -v ^0 > $feat_basedir/bypair/hr_sr_katz4_train.feat
cat $feat_basedir/features_slavic_train_katz3 | grep -v ^0 > $feat_basedir/bypair/hr_sr_katz3_train.feat

cat $feat_basedir/features_slavic_dev_katz4 | grep -v ^2 > $feat_basedir/bypair/bs_hr_katz4_dev.feat
cat $feat_basedir/features_slavic_dev_katz3 | grep -v ^2 > $feat_basedir/bypair/bs_hr_katz3_dev.feat

cat $feat_basedir/features_slavic_dev_katz4 | grep -v ^1 > $feat_basedir/bypair/bs_sr_katz4_dev.feat
cat $feat_basedir/features_slavic_dev_katz3 | grep -v ^1 > $feat_basedir/bypair/bs_sr_katz3_dev.feat

cat $feat_basedir/features_slavic_dev_katz4 | grep -v ^0 > $feat_basedir/bypair/hr_sr_katz4_dev.feat
cat $feat_basedir/features_slavic_dev_katz3 | grep -v ^0 > $feat_basedir/bypair/hr_sr_katz3_dev.feat

langs=( bs_hr bs_sr hr_sr )
pearson_th=( 5000 1000 100 )
pearson_n=( 1 2 3 )
katz=( 3 4 )

for pair in ${langs[@]}; do
    echo $pair
    #tfidf
    if [ -f $feat_basedir/bypair/${pair}_train_tfidf.feat ]; then
        echo "file exists: $feat_basedir/bypair/${pair}_train_tfidf.feat"
    else
        python dsl/features/tfidf.py --train $data_basedir/${pair}/train/ --test $data_basedir/${pair}/train/ --topn 100000 --tokenize word --tf lognorm --idf smooth --rare 0 | grep ^[0-9] > $feat_basedir/bypair/${pair}_train_tfidf.feat
    fi
    if [ -f $feat_basedir/bypair/${pair}_dev_tfidf.feat ]; then
        echo "file exists: $feat_basedir/bypair/${pair}_dev_tfidf.feat"
    else
        python dsl/features/tfidf.py --train $data_basedir/${pair}/train/ --test $data_basedir/${pair}/dev/ --topn 100000 --tokenize word --tf lognorm --idf smooth --rare 0 | grep ^[0-9] > $feat_basedir/bypair/${pair}_dev_tfidf.feat
    fi

    for pearson in ${pearson_th[@]}; do
        for n in ${pearson_n[@]}; do
            if [ -f $feat_basedir/bypair/${pair}_n${n}_top${pearson}_dev.feat ]; then
                echo "file exists: $feat_basedir/bypair/${pair}_n${n}_top${pearson}_dev.feat"
            else
                python dsl/features/featurize_pearson.py -N ${n} $data_basedir/${pair}/train/ $data_basedir/${pair}/dev/ -t 5 --train-out $feat_basedir/bypair/${pair}_n${n}_top${pearson}_train.feat --test-out $feat_basedir/bypair/${pair}_n${n}_top${pearson}_dev.feat --topn ${pearson}
            fi
            for k in ${katz[@]}; do
                if [ -f $feat_basedir/matrix/${pair}_katz${k}_pearson${n}_top${pearson}_train.matrix ]; then
                    echo "exists"
                else
                    paste -d" " $feat_basedir/bypair/${pair}_katz${k}_train.feat <(cut -f2- -d" " $feat_basedir/bypair/${pair}_train_tfidf.feat) $feat_basedir/bypair/${pair}_n${n}_top${pearson}_train.feat > $feat_basedir/matrix/${pair}_katz${k}_pearson${n}_top${pearson}_train.matrix
                fi

                if [ -f $feat_basedir/matrix/${pair}_katz${k}_pearson${n}_top${pearson}_dev.matrix ]; then
                    echo "exists"
                else
                    paste -d" " $feat_basedir/bypair/${pair}_katz${k}_dev.feat <(cut -f2- -d" " $feat_basedir/bypair/${pair}_dev_tfidf.feat) $feat_basedir/bypair/${pair}_n${n}_top${pearson}_dev.feat > $feat_basedir/matrix/${pair}_katz${k}_pearson${n}_top${pearson}_dev.matrix
                fi
            done
        done
    done
done
