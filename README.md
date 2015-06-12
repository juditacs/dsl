# dsl
 Discriminating between Similar Languages

    python dsl/tfidf.py --train dat/confused/all_slavic/train/ --test dat/confused/all_slavic/dev/ --topn 100000 --tokenize word --tf lognorm --idf smooth --rare 0 --N 3 --qf smooth > out

Or an instant candy

    python dsl/tfidf.py --train dat/train/ --test dat/dev/ --topn 10000 --tokenize word --tf lognorm --idf smooth --rare 0 --N 3 --qf smooth > out

Download this year's training and development datasets and split them to one file-per-language:

    bash scripts/download_and_split.sh dat/

## Running an experiment

### Prequisites

1. The newest version of the repo obviously
2. Train and dev features files in the format of space separated feature matrix, where the first column contains the labels and the rest of them the feature values.
3. A directory where the language names can be looked up. The default is `dat/t1/train`, this presumes that you are running an experiment on all 14 languages. If you're running on fewer languages, then please specify a directory where for each language, a file named as the language exists. These names will be used in the classifier's output instead of integer labels.

### Parameters

1. --train: train file, space-separated matrix, where the first column contains the labels
2. --test: same as train. Unlabeled test files are not supported right now
3. --encoder: name of the encoder
4. --classifier: name of the classifier
5. params: positional arguments, i.e. anything without a --, these arguments split by '=' and passed to the Representation constructor as keyword arguments. For example pca\_latentDimension=50 is converted to a parameter, where the key ia pca\_latentDimension and its value is 50.

    python dsl/representation/run_experiment.py --train train_matrix --test dev_matrix --lang-map dat/t1/train --encoder pca --class svm svm_ktype=linearsvc pca_latentDimension=50 > out
