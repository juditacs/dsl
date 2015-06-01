# dsl
 Discriminating between Similar Languages

    python dsl/tfidf.py --train dat/confused/all_slavic/train/ --test dat/confused/all_slavic/dev/ --topn 100000 --tokenize word --tf lognorm --idf smooth --rare 0 --N 3 --qf smooth > out

Download this year's training and development datasets and split them to one file-per-language:

    bash scripts/download_and_split.sh dat/
