# dsl
 Discriminating between Similar Languages

    python dsl/tfidf.py --train dat/confused/all_slavic/train/ --test dat/confused/all_slavic/dev/ --topn 100000 --tokenize word --tf lognorm --idf smooth --rare 0 --N 3 --qf smooth > out
