#!/usr/bin/env bash

# creating ngram features
./scripts/create_chngram_matrix.sh dat/train_dev/noxx/train/ final/eval_dat/test_dir/ final/features/langid/

# global tfidf to distinguish language groups, should be 100%
./scripts/first_level.sh dat/train_dev/full/train/ final/eval_dat/test_dir/ final/closed/v1/tfidf/

# tfidf in each category
./scripts/subgroup_tfidf.sh /home/judit/projects/dsl/dat/train_dev/groups/train/ final/closed/v1/tfidf/dat/groups/ final/closed/v1/level2/groups/

./scripts/word_level.sh /home/judit/projects/dsl/dat/train_dev/groups/train/ final/closed/v1/tfidf/dat/groups/ final/closed/v1/level2/groups/ 500

./scripts/merge_features_traindev.sh dat/train_dev/groups/train/ final/closed/v1/tfidf/dat/groups/ final/closed/v1/level2/groups/ 200
