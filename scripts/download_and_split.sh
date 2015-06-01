#!/usr/bin/env bash

wget https://db.tt/CGvtrCK3
unzip DSLCC-v2.zip
mkdir train dev
langs=( bg  bs  cz  es-AR  es-ES  hr  id  mk  my  pt-BR  pt-PT  sk  sr  xx )

for lang in "${langs[@]}"; do
    grep ${lang}$ DSLCC-v2.1/train.txt | cut -f1 > train/${lang}
    grep ${lang}$ DSLCC-v2.1/devel.txt | cut -f1 > dev/${lang}
done
