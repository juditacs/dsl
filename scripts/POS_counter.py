import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import cess_esp
from nltk import UnigramTagger as ut
from pickle import load
from collections import Counter

def main():
    f = open("es-ES")
    cnt = Counter()
    #load the training tagger
    input = open('uni_tag.pkl', 'rb')
    uni_tag = load(input)
    input.close()
    
    cnt = Counter()
    for line in f:
        sentence = word_tokenize(line.decode("utf8"))
        for (x,y) in uni_tag.tag(sentence):
            cnt[y] += 1
    for (a,b) in sorted(cnt.iteritems(), key = lambda x: -x[1]):
        print a, "\t", b

if __name__ == "__main__":
    main()
