import nltk
from nltk.corpus import cess_esp
from nltk import UnigramTagger as ut
from pickle import dump

def main():
    
    # Read the corpus into a list, 
    # each entry in the list is one sentence.
    cess_sents = cess_esp.tagged_sents()

    # Train the unigram tagger
    uni_tag = ut(cess_sents)
    
    output = open('uni_tag.pkl', 'wb')
    dump(uni_tag, output, -1)
    output.close()

if __name__ == "__main__":
    main()
