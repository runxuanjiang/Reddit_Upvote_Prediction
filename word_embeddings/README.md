# Instructions for downloading the word vectors:

I used GLOVE embeddings, the website is [here](https://nlp.stanford.edu/projects/glove/).

For the example used in `word_embeddings.ipynb`, download the zip file [here](https://nlp.stanford.edu/data/glove.6B.zip) or run

```
    wget https://nlp.stanford.edu/data/glove.6B.zip
```

unzip the file and all the word vectors files are in there.

Each line corresponds to one word, with the first string being the word itself, and the remaining strings (per line) being the vector. All the vectors are the same length (within each file).