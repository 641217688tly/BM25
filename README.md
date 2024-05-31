# COMP3009J Information Retrieval

## Author
Student Number: 21207500

Student Name: Liyan Tao

## Notification
Here are the commands to help you quickly run my scripts.
I assume that you have placed the small corpus and large corpus folders at the same level as the root directory of this project. If the file paths of your corpora are different, please modify the path parameters in the commands.
Before using the BM25 model with either the small corpus or the large corpus, you need to navigate to the small_corpus_handler or large_corpus_handler folder in the command line. Only then can you correctly run my scripts.
## Small Corpus Commands

**Enter the small_corpus_handler folder:**

```cmd
cd .\small_corpus_handler\
```

**index_small_corpus.py:**

```cmd
python .\index_small_corpus.py -p "../../comp3009j-corpus-small"
```

**query_small_corpus.py:**

```cmd
python .\query_small_corpus.py -m interactive -p "../../comp3009j-corpus-small"
```

```cmd
python .\query_small_corpus.py -m automatic -p "../../comp3009j-corpus-small"
```

**evaluate_small_corpus.py:**

```cmd
python .\evaluate_small_corpus.py -p "../../comp3009j-corpus-small"
```


## Large Corpus Commands

**Enter the large_corpus_handler folder:**

```cmd
cd .\large_corpus_handler\
```

**index_large_corpus.py:**

```cmd
python .\index_large_corpus.py -p "../../comp3009j-corpus-large"
```

**query_large_corpus.py:**

```cmd
python .\query_large_corpus.py -m interactive -p "../../comp3009j-corpus-large"
```

```cmd
python .\query_large_corpus.py -m automatic -p "../../comp3009j-corpus-large"
```

**evaluate_large_corpus.py:**

```cmd
python .\evaluate_large_corpus.py -p "../../comp3009j-corpus-large"
```
