"""
Student ID: 21207500
Student Name: Liyan Tao
"""

import os
import math
import json
import time
from files import porter
import argparse


class DocumentProcessor:
    def __init__(self, documents_dir_path, stopwords_file_path='large_corpus_handler/files/stopwords.txt'):
        self.documents_dir_path = documents_dir_path
        self.stopwords_file_path = stopwords_file_path
        self.documents = {}
        self.stopwords = set()
        self.stemmer = porter.PorterStemmer()
        self.stemmer_accelerator = {}
        self.docs_num = 0  # Total number of documents
        self.total_doc_len = 0  # Total document length for calculating average length
        self.avg_doc_len = 0  # Average document length
        self.load_stopwords()
        self.process_documents()

    def count_valid_files_number(self, directory):
        """Count the number of files starting with 'GX' in the directory and its subdirectories"""
        total_files = 0
        for subdir, dirs, files in os.walk(directory):
            for file in files:
                if file.startswith("GX"):
                    total_files += 1
        return total_files

    def load_stopwords(self):
        """Load stopwords list from file"""
        with open(self.stopwords_file_path, 'r', encoding='UTF-8') as file:
            for line in file:
                self.stopwords.add(line.strip())

    def process_documents(self):
        """Execute the complete document processing workflow: reading, removing stopwords, and stemming"""
        print("Start documents preprocessing, please wait...")
        start = time.time()
        total_files = self.count_valid_files_number(self.documents_dir_path)
        for subdir in os.listdir(self.documents_dir_path):  # Traverse all subdirectories in the documents directory
            subdir_path = os.path.join(self.documents_dir_path, subdir)
            if os.path.isdir(subdir_path):
                for filename in os.listdir(subdir_path):  # Traverse files in subdirectories (e.g., GX000, GX001...)
                    if filename.startswith("GX"):  # Ensure the file starts with 'GX'
                        file_path = os.path.join(subdir_path, filename)
                        with open(file_path, 'r', encoding='UTF-8') as file:
                            text = file.read().lower()
                            text = text.translate(str.maketrans('', '', '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'))  # Remove punctuation
                            tokens = text.split()
                            tokens = self.remove_stopwords(tokens)  # Remove stopwords
                            tokens = self.stem_words(tokens)  # Perform stemming
                            self.documents[filename] = tokens
                            self.docs_num += 1  # Update the total number of documents
                            self.total_doc_len += len(tokens)  # Accumulate the length of the current document
                        if self.docs_num % 100 == 0 or self.docs_num == total_files:
                            print(f"Processing documents {self.docs_num}/{total_files}")
        # After processing all documents, calculate the average document length
        if self.docs_num > 0:
            self.avg_doc_len = self.total_doc_len / self.docs_num
        end = time.time()
        print(f"Document processing completed in {end - start:.2f} seconds.\n")

    def remove_stopwords(self, tokens):
        """Remove stopwords from the given list of terms."""
        return [term for term in tokens if term not in self.stopwords]

    def stem_words(self, tokens):
        """Perform stemming on the given list of terms. Use a stemmer accelerator to speed up the stemming process."""
        stemmed_tokens = []
        for term in tokens:
            if term in self.stemmer_accelerator:
                stemmed_tokens.append(self.stemmer_accelerator[term])
            else:
                stemmed_term = self.stemmer.stem(term)
                self.stemmer_accelerator[term] = stemmed_term
                stemmed_tokens.append(stemmed_term)
        return stemmed_tokens


class BM25Index:
    def __init__(self, document_processor, k=1, b=0.75):
        self.idf = {}
        self.tf = {}
        self.bm25_scores = {}
        self.processor = document_processor
        self.k = k
        self.b = b
        self.get_document_score()

    def compute_terms_idf(self):
        """Compute the IDF of each term, idf_i = log2( (N-n_i+0.5) / (n_i+0.5) + 1) : N is the total number of documents, n_i is the number of documents containing term i"""
        print("Computing terms IDF values...")
        documents = self.processor.documents
        idf = {}  # Store the number of documents each term appears in the document collection
        for i, doc in enumerate(documents):  # doc is the key of the document
            for term in set(documents[doc]):
                if term not in idf:
                    idf[term] = 1
                else:
                    idf[term] += 1
            if (i + 1) % 1000 == 0 or i + 1 == self.processor.docs_num:
                print(f"Computing terms IDF values for documents {i + 1}/{self.processor.docs_num}")
        for term in idf:
            idf[term] = math.log2(1 + (self.processor.docs_num - idf[term] + 0.5) / (idf[term] + 0.5))
        return idf

    def compute_documents_tf(self):
        """Compute the TF value of each term in the document collection"""
        print("Computing documents TF values...")
        documents = self.processor.documents
        tf = {}  # Store the TF value of each term in each document: {doc1: {term1: tf1, term2: tf2, ...}, doc2: {term1: tf1, term2: tf2, ...}, ...}
        for i, doc in enumerate(documents):
            tf[doc] = {}  # tf = { doc1: { term1 : num, term2 : num, ... }, doc2: { term1 : num, term2 : num, ... }, ... }
            for term in documents[doc]:  # Traverse all terms in the document with key doc
                if term not in tf[doc]:
                    tf[doc][term] = 1
                else:
                    tf[doc][term] += 1
            # Calculate the TF value of each term in the current document
            for term in tf[doc]:
                f_ij = tf[doc][term]
                tf[doc][term] = (f_ij * (1 + self.k)) / (f_ij + self.k * (1 - self.b + self.b * len(documents[doc]) / self.processor.avg_doc_len))
            if (i + 1) % 1000 == 0 or i + 1 == self.processor.docs_num:
                print(f"Computing documents TF values for documents {i + 1}/{self.processor.docs_num}")
        # print("Documents TF values: " + str(tf) + "\n")
        # The TF of each document in the document collection after calculation is as follows:
        # {'GX000-01-10544170': {'static': 1.0586916337295802, 'aerodynam': 1.542763787034428, 'characterist': 1.542763787034428, 'short': 1.3845026050418516, 'blunt': 1.3845026050418516, 'cone': 1.799952348010274, 'variou': 1.0586916337295802, 'nose': 1.7418759813234317, 'base': 1.820181404463512, 'angl': 1.799952348010274, 'mach': 1.3845026050418516, 'number': 1.542763787034428, '0': 1.3845026050418516, '6': 1.0586916337295802, '5': 1.3845026050418516, 'attack': 1.3845026050418516, '180': 1.542763787034428, 'windtunnel': 1.0586916337295802, 'test': 1.542763787034428, 'perform': 1.0586916337295802, '06': 1.0586916337295802, '55': 1.0586916337295802, 'determin': 1.0586916337295802, 'coeffici': 1.0586916337295802, 'normal': 1.0586916337295802, 'forc': 1.3845026050418516, 'axial': 1.0586916337295802, 'pitch': 1.0586916337295802, 'moment': 1.0586916337295802, 'affect': 1.0586916337295802, 'chang': 1.542763787034428, 'model': 1.6980453523003483, 'halfangl': 1.7745949513495383, '10': 1.3845026050418516, '20': 1.3845026050418516, 'investig': 1.0586916337295802, 'flat': 1.3845026050418516, '50': 1.542763787034428, '70': 1.3845026050418516, 'reynold': 1.0586916337295802, 'rang': 1.0586916337295802, 'maximum': 1.0586916337295802, 'diamet': 1.0586916337295802, 'variat': 1.3845026050418516, 'result': 1.542763787034428, 'signific': 1.0586916337295802, 'lesser': 1.0586916337295802, 'effect': 1.0586916337295802, 'particular': 1.0586916337295802, 'conic': 1.3845026050418516, 'on': 1.0586916337295802, 'trim': 1.542763787034428, 'two': 1.0586916337295802, 'estim': 1.0586916337295802, 'mean': 1.0586916337295802, 'modifi': 1.0586916337295802, 'newtonian': 1.0586916337295802, 'theori': 1.3845026050418516, 'good': 1.0586916337295802, 'agreement': 1.0586916337295802, 'experiment': 1.0586916337295802, 'fail': 1.0586916337295802, 'predict': 1.0586916337295802, 'point': 1.0586916337295802, 'flatbas': 1.0586916337295802}, ...}
        return tf

    def compute_bm25_scores(self):
        """Compute the BM25 score of each document in the document collection"""
        print("Computing BM25 scores...")
        documents = self.processor.documents
        scores = {}
        for i, doc in enumerate(documents):
            # scores = { doc1: { term1 : score, term2 : score, ... }, doc2: { term1 : score, term2 : score, ... }, ... }
            scores[doc] = {}
            for term in documents[doc]:
                if term not in self.idf:
                    scores[doc][term] = 0
                else:
                    scores[doc][term] = self.idf[term] * self.tf[doc][term]
            if (i + 1) % 1000 == 0 or i + 1 == self.processor.docs_num:
                print(f"Computing BM25 scores for documents {i + 1}/{self.processor.docs_num}")
        # print("Documents BM25 scores: " + str(scores) + "\n")
        return scores

    def get_document_score(self):
        """First compute TF and IDF, then compute BM25 scores of each document"""
        print("Start computing BM25 scores, please wait...")
        start = time.time()
        self.idf = self.compute_terms_idf()
        self.tf = self.compute_documents_tf()
        self.bm25_scores = self.compute_bm25_scores()
        end = time.time()
        print(f"BM25 scores computed in {end - start:.2f} seconds.\n")

    def export_to_json(self, output_dir):
        """Export BM25 scores to a JSON file"""
        print("Exporting BM25 scores to JSON file...")
        file_path = os.path.join(output_dir, '21207500-large.index.json')
        if self.bm25_scores.keys():
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.bm25_scores, f, ensure_ascii=False, indent=4)
        else:
            print("No BM25 scores to export.")


def main():
    parser = argparse.ArgumentParser(description="Process and index documents.")
    parser.add_argument('-p', '--path', type=str, required=True, help="Path to the large corpus")
    args = parser.parse_args()

    documents_path = os.path.join(args.path, "documents")
    stopwords_path = os.path.join(args.path, "files", "stopwords.txt")

    start = time.time()
    index = BM25Index(DocumentProcessor(documents_path, stopwords_path))
    index.export_to_json(os.getcwd())
    end = time.time()
    print(f"Indexing completed in {end - start:.2f} seconds.")


if __name__ == "__main__":
    main()
