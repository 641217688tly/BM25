"""
Student ID: 21207500
Student Name: Liyan Tao
"""

import os
import json
import argparse
from files import porter
import time


class BM25:
    def __init__(self, index_file_path, stopwords_file_path):
        self.index = self.load_index(index_file_path)
        self.stopwords = self.load_stopwords(stopwords_file_path)
        self.stemmer = porter.PorterStemmer()

    def load_index(self, index_file_path):
        """Load the index file into memory."""
        print(f"Loading index file...")
        start = time.time()
        with open(index_file_path, 'r', encoding='utf-8') as file:
            index = json.load(file)
        end = time.time()
        print(f"Index loaded in {end - start:.4f} seconds.")
        return index

    def load_stopwords(self, stopwords_file):
        """Load stopwords from file."""
        with open(stopwords_file, 'r', encoding='utf-8') as file:
            return set(line.strip() for line in file)

    def process_query(self, query):
        """Process the query: convert to lowercase, remove punctuation, remove stopwords, and perform stemming."""
        query = query.translate(str.maketrans('', '', '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'))  # Remove punctuation
        terms = query.lower().split()  # Convert to lowercase and split
        terms = [term for term in terms if term not in self.stopwords]  # Remove stopwords
        terms = [self.stemmer.stem(term) for term in terms]  # Perform stemming
        return terms

    def perform_query(self, query):
        """Perform the query and return sorted results."""
        query_terms = self.process_query(query)
        results = {}
        for doc_id, doc_terms in self.index.items():
            score = sum(doc_terms.get(term, 0) for term in query_terms)
            if score > 0:
                results[doc_id] = score
        # Sort results by score in descending order and return the top 15 results, or all results if fewer than 15
        sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)[:15]
        return sorted_results

    def interactive_mode(self):
        """Interactive mode: input queries and print results."""
        while True:
            query = input("Enter your query (or 'QUIT' to exit): ")
            if query.upper() == 'QUIT':
                break
            start_time = time.time()  # Query start time
            results = self.perform_query(query)
            end_time = time.time()  # Query end time
            duration = end_time - start_time  # Calculate time difference
            print(f"Query completed in {duration:.4f} seconds.")  # Print query time
            if results:
                print(f"{'Rank':<10}{'Doc ID':<25}{'Score'}")
                for rank, (doc_id, score) in enumerate(results, start=1):
                    print(f"{rank:<10}{doc_id:<25}{score:.4f}")
            else:
                print("No results found.")

    def automatic_mode(self, queries_file, output_file):
        """Automatic mode: read queries from file and write results to the same directory as the script."""
        with open(queries_file, 'r', encoding='utf-8') as qfile, \
                open(output_file, 'w', encoding='utf-8') as ofile:
            total_time = 0
            for line in qfile:
                query_id, query = line.strip().split(' ', 1)
                start = time.time()
                results = self.perform_query(query)
                end = time.time()
                total_time += end - start
                for rank, (doc_id, score) in enumerate(results, start=1):
                    ofile.write(f"{query_id} {doc_id} {rank} {score:.4f}\n")
            print(f"Queries completed in {total_time:.4f} seconds.")


def main():
    parser = argparse.ArgumentParser(description="Query and retrieve documents.")
    parser.add_argument('-m', '--mode', type=str, choices=['interactive', 'automatic'], required=True,
                        help="Mode of operation")
    parser.add_argument('-p', '--path', type=str, required=True, help="Path to the large corpus")
    args = parser.parse_args()

    index_file_path = os.path.join(os.getcwd(), "21207500-large.index.json")
    stopwords_file_path = os.path.join(args.path, "files", "stopwords.txt")
    if not os.path.exists(stopwords_file_path):  # If the file in stopwords_file_path does not exist, use the project's stopwords.txt file
        print(f"Stopwords file not found at {stopwords_file_path}. Using project's stopwords file.")
        stopwords_file_path = os.path.join(os.getcwd(), "files", "stopwords.txt")

    bm25 = BM25(index_file_path, stopwords_file_path)

    if args.mode == 'interactive':
        bm25.interactive_mode()
    elif args.mode == 'automatic':
        queries_file = os.path.join(args.path, "files", "queries.txt")
        output_file = os.path.join(os.getcwd(), "21207500-large.results")
        bm25.automatic_mode(queries_file, output_file)


if __name__ == "__main__":
    main()
