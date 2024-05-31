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
        """加载索引文件到内存中。"""
        with open(index_file_path, 'r', encoding='utf-8') as file:
            index = json.load(file)
        return index

    def load_stopwords(self, stopwords_file):
        """从文件中加载停用词。"""
        with open(stopwords_file, 'r', encoding='utf-8') as file:
            return set(line.strip() for line in file)

    def process_query(self, query):
        """处理查询：转换为小写，移除标点，去除停用词，执行词干提取。"""
        query = query.translate(str.maketrans('', '', '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'))  # 移除标点
        terms = query.lower().split()  # 转换为小写并分割
        terms = [term for term in terms if term not in self.stopwords]  # 去除停用词
        terms = [self.stemmer.stem(term) for term in terms]  # 执行词干提取
        return terms

    def perform_query(self, query):
        """执行查询并返回排序后的结果"""
        query_terms = self.process_query(query)
        results = {}
        for doc_id, doc_terms in self.index.items():
            score = sum(doc_terms.get(term, 0) for term in query_terms)
            if score > 0:
                results[doc_id] = score
        # 结果按得分降序排序, 并返回前15个结果, 如果不足15个则返回全部结果
        sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)[:15]
        return sorted_results

    def interactive_mode(self):
        """交互模式：输入查询并打印结果。"""
        while True:
            query = input("Enter your query (or 'QUIT' to exit): ")
            if query.upper() == 'QUIT':
                break
            start_time = time.time()  # 查询开始时间
            results = self.perform_query(query)
            end_time = time.time()  # 查询结束时间
            duration = end_time - start_time  # 计算时间差
            print(f"Query completed in {duration:.4f} seconds.")  # 打印查询用时
            if results:
                print(f"{'Rank':<5}{'Doc ID':<10}{'Score':<10}")
                for rank, (doc_id, score) in enumerate(results, start=1):
                    print(f"{rank:<5}{doc_id:<10}{score:<10.4f}")
            else:
                print("No results found.")

    def automatic_mode(self, queries_file, output_file):
        """自动模式：从文件读取查询并将结果写入到脚本的同级目录下"""
        with open(queries_file, 'r', encoding='utf-8') as qfile, \
                open(output_file, 'w', encoding='utf-8') as ofile:
            total_time = 0
            for line in qfile:
                query_id, query = line.strip().split(' ', 1)
                start_time = time.time()
                results = self.perform_query(query)
                end_time = time.time()
                duration = end_time - start_time
                total_time += duration
                for rank, (doc_id, score) in enumerate(results, start=1):
                    ofile.write(f"{query_id} {doc_id} {rank} {score:.4f}\n")
            print(f"Queries completed in {total_time:.4f} seconds.")


def main():
    parser = argparse.ArgumentParser(description="Query and retrieve documents.")
    parser.add_argument('-m', '--mode', type=str, choices=['interactive', 'automatic'], required=True,
                        help="Mode of operation")
    parser.add_argument('-p', '--path', type=str, required=True, help="Path to the small corpus")
    args = parser.parse_args()

    index_file_path = os.path.join(os.getcwd(), "21207500-small.index.json")
    stopwords_file_path = os.path.join(args.path, "files", "stopwords.txt")
    if not os.path.exists(stopwords_file_path):  # 如果stopwords_file_path中的文件不存在,则使用项目内的的stopwords.txt文件
        print(f"Stopwords file not found at {stopwords_file_path}. Using project's stopwords file.")
        stopwords_file_path = os.path.join(os.getcwd(), "files", "stopwords.txt")
    bm25 = BM25(index_file_path, stopwords_file_path)

    if args.mode == 'interactive':
        bm25.interactive_mode()
    elif args.mode == 'automatic':
        queries_file = os.path.join(args.path, "files", "queries.txt")
        output_file = os.path.join(os.getcwd(), "21207500-small.results")
        bm25.automatic_mode(queries_file, output_file)


if __name__ == "__main__":
    main()
