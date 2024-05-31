"""
Student ID: 21207500
Student Name: Liyan Tao

query_large_corpus.py

该程序允许用户提交查询以从小型语料库中检索内容，或运行标准语料库查询以评估系统。必须使用BM25模型进行检索。
每次运行此程序时，应首先将索引加载到内存中（在当前工作目录中名为“21207500-large.index”），以便查询尽可能快。

该程序应根据名为“-m”的命令行参数提供两种模式。这些模式如下：

1. 交互模式
在此模式下，用户可以手动输入查询，并在命令行中看到前15个结果，按相似度得分从高到低排序。
输出应包含三列：排名、文档ID和相似度得分。
程序运行示例如下所示:
用户应继续被提示输入进一步的查询，直到他们输入“QUIT”。

2. 自动模式
在此模式下，应从语料库的“files”目录中的“queries.txt”文件读取标准查询。
该文件每行有一个查询，从其查询ID开始。
结果应存储在当前工作目录中的名为“21207500-large.results”的文件中（将“21207500”替换为你的UCD学生号码），
该文件应包含四列：查询ID、文档ID、排名和相似度得分。所需输出的示例可以在语料库的“files”目录中的“sample_output.txt”文件中找到。

通过以下方式运行程序可以激活自动模式：
./query_large_corpus.py -m automatic -p /path/to/comp3009j-corpus-large
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
        print(f"Loading index file...")
        start = time.time()
        with open(index_file_path, 'r', encoding='utf-8') as file:
            index = json.load(file)
        end = time.time()
        print(f"Index loaded in {end - start:.4f} seconds.")
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
        """执行查询并返回排序后的文档列表。假设使用简单的BM25得分排序。"""
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
        """交互模式：用户输入查询并看到结果。"""
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
                print(f"{'Rank':<10}{'Doc ID':<25}{'Score'}")
                for rank, (doc_id, score) in enumerate(results, start=1):
                    print(f"{rank:<10}{doc_id:<25}{score:.4f}")
            else:
                print("No results found.")

    def automatic_mode(self, queries_file, output_file):
        """自动模式：从文件读取查询并将结果写入另一个文件。"""
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
    parser.add_argument('-p', '--path', type=str, required=True, help="Path to the corpus")
    args = parser.parse_args()

    index_file_path = os.path.join(os.getcwd(), "21207500-large.index.json")
    stopwords_file_path = os.path.join(args.path, "files", "stopwords.txt")
    if not os.path.exists(stopwords_file_path):  # 如果stopwords_file_path中的文件不存在,则使用默认的stopwords.txt文件
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
