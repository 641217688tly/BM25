"""
index_small_corpus.py

该程序旨在读取小型语料库, 处理其内容并创建索引。
必须可以将小型语料库的路径作为命令行参数“-p”传递给该程序：
./index_small_corpus.py -p /path/to/comp3009j-corpus-small

该程序必须执行以下任务：

1. 提取所提供语料库中的文档。你必须以适当的方式将文档划分为术语（这些术语包含在语料库的“documents”目录中）。该策略必须在源代码注释中记录。
2. 执行停用词删除。要使用的停用词列表可以从语料库的“files”目录中提供的stopwords.txt文件加载。
3. 执行词干提取。对于此任务，你可以使用“files”目录中的porter.py代码。
4. 创建适当的索引，以便可以使用BM25方法进行信息检索。这里的索引是任何适合以后执行检索的数据结构。

这将要求你计算适当的权重并进行尽可能多的预计算。应将其存储在一个单一的外部文件中，以某种人类可读的格式存储。
不要使用数据库系统（例如MySQL、SQL Server、SQLite等）。
该程序的输出应为一个单一的索引文件，存储在当前工作目录中，文件名为“21207500-small.index”（将“21207500”替换为你的UCD学生号码）。
"""
import os
import math
import json
from files import porter
import argparse


class DocumentProcessor:
    def __init__(self, documents_dir_path, stopwords_file_path='small_corpus_handler/files/stopwords.txt'):
        self.documents_dir_path = documents_dir_path
        self.stopwords_file_path = stopwords_file_path
        self.documents = {}
        self.stopwords = set()
        self.docs_num = 0  # 文档总数
        self.total_doc_len = 0  # 文档总长度，用于计算平均长度
        self.avg_doc_len = 0  # 平均文档长度
        self.process_documents()

    def read_documents(self):
        """读取文档集合并做简单的处理：转换为小写、移除标点、分割为术语。"""
        for filename in os.listdir(self.documents_dir_path):
            file_path = os.path.join(self.documents_dir_path, filename)
            with open(file_path, 'r', encoding='UTF-8') as file:
                text = file.read().lower()
                text = text.translate(str.maketrans('', '', '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'))
                tokens = text.split()
                self.documents[filename] = tokens
                self.docs_num += 1  # 更新文档总数
                self.total_doc_len += len(tokens)  # 累加当前文档的长度
        # 所有文档处理完毕后计算平均文档长度
        if self.docs_num > 0:
            self.avg_doc_len = self.total_doc_len / self.docs_num

    def load_stopwords(self):
        """从文件加载停用词列表。"""
        with open(self.stopwords_file_path, 'r', encoding='UTF-8') as file:
            for line in file:
                self.stopwords.add(line.strip())

    def remove_stopwords(self):
        """从文档中移除停用词。"""
        for doc in self.documents:
            self.documents[doc] = [term for term in self.documents[doc] if term not in self.stopwords]

    def stem_words(self):
        """对文档中的术语进行词干提取。"""
        stemmer = porter.PorterStemmer()
        for doc in self.documents:
            self.documents[doc] = [stemmer.stem(term) for term in self.documents[doc]]

    def process_documents(self):
        """执行完整的文档处理流程：读取、去停用词、词干提取。"""
        self.read_documents()
        self.load_stopwords()
        self.remove_stopwords()
        self.stem_words()


class BM25Index:
    def __init__(self, document_processor, k=1, b=0.75):
        self.idf = {}
        self.tf = {}
        self.bm25_scores = {}
        self.processor = document_processor
        self.k = k
        self.b = b
        self.get_document_score()

    def compute_terms_idf(self):  # 计算每个Term的IDF, idf_i = log2( (N-n_i+0.5) / (n_i+0.5) + 1) : N为文档总数, n_i为包含词i的文档数
        documents = self.processor.documents
        idf = {}  # 存储每个term在文档集合中出现在了几个文档中
        for doc in documents:  # doc为文档的key
            for term in set(documents[doc]):
                if term not in idf:
                    idf[term] = 1
                else:
                    idf[term] += 1
        for term in idf:
            idf[term] = math.log2(1 + (self.processor.docs_num - idf[term] + 0.5) / (idf[term] + 0.5))
        return idf

    def compute_documents_tf(self):  # 计算文档集合内各个文档中Term的TF值
        documents = self.processor.documents
        tf = {}  # 存储每个文档中每个term的TF值: {doc1: {term1: tf1, term2: tf2, ...}, doc2: {term1: tf1, term2: tf2, ...}, ...}
        for doc in documents:
            tf[
                doc] = {}  # tf = { doc1: { term1 : num, term2 : num, ... }, doc2: { term1 : num, term2 : num, ... }, ... }
            for term in documents[doc]:  # 遍历key为doc的文档的所有term
                if term not in tf[doc]:
                    tf[doc][term] = 1
                else:
                    tf[doc][term] += 1
            # 计算当前文档每个term的TF值
            for term in tf[doc]:
                f_ij = tf[doc][term]
                tf[doc][term] = (f_ij * (1 + self.k)) / (
                        f_ij + self.k * (1 - self.b + self.b * len(documents[doc]) / self.processor.avg_doc_len))
        # print("Documents TF values: " + str(tf) + "\n")
        # 计算后的文档集合中的各个文档的TF的格式如下:
        # {'999': {'static': 1.0586916337295802, 'aerodynam': 1.542763787034428, 'characterist': 1.542763787034428, 'short': 1.3845026050418516, 'blunt': 1.3845026050418516, 'cone': 1.799952348010274, 'variou': 1.0586916337295802, 'nose': 1.7418759813234317, 'base': 1.820181404463512, 'angl': 1.799952348010274, 'mach': 1.3845026050418516, 'number': 1.542763787034428, '0': 1.3845026050418516, '6': 1.0586916337295802, '5': 1.3845026050418516, 'attack': 1.3845026050418516, '180': 1.542763787034428, 'windtunnel': 1.0586916337295802, 'test': 1.542763787034428, 'perform': 1.0586916337295802, '06': 1.0586916337295802, '55': 1.0586916337295802, 'determin': 1.0586916337295802, 'coeffici': 1.0586916337295802, 'normal': 1.0586916337295802, 'forc': 1.3845026050418516, 'axial': 1.0586916337295802, 'pitch': 1.0586916337295802, 'moment': 1.0586916337295802, 'affect': 1.0586916337295802, 'chang': 1.542763787034428, 'model': 1.6980453523003483, 'halfangl': 1.7745949513495383, '10': 1.3845026050418516, '20': 1.3845026050418516, 'investig': 1.0586916337295802, 'flat': 1.3845026050418516, '50': 1.542763787034428, '70': 1.3845026050418516, 'reynold': 1.0586916337295802, 'rang': 1.0586916337295802, 'maximum': 1.0586916337295802, 'diamet': 1.0586916337295802, 'variat': 1.3845026050418516, 'result': 1.542763787034428, 'signific': 1.0586916337295802, 'lesser': 1.0586916337295802, 'effect': 1.0586916337295802, 'particular': 1.0586916337295802, 'conic': 1.3845026050418516, 'on': 1.0586916337295802, 'trim': 1.542763787034428, 'two': 1.0586916337295802, 'estim': 1.0586916337295802, 'mean': 1.0586916337295802, 'modifi': 1.0586916337295802, 'newtonian': 1.0586916337295802, 'theori': 1.3845026050418516, 'good': 1.0586916337295802, 'agreement': 1.0586916337295802, 'experiment': 1.0586916337295802, 'fail': 1.0586916337295802, 'predict': 1.0586916337295802, 'point': 1.0586916337295802, 'flatbas': 1.0586916337295802}, ...}
        return tf

    def compute_bm25_scores(self):  # 预先计算文档集合中每个文档的BM25得分
        documents = self.processor.documents
        scores = {}
        for doc in documents:
            # scores = { doc1: { term1 : score, term2 : score, ... }, doc2: { term1 : score, term2 : score, ... }, ... }
            scores[doc] = {}
            for term in documents[doc]:
                if term not in self.idf:
                    scores[doc][term] = 0
                else:
                    scores[doc][term] = self.idf[term] * self.tf[doc][term]
        # print("Documents BM25 scores: " + str(scores) + "\n")
        return scores

    def get_document_score(self):
        self.idf = self.compute_terms_idf()
        self.tf = self.compute_documents_tf()
        self.bm25_scores = self.compute_bm25_scores()

    def export_to_json(self, output_dir):
        """导出 BM25 分数到 JSON 文件"""
        file_path = os.path.join(output_dir, '21207500-small.index.json')
        if self.bm25_scores.keys():
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.bm25_scores, f, ensure_ascii=False, indent=4)
        else:
            print("No BM25 scores to export.")


def main():
    parser = argparse.ArgumentParser(description="Process and index documents.")
    parser.add_argument('-p', '--path', type=str, required=True, help="Path to the corpus")
    args = parser.parse_args()

    documents_path = os.path.join(args.path, "documents")
    stopwords_path = os.path.join(args.path, "files", "stopwords.txt")

    index = BM25Index(DocumentProcessor(documents_path, stopwords_path))
    index.export_to_json(os.getcwd())
    print(f"Index has been exported to 21207500-small.index.json")


if __name__ == "__main__":
    main()