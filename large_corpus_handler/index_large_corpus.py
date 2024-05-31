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
        self.docs_num = 0  # 文档总数
        self.total_doc_len = 0  # 文档总长度，用于计算平均长度
        self.avg_doc_len = 0  # 平均文档长度
        self.load_stopwords()
        self.process_documents()

    def count_valid_files_number(self, directory):
        """计算目录及其子目录中以'GX'开头的文件数量"""
        total_files = 0
        for subdir, dirs, files in os.walk(directory):
            for file in files:
                if file.startswith("GX"):
                    total_files += 1
        return total_files

    def load_stopwords(self):
        """从文件加载停用词列表"""
        with open(self.stopwords_file_path, 'r', encoding='UTF-8') as file:
            for line in file:
                self.stopwords.add(line.strip())

    def process_documents(self):
        """执行完整的文档处理流程：读取、去停用词、词干提取"""
        print("Start documents preprocessing, please Waiting...")
        start = time.time()
        total_files = self.count_valid_files_number(self.documents_dir_path)
        for subdir in os.listdir(self.documents_dir_path):  # 遍历documents目录下的所有子文件夹
            subdir_path = os.path.join(self.documents_dir_path, subdir)
            if os.path.isdir(subdir_path):
                for filename in os.listdir(subdir_path):  # 遍历子文件夹(比如GX000, GX001...)中的文件
                    if filename.startswith("GX"):  # 确保文件以'GX'开头
                        file_path = os.path.join(subdir_path, filename)
                        with open(file_path, 'r', encoding='UTF-8') as file:
                            text = file.read().lower()
                            text = text.translate(str.maketrans('', '', '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'))  # 移除标点符号
                            tokens = text.split()
                            tokens = self.remove_stopwords(tokens)  # 移除停用词
                            tokens = self.stem_words(tokens)  # 词干提取
                            self.documents[filename] = tokens
                            self.docs_num += 1  # 更新文档总数
                            self.total_doc_len += len(tokens)  # 累加当前文档的长度
                        if self.docs_num % 100 == 0 or self.docs_num == total_files:
                            print(f"Processing documents {self.docs_num}/{total_files}")
            # 所有文档处理完毕后计算平均文档长度
        if self.docs_num > 0:
            self.avg_doc_len = self.total_doc_len / self.docs_num
        end = time.time()
        print(f"Document processing completed in {end - start:.2f} seconds.\n")

    def remove_stopwords(self, tokens):
        """从给定的术语列表中移除停用词。"""
        return [term for term in tokens if term not in self.stopwords]

    def stem_words(self, tokens):
        """对给定的术语列表进行词干提取。使用词干加速器加速词干提取过程。"""
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
        """计算每个Term的IDF, idf_i = log2( (N-n_i+0.5) / (n_i+0.5) + 1) : N为文档总数, n_i为包含词i的文档数"""
        print("Computing terms IDF values...")
        documents = self.processor.documents
        idf = {}  # 存储每个term在文档集合中出现在了几个文档中
        for i, doc in enumerate(documents):  # doc为文档的key
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
        """计算文档集合内各个文档中Term的TF值"""
        print("Computing documents TF values...")
        documents = self.processor.documents
        tf = {}  # 存储每个文档中每个term的TF值: {doc1: {term1: tf1, term2: tf2, ...}, doc2: {term1: tf1, term2: tf2, ...}, ...}
        for i, doc in enumerate(documents):
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
            if (i + 1) % 1000 == 0 or i + 1 == self.processor.docs_num:
                print(f"Computing documents TF values for documents {i + 1}/{self.processor.docs_num}")
        # print("Documents TF values: " + str(tf) + "\n")
        # 计算后的文档集合中的各个文档的TF的格式如下:
        # {'GX000-01-10544170': {'static': 1.0586916337295802, 'aerodynam': 1.542763787034428, 'characterist': 1.542763787034428, 'short': 1.3845026050418516, 'blunt': 1.3845026050418516, 'cone': 1.799952348010274, 'variou': 1.0586916337295802, 'nose': 1.7418759813234317, 'base': 1.820181404463512, 'angl': 1.799952348010274, 'mach': 1.3845026050418516, 'number': 1.542763787034428, '0': 1.3845026050418516, '6': 1.0586916337295802, '5': 1.3845026050418516, 'attack': 1.3845026050418516, '180': 1.542763787034428, 'windtunnel': 1.0586916337295802, 'test': 1.542763787034428, 'perform': 1.0586916337295802, '06': 1.0586916337295802, '55': 1.0586916337295802, 'determin': 1.0586916337295802, 'coeffici': 1.0586916337295802, 'normal': 1.0586916337295802, 'forc': 1.3845026050418516, 'axial': 1.0586916337295802, 'pitch': 1.0586916337295802, 'moment': 1.0586916337295802, 'affect': 1.0586916337295802, 'chang': 1.542763787034428, 'model': 1.6980453523003483, 'halfangl': 1.7745949513495383, '10': 1.3845026050418516, '20': 1.3845026050418516, 'investig': 1.0586916337295802, 'flat': 1.3845026050418516, '50': 1.542763787034428, '70': 1.3845026050418516, 'reynold': 1.0586916337295802, 'rang': 1.0586916337295802, 'maximum': 1.0586916337295802, 'diamet': 1.0586916337295802, 'variat': 1.3845026050418516, 'result': 1.542763787034428, 'signific': 1.0586916337295802, 'lesser': 1.0586916337295802, 'effect': 1.0586916337295802, 'particular': 1.0586916337295802, 'conic': 1.3845026050418516, 'on': 1.0586916337295802, 'trim': 1.542763787034428, 'two': 1.0586916337295802, 'estim': 1.0586916337295802, 'mean': 1.0586916337295802, 'modifi': 1.0586916337295802, 'newtonian': 1.0586916337295802, 'theori': 1.3845026050418516, 'good': 1.0586916337295802, 'agreement': 1.0586916337295802, 'experiment': 1.0586916337295802, 'fail': 1.0586916337295802, 'predict': 1.0586916337295802, 'point': 1.0586916337295802, 'flatbas': 1.0586916337295802}, ...}
        return tf

    def compute_bm25_scores(self):
        """计算文档集合中每个文档的BM25得分"""
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
        """先计算TF和IDF,然后计算各个文档的BM25分数"""
        print("Start computing BM25 scores, please Waiting...")
        start = time.time()
        self.idf = self.compute_terms_idf()
        self.tf = self.compute_documents_tf()
        self.bm25_scores = self.compute_bm25_scores()
        end = time.time()
        print(f"BM25 scores computed in {end - start:.2f} seconds.\n")

    def export_to_json(self, output_dir):
        """导出 BM25 分数到 JSON 文件"""
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
