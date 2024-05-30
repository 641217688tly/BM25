"""
evaluate_small_corpus.py

该程序基于query_small_corpus.py自动模式的输出（存储在当前工作目录中的“21207500-small.results”文件中，将“21207500”替换为你的UCD学生号码）计算适当的评估指标。

程序应根据语料库的“files”目录中的“qrels.txt”文件中包含的相关性判断计算以下指标：

Precision
Recall
R-Precision
P@15
MAP
NDCG@15

程序应通过以下方式运行：
./evaluate_small_corpus.py -p /path/to/comp3009j-corpus-small
"""
import argparse
import copy
import math
import os


class Estimator:
    def __init__(self, qrels_file_path, results_file_path):
        self.ret = self.load_results(results_file_path)
        self.rel = self.load_qrels(qrels_file_path)

    def load_results(self, file_path='results-single.txt'):
        retrieved = {}
        with open(file_path, 'r') as file:
            for line in file:
                query_id, doc_id, rank, score = line.strip().split()
                if query_id not in retrieved:
                    retrieved[query_id] = []
                retrieved[query_id].append({'doc_id': doc_id, 'rank': int(rank), 'score': float(score)})
        # retrieved: {'1': [{'doc_id': 'd12', 'rank': 1, 'score': 18.0}, {'doc_id': 'd1', 'rank': 2, 'score': 17.0}, {'doc_id': 'd19', 'rank': 3, 'score': 16.0}, {'doc_id': 'd15', 'rank': 4, 'score': 15.0}, {'doc_id': 'd11', 'rank': 5, 'score': 14.0}, {'doc_id': 'd4', 'rank': 6, 'score': 13.0}, {'doc_id': 'd7', 'rank': 7, 'score': 12.0}, {'doc_id': 'd9', 'rank': 8, 'score': 11.0}, {'doc_id': 'd6', 'rank': 9, 'score': 10.0}, {'doc_id': 'd14', 'rank': 10, 'score': 9.0}, {'doc_id': 'd3', 'rank': 11, 'score': 8.0}, {'doc_id': 'd5', 'rank': 12, 'score': 7.0}, {'doc_id': 'd16', 'rank': 13, 'score': 6.0}, {'doc_id': 'd18', 'rank': 14, 'score': 5.0}, {'doc_id': 'd13', 'rank': 15, 'score': 4.0}, {'doc_id': 'd20', 'rank': 16, 'score': 3.0}, {'doc_id': 'd8', 'rank': 17, 'score': 2.0}, {'doc_id': 'd2', 'rank': 18, 'score': 1.0}]}
        return retrieved

    def load_qrels(self, file_path='qrels-single.txt'):
        relevant = {}
        with open(file_path, 'r') as file:
            for line in file:
                query_id, _, doc_id, relevance_score = line.strip().split()
                if query_id not in relevant:
                    relevant[query_id] = {}
                relevant[query_id][doc_id] = int(relevance_score)
        # relevant: {'1': {'d1': 3, 'd3': 2, 'd7': 1, 'd10': 3, 'd11': 2, 'd16': 3, 'd17': 2, 'd18': 1, 'd2': 0, 'd6': 0, 'd8': 0, 'd12': 0, 'd13': 0, 'd14': 0, 'd15': 0, 'd20': 0}}
        # TODO 注意!!!此处的relevance不是严格意义上的rel, 它包含了所有的相关性判断, 即相关性既可能大于0(有相关性), 也可能等于0(无相关性)
        # TODO 所以在求某个query的|rel|时需要先遍历relevant[query_id]中相关性判断大于0(即有相关性)的文档数
        return relevant

    def retrieved_relevant(self, retrieved, relevant, query_id):  # 取retrieved和relevant的交集, 使得retrieved中的文档都有相关性判断
        rel_docs = relevant[
            query_id]  # {'d1': 3, 'd3': 2, 'd7': 1, 'd10': 3, 'd11': 2, 'd16': 3, 'd17': 2, 'd18': 1, 'd2': 0, 'd6': 0, 'd8': 0, 'd12': 0, 'd13': 0, 'd14': 0, 'd15': 0, 'd20': 0}
        ret_docs = retrieved[
            query_id]  # [{'doc_id': 'd12', 'rank': 1, 'score': 18.0}, {'doc_id': 'd1', 'rank': 2, 'score': 17.0}, ...]
        # 获取rel_docs和ret_docs的交集
        rel_ret_docs = []
        for ret_doct in ret_docs:  # 遍历IR系统返回的文档
            if ret_doct['doc_id'] in rel_docs:
                ret_doct['relevance'] = rel_docs[ret_doct['doc_id']]
                rel_ret_docs.append(ret_doct)
            else:
                ret_doct['relevance'] = -1  # 如果没有相关性判断, 则将其相关性设置为-1
                rel_ret_docs.append(ret_doct)
        # rel_ret_docs: [{'doc_id': 'd12', 'rank': 1, 'score': 18.0, 'relevance': 0}, {'doc_id': 'd1', 'rank': 2, 'score': 17.0, 'relevance': 3}, {'doc_id': 'd19', 'rank': 3, 'score': 16.0, 'relevance': -1}, {'doc_id': 'd15', 'rank': 4, 'score': 15.0, 'relevance': 0}, {'doc_id': 'd11', 'rank': 5, 'score': 14.0, 'relevance': 2}, {'doc_id': 'd4', 'rank': 6, 'score': 13.0, 'relevance': -1}, {'doc_id': 'd7', 'rank': 7, 'score': 12.0, 'relevance': 1}, {'doc_id': 'd9', 'rank': 8, 'score': 11.0, 'relevance': -1}, {'doc_id': 'd6', 'rank': 9, 'score': 10.0, 'relevance': 0}, {'doc_id': 'd14', 'rank': 10, 'score': 9.0, 'relevance': 0}, {'doc_id': 'd3', 'rank': 11, 'score': 8.0, 'relevance': 2}, {'doc_id': 'd5', 'rank': 12, 'score': 7.0, 'relevance': -1}, {'doc_id': 'd16', 'rank': 13, 'score': 6.0, 'relevance': 3}, {'doc_id': 'd18', 'rank': 14, 'score': 5.0, 'relevance': 1}, {'doc_id': 'd13', 'rank': 15, 'score': 4.0, 'relevance': 0}, {'doc_id': 'd20', 'rank': 16, 'score': 3.0, 'relevance': 0}, {'doc_id': 'd8', 'rank': 17, 'score': 2.0, 'relevance': 0}, {'doc_id': 'd2', 'rank': 18, 'score': 1.0, 'relevance': 0}]
        # TODO rel_ret_docs将为retrieved添加relevance字段, 其中没有被相关性判断(即没有出现在relevant中)的文档其相关性被赋值为-1
        return rel_ret_docs

    def precision(self, retrieved, relevant):
        precisions = 0
        for query_id in retrieved:
            rel_ret_docs = self.retrieved_relevant(retrieved, relevant, query_id)
            relevant_docs_num = 0
            for rel_ret_doc in rel_ret_docs:
                if rel_ret_doc['relevance'] > 0:
                    relevant_docs_num = relevant_docs_num + 1
            precisions = precisions + relevant_docs_num / len(retrieved[query_id])
        return precisions / len(retrieved)

    def recall(self, retrieved, relevant):
        recalls = 0
        for query_id in retrieved:  # 对于某一个查询
            rel_ret_docs = self.retrieved_relevant(retrieved, relevant, query_id)
            relevant_docs_num = 0
            for rel_ret_doc in rel_ret_docs:
                if rel_ret_doc['relevance'] > 0:
                    relevant_docs_num = relevant_docs_num + 1
            relevant_docs_total_num = 0
            for rel_doc in relevant[query_id]:
                if relevant[query_id][rel_doc] > 0:
                    relevant_docs_total_num = relevant_docs_total_num + 1
            recalls = recalls + relevant_docs_num / relevant_docs_total_num  # relevant_docs_total_num为|Relevant|, 此处求Relevant的逻辑是获取qrels中relevance>0的文档数
        return recalls / len(retrieved)

    def precision_at_10(self, retrieved, relevant):
        precisions = 0
        for query_id in retrieved:
            relevant_docs_num = 0
            at_n = min(10, len(retrieved[query_id]))
            for i in range(at_n):
                if retrieved[query_id][i]['doc_id'] in relevant[query_id] and relevant[query_id][retrieved[query_id][i][
                    'doc_id']] > 0:  # TODO 对于没有被相关性判断的文档其相关性也被视作为0, 如果想让没有经过相关性判断的文档不影响结果, 则需要在调整遍历条件, 使得只有相关性判断的文档才会被遍历
                    relevant_docs_num = relevant_docs_num + 1
            precisions = precisions + relevant_docs_num / at_n
        return precisions / len(retrieved)

    def r_precision(self, retrieved, relevant):
        r_precisions = 0
        for query_id in retrieved:
            relevant_docs_total_num = 0
            for rel_doc in relevant[query_id]:
                if relevant[query_id][rel_doc] > 0:
                    relevant_docs_total_num = relevant_docs_total_num + 1
            relevant_docs_num = 0
            r = min(relevant_docs_total_num, len(retrieved[query_id]))
            for i in range(r):  # TODO 对于没有被相关性判断的文档其相关性也被视作为0, 如果想让没有经过相关性判断的文档不影响结果, 则需要在调整遍历条件, 使得只有相关性判断的文档才会被遍历
                if retrieved[query_id][i]['doc_id'] in relevant[query_id] and relevant[query_id][
                    retrieved[query_id][i]['doc_id']] > 0:
                    relevant_docs_num = relevant_docs_num + 1
            r_precisions = r_precisions + relevant_docs_num / r
        return r_precisions / len(retrieved)

    def map(self, retrieved, relevant):
        maps = 0
        for query_id in retrieved:
            relevant_docs_total_num = 0
            for rel_doc in relevant[query_id]:
                if relevant[query_id][rel_doc] > 0:
                    relevant_docs_total_num = relevant_docs_total_num + 1
            map = 0
            relevant_docs_num = 0
            for i in range(1, len(retrieved[
                                      query_id]) + 1):  # TODO 对于没有被相关性判断的文档其相关性也被视作为0, 如果想让没有经过相关性判断的文档不影响结果, 则需要在调整遍历条件, 使得只有相关性判断的文档才会被遍历
                if retrieved[query_id][i - 1]['doc_id'] in relevant[query_id] and relevant[query_id][
                    retrieved[query_id][i - 1]['doc_id']] > 0:
                    relevant_docs_num = relevant_docs_num + 1
                    map = map + relevant_docs_num / i
            maps = maps + map / (relevant_docs_total_num)
        return maps / len(retrieved)

    def ndcg_at_15(self, retrieved, relevant):  # TODO 此方法将没有进行相关度判断的文档也视作相关度为0
        ndcgs = 0
        for query_id in retrieved:
            ndcg = 0
            rel_ret_docs = self.retrieved_relevant(retrieved, relevant, query_id)
            for i in range(len(rel_ret_docs)):  # 计算 Discounted Cumulated Gain
                if i == 0:
                    if rel_ret_docs[i]['relevance'] > 0:
                        rel_ret_docs[i]['dcg'] = rel_ret_docs[i]['relevance']
                    elif rel_ret_docs[i]['relevance'] == 0:
                        rel_ret_docs[i]['dcg'] = 0
                    else:  # rel_ret_docs[j]['relevance'] == -1
                        rel_ret_docs[i]['dcg'] = 0
                else:
                    if rel_ret_docs[i]['relevance'] > 0:
                        rel_ret_docs[i]['dcg'] = (rel_ret_docs[i]['relevance'] / math.log2(i + 1)) + \
                                                 rel_ret_docs[i - 1][
                                                     'dcg']
                    elif rel_ret_docs[i]['relevance'] == 0:
                        rel_ret_docs[i]['dcg'] = 0 + rel_ret_docs[i - 1]['dcg']
                    else:  # rel_ret_docs[j]['relevance'] == -1
                        rel_ret_docs[i]['dcg'] = 0 + rel_ret_docs[i - 1]['dcg']
            # 将temp_rel_ret_docs按照relevance的大小降序排序
            temp_rel_ret_docs = copy.deepcopy(rel_ret_docs)
            temp_rel_ret_docs.sort(key=lambda x: x['relevance'], reverse=True)

            for j in range(len(temp_rel_ret_docs)):  # 计算 Ideal DCG vector和 Normalised Discounted Cumulated Gain
                if j == 0:
                    if temp_rel_ret_docs[j]['relevance'] > 0:
                        temp_rel_ret_docs[j]['idcg'] = temp_rel_ret_docs[j]['relevance']
                    elif temp_rel_ret_docs[j]['relevance'] == 0:
                        temp_rel_ret_docs[j]['idcg'] = 0
                    else:  # rel_ret_docs[j]['relevance'] == -1
                        temp_rel_ret_docs[j]['idcg'] = 0
                else:
                    if temp_rel_ret_docs[j]['relevance'] > 0:
                        temp_rel_ret_docs[j]['idcg'] = (temp_rel_ret_docs[j]['relevance'] / math.log2(j + 1)) + \
                                                       temp_rel_ret_docs[j - 1][
                                                           'idcg']
                    elif temp_rel_ret_docs[j]['relevance'] == 0:
                        temp_rel_ret_docs[j]['idcg'] = 0 + temp_rel_ret_docs[j - 1]['idcg']
                    else:  # rel_ret_docs[j]['relevance'] == -1
                        temp_rel_ret_docs[j]['idcg'] = 0 + temp_rel_ret_docs[j - 1]['idcg']
            # 计算NDCG@10(如果retrieved的文档数小于10, 则取最接近10的文档数)
            count = 0
            while count < 15 and count < len(rel_ret_docs):
                if temp_rel_ret_docs[count]['idcg'] != 0:
                    ndcg = ndcg + rel_ret_docs[count]['dcg'] / temp_rel_ret_docs[count]['idcg']
                count = count + 1
            ndcgs = ndcgs + ndcg / (count + 1)
        return ndcgs / len(retrieved)

    def evaluate(self):
        print("Evaluation  results:")
        print(f"Precision:    {self.precision(self.ret, self.rel):.3f}")
        print(f"Recall:       {self.recall(self.ret, self.rel):.3f}")
        print(f"R-precision:  {self.r_precision(self.ret, self.rel):.3f}")
        print(f"P@15:         {self.precision_at_10(self.ret, self.rel):.3f}")
        print(f"NDCG@15       {self.ndcg_at_15(self.ret, self.rel):.3f}")
        print(f"MAP:          {self.map(self.ret, self.rel):.3f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate search results.")
    parser.add_argument('-p', '--path', required=True, help='Path to the corpus.')
    args = parser.parse_args()

    qrels_file_path = os.path.join(args.path, 'files', 'qrels.txt')
    if not os.path.exists(qrels_file_path):
        qrels_file_path = os.path.join(os.getcwd(), 'files', 'qrels-single.txt')

    results_file_path = os.path.join(os.getcwd(), '21207500-small.results')
    if not os.path.exists(results_file_path):  # 如果stopwords_file_path中的文件不存在,则使用默认的stopwords.txt文件
        print(
            "Results file not found. You should run the query_small_corpus.py script under small_corpus_handler folder.")

    evaluator = Estimator(qrels_file_path, results_file_path)
    evaluator.evaluate()


if __name__ == "__main__":
    main()
