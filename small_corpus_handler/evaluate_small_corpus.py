"""
Student ID: 21207500
Student Name: Liyan Tao
"""

import argparse
import math
import os


class Estimator:
    def __init__(self, qrels_file_path, results_file_path):
        self.ret = self.load_results(results_file_path)
        self.rel = self.load_qrels(qrels_file_path)

    def load_results(self, file_path='results.txt'):
        """Load the results file and return a dictionary where the key is query_id and the value is a list of dictionaries. Each dictionary has keys doc_id, rank, and score."""
        retrieved = {}
        with open(file_path, 'r') as file:
            for line in file:
                query_id, doc_id, rank, score = line.strip().split()
                if query_id not in retrieved:
                    retrieved[query_id] = []
                retrieved[query_id].append({'doc_id': doc_id, 'rank': int(rank), 'score': float(score)})
        # retrieved: {'1': [{'doc_id': 'd12', 'rank': 1, 'score': 18.0}, {'doc_id': 'd1', 'rank': 2, 'score': 17.0}, {'doc_id': 'd19', 'rank': 3, 'score': 16.0}, {'doc_id': 'd15', 'rank': 4, 'score': 15.0}, {'doc_id': 'd11', 'rank': 5, 'score': 14.0}, {'doc_id': 'd4', 'rank': 6, 'score': 13.0}, {'doc_id': 'd7', 'rank': 7, 'score': 12.0}, {'doc_id': 'd9', 'rank': 8, 'score': 11.0}, {'doc_id': 'd6', 'rank': 9, 'score': 10.0}, {'doc_id': 'd14', 'rank': 10, 'score': 9.0}, {'doc_id': 'd3', 'rank': 11, 'score': 8.0}, {'doc_id': 'd5', 'rank': 12, 'score': 7.0}, {'doc_id': 'd16', 'rank': 13, 'score': 6.0}, {'doc_id': 'd18', 'rank': 14, 'score': 5.0}, {'doc_id': 'd13', 'rank': 15, 'score': 4.0}, {'doc_id': 'd20', 'rank': 16, 'score': 3.0}, {'doc_id': 'd8', 'rank': 17, 'score': 2.0}, {'doc_id': 'd2', 'rank': 18, 'score': 1.0}]}
        return retrieved

    def load_qrels(self, file_path='qrels.txt'):
        """Load the qrels file and return a dictionary where the key is query_id and the value is a dictionary. The dictionary has keys as doc_id and values as relevance_score."""
        relevant = {}
        with open(file_path, 'r') as file:
            for line in file:
                query_id, _, doc_id, relevance_score = line.strip().split()
                if query_id not in relevant:
                    relevant[query_id] = {}
                relevant[query_id][doc_id] = int(relevance_score)
        # relevant: {'1': {'d1': 3, 'd3': 2, 'd7': 1, 'd10': 3, 'd11': 2, 'd16': 3, 'd17': 2, 'd18': 1, 'd2': 0, 'd6': 0, 'd8': 0, 'd12': 0, 'd13': 0, 'd14': 0, 'd15': 0, 'd20': 0}}
        # Note!!! The relevance here is not strictly rel. It includes all relevance judgments, i.e., relevance can be greater than 0 (relevant) or equal to 0 (non-relevant).
        # Therefore, when calculating |rel| for a query, you need to traverse the relevant[query_id] to count the number of documents with relevance greater than 0 (i.e., relevant).
        return relevant

    def retrieved_relevant(self, retrieved, relevant, query_id):
        """Get the intersection of retrieved and relevant. Documents in retrieved that are not in relevant will have their relevance set to -1, so that all documents in retrieved have relevance judgments."""
        rel_docs = relevant[query_id]  # {'d1': 3, 'd3': 2, 'd7': 1, 'd10': 3, 'd11': 2, 'd16': 3, 'd17': 2, 'd18': 1, 'd2': 0, 'd6': 0, 'd8': 0, 'd12': 0, 'd13': 0, 'd14': 0, 'd15': 0, 'd20': 0}
        ret_docs = retrieved[query_id]  # [{'doc_id': 'd12', 'rank': 1, 'score': 18.0}, {'doc_id': 'd1', 'rank': 2, 'score': 17.0}, ...]
        # Get the intersection of rel_docs and ret_docs
        rel_ret_docs = []
        for ret_doct in ret_docs:  # Traverse the documents returned by the IR system
            if ret_doct['doc_id'] in rel_docs:
                ret_doct['relevance'] = rel_docs[ret_doct['doc_id']]
                rel_ret_docs.append(ret_doct)
            else:
                ret_doct['relevance'] = -1  # If there is no relevance judgment, set the relevance to -1
                rel_ret_docs.append(ret_doct)
        # rel_ret_docs: [{'doc_id': 'd12', 'rank': 1, 'score': 18.0, 'relevance': 0}, {'doc_id': 'd1', 'rank': 2, 'score': 17.0, 'relevance': 3}, {'doc_id': 'd19', 'rank': 3, 'score': 16.0, 'relevance': -1}, {'doc_id': 'd15', 'rank': 4, 'score': 15.0, 'relevance': 0}, {'doc_id': 'd11', 'rank': 5, 'score': 14.0, 'relevance': 2}, {'doc_id': 'd4', 'rank': 6, 'score': 13.0, 'relevance': -1}, {'doc_id': 'd7', 'rank': 7, 'score': 12.0, 'relevance': 1}, {'doc_id': 'd9', 'rank': 8, 'score': 11.0, 'relevance': -1}, {'doc_id': 'd6', 'rank': 9, 'score': 10.0, 'relevance': 0}, {'doc_id': 'd14', 'rank': 10, 'score': 9.0, 'relevance': 0}, {'doc_id': 'd3', 'rank': 11, 'score': 8.0, 'relevance': 2}, {'doc_id': 'd5', 'rank': 12, 'score': 7.0, 'relevance': -1}, {'doc_id': 'd16', 'rank': 13, 'score': 6.0, 'relevance': 3}, {'doc_id': 'd18', 'rank': 14, 'score': 5.0, 'relevance': 1}, {'doc_id': 'd13', 'rank': 15, 'score': 4.0, 'relevance': 0}, {'doc_id': 'd20', 'rank': 16, 'score': 3.0, 'relevance': 0}, {'doc_id': 'd8', 'rank': 17, 'score': 2.0, 'relevance': 0}, {'doc_id': 'd2', 'rank': 18, 'score': 1.0, 'relevance': 0}]
        # rel_ret_docs will add the relevance field to retrieved. Documents not judged for relevance (i.e., not in relevant) will have their relevance set to -1.
        return rel_ret_docs

    def precision(self, retrieved, relevant):
        """Calculate the precision"""
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
        """Calculate the recall"""
        recalls = 0
        for query_id in retrieved:  # For a specific query
            rel_ret_docs = self.retrieved_relevant(retrieved, relevant, query_id)
            relevant_docs_num = 0
            for rel_ret_doc in rel_ret_docs:
                if rel_ret_doc['relevance'] > 0:
                    relevant_docs_num = relevant_docs_num + 1
            relevant_docs_total_num = 0
            for rel_doc in relevant[query_id]:
                if relevant[query_id][rel_doc] > 0:
                    relevant_docs_total_num = relevant_docs_total_num + 1
            recalls = recalls + relevant_docs_num / relevant_docs_total_num  # relevant_docs_total_num is |Relevant|. Here, the logic for obtaining Relevant is to count the number of documents with relevance > 0 in qrels.
        return recalls / len(retrieved)

    def precision_at_10(self, retrieved, relevant):
        """Calculate the precision@10"""
        precisions = 0
        for query_id in retrieved:
            relevant_docs_num = 0
            at_n = min(10, len(retrieved[query_id]))
            for i in range(at_n):
                if retrieved[query_id][i]['doc_id'] in relevant[query_id] and relevant[query_id][retrieved[query_id][i]['doc_id']] > 0:
                    relevant_docs_num = relevant_docs_num + 1
            precisions = precisions + relevant_docs_num / at_n
        return precisions / len(retrieved)

    def r_precision(self, retrieved, relevant):
        """Calculate the R-precision"""
        r_precisions = 0
        for query_id in retrieved:
            relevant_docs_total_num = 0
            for rel_doc in relevant[query_id]:
                if relevant[query_id][rel_doc] > 0:
                    relevant_docs_total_num = relevant_docs_total_num + 1
            relevant_docs_num = 0
            r = min(relevant_docs_total_num, len(retrieved[query_id]))
            for i in range(r):
                if retrieved[query_id][i]['doc_id'] in relevant[query_id] and relevant[query_id][retrieved[query_id][i]['doc_id']] > 0:
                    relevant_docs_num = relevant_docs_num + 1
            r_precisions = r_precisions + relevant_docs_num / r
        return r_precisions / len(retrieved)

    def map(self, retrieved, relevant):
        """Calculate the MAP"""
        maps = 0
        for query_id in retrieved:
            relevant_docs_total_num = 0
            for rel_doc in relevant[query_id]:
                if relevant[query_id][rel_doc] > 0:
                    relevant_docs_total_num = relevant_docs_total_num + 1
            map = 0
            relevant_docs_num = 0
            for i in range(1, len(retrieved[query_id]) + 1):
                if retrieved[query_id][i - 1]['doc_id'] in relevant[query_id] and relevant[query_id][retrieved[query_id][i - 1]['doc_id']] > 0:
                    relevant_docs_num = relevant_docs_num + 1
                    map = map + relevant_docs_num / i
            maps = maps + map / (relevant_docs_total_num)
        return maps / len(retrieved)

    def ndcg_at_n(self, retrieved, relevant, n=15):
        """Calculate the NDCG@n"""
        ndcgs = 0
        for query_id in retrieved:
            rel_ret_docs = self.retrieved_relevant(retrieved, relevant, query_id)
            for i in range(len(rel_ret_docs)):  # Calculate Discounted Cumulative Gain
                if i == 0:
                    if rel_ret_docs[i]['relevance'] > 0:
                        rel_ret_docs[i]['dcg'] = rel_ret_docs[i]['relevance']
                    elif rel_ret_docs[i]['relevance'] == 0:
                        rel_ret_docs[i]['dcg'] = 0
                    else:  # rel_ret_docs[j]['relevance'] == -1
                        rel_ret_docs[i]['dcg'] = 0
                else:
                    if rel_ret_docs[i]['relevance'] > 0:
                        rel_ret_docs[i]['dcg'] = (rel_ret_docs[i]['relevance'] / math.log2(i + 1)) + rel_ret_docs[i - 1]['dcg']
                    elif rel_ret_docs[i]['relevance'] == 0:
                        rel_ret_docs[i]['dcg'] = 0 + rel_ret_docs[i - 1]['dcg']
                    else:  # rel_ret_docs[j]['relevance'] == -1
                        rel_ret_docs[i]['dcg'] = 0 + rel_ret_docs[i - 1]['dcg']
            # Sort temp_rel_ret_docs in descending order of relevance
            temp_rel_ret_docs = relevant[query_id]
            temp_rel_ret_docs = sorted(temp_rel_ret_docs.items(), key=lambda x: x[1], reverse=True)
            temp_rel_ret_docs = [{'doc_id': doc_id, 'relevance': relevance} for doc_id, relevance in temp_rel_ret_docs]
            for j in range(len(temp_rel_ret_docs)):  # Calculate Ideal DCG vector and Normalized Discounted Cumulative Gain
                if j == 0:
                    if temp_rel_ret_docs[j]['relevance'] > 0:
                        temp_rel_ret_docs[j]['idcg'] = temp_rel_ret_docs[j]['relevance']
                    elif temp_rel_ret_docs[j]['relevance'] == 0:
                        temp_rel_ret_docs[j]['idcg'] = 0
                    else:  # rel_ret_docs[j]['relevance'] == -1
                        temp_rel_ret_docs[j]['idcg'] = 0
                else:
                    if temp_rel_ret_docs[j]['relevance'] > 0:
                        # print(f'j: {j}; temp_rel_ret_docs[j]: {temp_rel_ret_docs[j]};  temp_rel_ret_docs[j - 1]: {temp_rel_ret_docs[j - 1]}')
                        temp_rel_ret_docs[j]['idcg'] = (temp_rel_ret_docs[j]['relevance'] / math.log2(j + 1)) + temp_rel_ret_docs[j - 1]['idcg']
                    elif temp_rel_ret_docs[j]['relevance'] == 0:
                        temp_rel_ret_docs[j]['idcg'] = 0 + temp_rel_ret_docs[j - 1]['idcg']
                    else:  # rel_ret_docs[j]['relevance'] == -1
                        temp_rel_ret_docs[j]['idcg'] = 0 + temp_rel_ret_docs[j - 1]['idcg']
            # Calculate NDCG@10 (if the number of retrieved documents is less than 10, take the closest number to 10)
            if len(rel_ret_docs) < n:
                n = len(rel_ret_docs)
            ndcg = rel_ret_docs[min(n - 1, len(rel_ret_docs) - 1)]['dcg'] / temp_rel_ret_docs[min(n - 1, len(temp_rel_ret_docs) - 1)]['idcg']
            ndcgs = ndcgs + ndcg
        return ndcgs / len(retrieved)

    def evaluate(self):
        """Print the evaluation results"""
        print("Evaluation results:")
        print(f"Precision:    {self.precision(self.ret, self.rel):.3f}")
        print(f"Recall:       {self.recall(self.ret, self.rel):.3f}")
        print(f"R-precision:  {self.r_precision(self.ret, self.rel):.3f}")
        print(f"P@15:         {self.precision_at_10(self.ret, self.rel):.3f}")
        print(f"MAP:          {self.map(self.ret, self.rel):.3f}")
        print(f"NDCG@15       {self.ndcg_at_n(self.ret, self.rel, 15):.3f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate search results.")
    parser.add_argument('-p', '--path', required=True, help='Path to the corpus.')
    args = parser.parse_args()

    qrels_file_path = os.path.join(args.path, 'files', 'qrels.txt')
    if not os.path.exists(qrels_file_path):  # If the file in qrels_file_path does not exist, use the project's qrels file
        print(f"Qrels file not found at {qrels_file_path}. Using project's qrels file.")
        qrels_file_path = os.path.join(os.getcwd(), 'files', 'qrels.txt')

    results_file_path = os.path.join(os.getcwd(), '21207500-small.results')
    if not os.path.exists(results_file_path):
        print("Results file not found. You should run the query_small_corpus.py script under small_corpus_handler folder.")

    evaluator = Estimator(qrels_file_path, results_file_path)
    evaluator.evaluate()


if __name__ == "__main__":
    main()
