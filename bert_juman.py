from pathlib import Path
import argparse

import numpy as np
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel
from pyknp import Juman

def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", default=None, type=str,
                        help="input csv")
    parser.add_argument("--bert_path", default=None, type=str,
                        help="bert path")

    args = parser.parse_args()
    return args


class JumanTokenizer():
    def __init__(self):
        self.juman = Juman()

    def tokenize(self, text):
        # Jumanを用いて、日本語の文章を分かち書きする。
        result = self.juman.analysis(text)
        return [mrph.midasi for mrph in result.mrph_list()]


class BertWithJumanModel():
    def __init__(self, bert_path, vocab_file_name="vocab.txt", use_cuda=False):
        # 日本語文章をBERTに食わせるためにJumanを読み込む
        self.juman_tokenizer = JumanTokenizer()
        # 事前学習済みのBERTモデルを読み込む
        self.model = BertModel.from_pretrained(bert_path)
        # 事前学習済みのBERTモデルのTokenizerを読み込む
        self.bert_tokenizer = BertTokenizer(Path(bert_path) / vocab_file_name,
                                            do_lower_case=False, do_basic_tokenize=False)
        # CUDA-GPUを利用するかどうかのフラグ読み込み
        self.use_cuda = use_cuda

    def _preprocess_text(self, text):
        # 事前処理、テキストの半角スペースは削除
        return text.replace(" ", "")  # for Juman

    def get_sentence_embedding(self, text, pooling_layer=-2, pooling_strategy="REDUCE_MEAN"):
        # テキストの半角スペースを削除する
        preprocessed_text = self._preprocess_text(text)
        # 日本語のテキストを分かち書きし、トークンリストに変換する
        tokens = self.juman_tokenizer.tokenize(preprocessed_text)
        # トークンを半角スペースで結合しstrに変換する
        bert_tokens = self.bert_tokenizer.tokenize(" ".join(tokens))
        # テキストのサイズは128までなので、ヘッダ + トークン126個 + フッタを作成
        # トークンをidに置換する
        ids = self.bert_tokenizer.convert_tokens_to_ids(["[CLS]"] + bert_tokens[:126] + ["[SEP]"]) # max_seq_len-2
        tokens_tensor = torch.tensor(ids).reshape(1, -1)

        if self.use_cuda:
            # GPUの利用チェック、利用
            tokens_tensor = tokens_tensor.to('cuda')
            self.model.to('cuda')

        # モデルを評価モードに変更
        self.model.eval()
        with torch.no_grad():
            # 自動微分を適用しない（メモリ・高速化などなど）
            # id列からベクトル表現を計算する
            all_encoder_layers, _ = self.model(tokens_tensor)

            # SWEMと同じ方法でベクトルを時間方向にaverage-poolingしているらしい
            # 文章列によって次元が可変になってしまうので、伸びていく方向に対してプーリングを行い次元を固定化する
            # https://yag-ays.github.io/project/swem/
            embedding = all_encoder_layers[pooling_layer].cpu().numpy()[0]
            if pooling_strategy == "REDUCE_MEAN":
                return np.mean(embedding, axis=0)
            elif pooling_strategy == "REDUCE_MAX":
                return np.max(embedding, axis=0)
            elif pooling_strategy == "REDUCE_MEAN_MAX":
                return np.r_[np.max(embedding, axis=0), np.mean(embedding, axis=0)]
            elif pooling_strategy == "CLS_TOKEN":
                return embedding[0]
            else:
                raise ValueError("specify valid pooling_strategy: {REDUCE_MEAN, REDUCE_MAX, REDUCE_MEAN_MAX, CLS_TOKEN}")

if __name__ == '__main__':
    import pandas as pd
    args = parse_argument()
    bwjm = BertWithJumanModel(
        bert_path=args.bert_path
    )
    df = pd.read_csv(
        args.csv_path
    )
    for row in df['detail'].tolist():
        print('\t'.join(map(str, bwjm.get_sentence_embedding(row))))