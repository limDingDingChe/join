# -*- coding: utf-8 -*-

# @Author  : xmh
# @Time    : 2021/3/3 10:02
# @File    : model_ner.py

import torch
import torch.nn as nn
from torchcrf import CRF
from utils.config_ner import USE_CUDA


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SeqLabel(nn.Module):
    def __init__(self, config):
        super().__init__()
        setup_seed(1)

        self.vocab_size = config.vocab_size
        self.embedding_dim = config.embedding_dim
        self.hidden_dim = config.hidden_dim_lstm
        self.num_layers = config.num_layers
        self.batch_size = config.batch_size
        self.layer_size = config.layer_size
        self.num_token_type = config.num_token_type
        self.config = config

        self.word_embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.token_type_embedding = nn.Embedding(config.num_token_type, config.token_type_dim)
        self.gru = nn.GRU(config.embedding_dim, config.hidden_dim_lstm, num_layers=config.num_layers, batch_first=True,
                          bidirectional=True)
        self.is_train = True
        if USE_CUDA:
            self.weights_rel = (torch.ones(self.config.num_relations) * 100).cuda()
        else:
            self.weights_rel = torch.ones(self.config.num_relations) * 100
        self.weights_rel[0] = 1

        self.V_ner = nn.Parameter(torch.rand((config.num_token_type, self.layer_size)))
        self.U_ner = nn.Parameter(torch.rand((self.layer_size, 2 * self.hidden_dim)))
        self.b_s_ner = nn.Parameter(torch.rand(self.layer_size))
        self.b_c_ner = nn.Parameter(torch.rand(config.num_token_type))

        self.dropout_embedding_layer = torch.nn.Dropout(config.dropout_embedding)
        self.dropout_ner_layer = torch.nn.Dropout(config.dropout_ner)
        self.dropout_lstm_layer = torch.nn.Dropout(config.dropout_lstm)
        self.crf_model = CRF(self.num_token_type, batch_first=True)

    def get_ner_score(self, output_lstm):
        res = torch.matmul(output_lstm, self.U_ner.transpose(-1, -2)) + self.b_s_ner
        res = torch.tanh(res)
        if self.config.use_dropout:
            res = self.dropout_ner_layer(res)
        ans = torch.matmul(res, self.V_ner.transpose(-1, -2)) + self.b_c_ner
        return ans

    def forward(self, data_item, is_test=False):
        embeddings = self.word_embedding(data_item['text_tokened'].to(torch.int64))
        if self.config.use_dropout:
            embeddings = self.dropout_embedding_layer(embeddings)
        if USE_CUDA:
            hidden_init = torch.randn(2 * self.num_layers, self.batch_size, self.hidden_dim).cuda()
        else:
            hidden_init = torch.randn(2 * self.num_layers, self.batch_size, self.hidden_dim)
        output_lstm, h_n = self.gru(embeddings, hidden_init)
        ner_score = self.get_ner_score(output_lstm)

        # Debugging information
        # print("ner_score shape:", ner_score.shape)
        # print("ner_score sample values:", ner_score[0][0][:10].tolist())

        if USE_CUDA:
            self.crf_model = self.crf_model.cuda()
        if not is_test:
            log_likelihood = self.crf_model(ner_score, data_item['token_type_list'].to(torch.int64).clamp(0,
                                                                                                          self.num_token_type - 1),
                                            mask=data_item['mask_tokens'])
            loss_ner = -log_likelihood

        pred_ner = self.crf_model.decode(ner_score)

        if is_test:
            return pred_ner
        return loss_ner, pred_ner