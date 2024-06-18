
"""
file description:：

"""
import torch

if torch.cuda.is_available():
    USE_CUDA = True
    print("USE_CUDA....")
else:
    USE_CUDA = False


class ConfigNer:
    def __init__(self,
                 lr=0.0001,
                 epochs=20,
                 vocab_size=22000,
                 embedding_dim=32,
                 hidden_dim_lstm=128,
                 num_layers=3,
                 batch_size=32,
                 layer_size=128,
                 token_type_dim=8
                 ):
        self.lr = lr
        self.epochs = epochs
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim_lstm = hidden_dim_lstm
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.layer_size = layer_size
        self.token_type_dim = token_type_dim
        self.relations = ["N", '受到', "利用", "带来", "面临", "防止", "修复", "通过", "使用", "存在"]
        self.num_relations = len(self.relations)
        # self.token_types_origin = ['Date', 'Number', 'Text', '书籍', '人物', '企业', '作品', '出版社', '历史人物', '国家', '图书作品', '地点', '城市', '学校', '学科专业',
        #  '影视作品', '景点', '机构', '歌曲', '气候', '生物', '电视综艺', '目', '网站', '网络小说', '行政区', '语言', '音乐专辑']
        self.token_types_origin = ["资产", "攻击", "威胁", "协议", "漏洞", "缓解措施"]
        self.token_types = self.get_token_types()
        self.num_token_type = len(self.token_types)
        self.vocab_file = '../data/vocab.txt'
        self.max_seq_length = 256
        self.num_sample = 204800

        self.dropout_embedding = 0.1  # 从0.2到0.1
        self.dropout_lstm = 0.1
        self.dropout_lstm_output = 0.9
        self.dropout_head = 0.9  # 只更改这个参数 0.9到0.5
        self.dropout_ner = 0.8
        self.use_dropout = True
        self.threshold_rel = 0.95  # 从0.7到0.95
        self.teach_rate = 0.2
        self.ner_checkpoint_path = '../models/sequence_labeling/'
    
    def get_token_types(self):
        token_type_bio = []
        for token_type in self.token_types_origin:
            token_type_bio.append('B-' + token_type)
            token_type_bio.append('I-' + token_type)
        token_type_bio.append('O')
        
        return token_type_bio

