import json
import torch
from transformers import BertTokenizer
from collections import defaultdict
import copy

# Assuming you have a ConfigRel class defined in utils.config_rel
from utils.config_rel import ConfigRel, USE_CUDA

class DataPreparationRel:
    def __init__(self, config):
        self.config = config
        self.get_token2id()
        self.rel_cnt = defaultdict(int)

    def get_data(self, file_path, is_test=False):
        data = []
        cnt = 0
        with open(file_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
            for entry in dataset:
                cnt += 1
                if cnt > self.config.num_sample:
                    break
                text = entry.get('data', {}).get('text', '').lower()
                print(f"Processing entry {cnt}: {text}")
                annotations = entry.get('annotations', [])

                for annotation in annotations:
                    result = annotation.get('result', [])
                    for res in result:
                        if 'labels' not in res or not res['labels']:
                            continue
                        if res.get('type', '') == 'relation':
                            try:
                                relation = res['labels'][0]
                            except IndexError:
                                continue  # Skip this entry if labels list is empty
                            subject_id = res.get('from_id', '')
                            object_id = res.get('to_id', '')
                            subject = next((item for item in result if item.get('id') == subject_id), {}).get('value',
                                                                                                              {}).get(
                                'text', '').lower()
                            object = next((item for item in result if item.get('id') == object_id), {}).get('value',
                                                                                                            {}).get(
                                'text', '').lower()

                            if self.rel_cnt[relation] > self.config.rel_num:
                                continue
                            self.rel_cnt[relation] += 1

                            sentence_cls = ''.join([text])
                            item = {'sentence_cls': sentence_cls, 'relation': relation, 'text': text,
                                    'subject': subject, 'object': object}
                            # print("item:", item)
                            # print("--------------------------------------------------------------------")
                            data.append(item)

                            # Ensure subject and object are defined
                            if subject and object:
                                sentence_neg = ''.join([text])
                                item_neg = {'sentence_cls': sentence_neg, 'relation': 'N', 'text': text,
                                            'subject': object, 'object': subject}
                                data.append(item_neg)

        if not data:
            raise ValueError(f"No data found in file: {file_path}")

        print(f"Loaded {len(data)} items from {file_path}")
        dataset = Dataset(data)
        if is_test:
            dataset.is_test = True
        data_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.config.batch_size,
            collate_fn=dataset.collate_fn,
            shuffle=True,
            drop_last=True
        )

        return data_loader

    def get_token2id(self):
        model_dir = '../pre_models/bert-base-chinese'
        self.bert_tokenizer = BertTokenizer.from_pretrained(model_dir)

    def get_train_dev_data(self, path_train=None, path_dev=None, path_test=None, is_test=False):
        train_loader, dev_loader, test_loader = None, None, None
        if path_train is not None:
            train_loader = self.get_data(path_train)
        if path_dev is not None:
            dev_loader = self.get_data(path_dev)
        if path_test is not None:
            test_loader = self.get_data(path_test, is_test=True)

        return train_loader, dev_loader, test_loader


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = copy.deepcopy(data)
        self.is_test = False
        with open('../data/my_rel2id.json', 'r', encoding='utf-8') as f:
            self.rel2id = json.load(f)
        model_dir = '../pre_models/bert-base-chinese'
        self.bert_tokenizer = BertTokenizer.from_pretrained(model_dir)

    def __getitem__(self, index):
        sentence_cls = self.data[index]['sentence_cls']
        relation = self.data[index]['relation']
        text = self.data[index]['text']
        subject = self.data[index]['subject']
        object = self.data[index]['object']

        data_info = {'sentence_cls': sentence_cls, 'relation': relation, 'text': text,
                     'subject': subject, 'object': object}
        return data_info

    def __len__(self):
        return len(self.data)

    def collate_fn(self, data_batch):
        def merge(sequences):
            lengths = [len(seq) for seq in sequences]
            max_length = max(lengths)
            padded_seqs = torch.zeros(len(sequences), max_length)
            tmp_pad = torch.ones(1, max_length)
            mask_tokens = torch.zeros(len(sequences), max_length)
            for i, seq in enumerate(sequences):
                end = lengths[i]
                seq = torch.LongTensor(seq)
                if len(seq) != 0:
                    padded_seqs[i, :end] = seq[:end]
                    mask_tokens[i, :end] = tmp_pad[0, :end]
            return padded_seqs, mask_tokens

        item_info = {}
        for key in data_batch[0].keys():
            item_info[key] = [d[key] for d in data_batch]

        sentence_cls = [self.bert_tokenizer.encode(sentence, add_special_tokens=True) for sentence in
                        item_info['sentence_cls']]

        relation = torch.Tensor([self.rel2id[rel] for rel in item_info['relation']]).to(torch.int64)

        sentence_cls, mask_tokens = merge(sentence_cls)
        sentence_cls = sentence_cls.to(torch.int64)
        mask_tokens = mask_tokens.to(torch.int64)
        relation = relation.to(torch.int64)

        if USE_CUDA:
            sentence_cls = sentence_cls.contiguous().cuda()
            mask_tokens = mask_tokens.contiguous().cuda()
            relation = relation.contiguous().cuda()
        else:
            sentence_cls = sentence_cls.contiguous()
            mask_tokens = mask_tokens.contiguous()
            relation = relation.contiguous()

        data_info = {"sentence_cls": sentence_cls, "mask_tokens": mask_tokens, "relation": relation}
        data_info['text'] = item_info['text']
        data_info['subject'] = item_info['subject']
        data_info['object'] = item_info['object']

        return data_info


if __name__ == '__main__':
    config = ConfigRel()
    process = DataPreparationRel(config)
    train_loader, dev_loader, test_loader = process.get_train_dev_data('../data/reldata/first_rel.json')
    if train_loader:
        print(f"Loaded train data with {len(train_loader.dataset)} items.")
    else:
        print("No train data loaded.")
    for item in train_loader:
        print("---------------")
        print(item)
