import json
import torch
import copy
from utils.config_ner import ConfigNer, USE_CUDA
import numpy as np


class ModelDataPreparation:
    def __init__(self, config):
        self.config = config
        self.get_type2id()

    def subject_object_labeling(self, label_list, text, text_tokened):
        labeling_list = ["O" for _ in range(len(text_tokened))]
        have_error = False

        for label_item in label_list:
            start = label_item["start"]
            end = label_item["end"]
            entity_text = label_item["text"]
            entity_type = label_item["labels"][0]

            entity_tokened = [c for c in entity_text.lower()]
            entity_length = len(entity_tokened)

            try:
                idx_start = text_tokened.index(entity_tokened[0], start)
                idx_end = idx_start + entity_length
                if text_tokened[idx_start:idx_end] == entity_tokened:
                    labeling_list[idx_start] = "B-" + entity_type
                    if entity_length > 1:
                        labeling_list[idx_start + 1:idx_end] = ["I-" + entity_type] * (entity_length - 1)
                else:
                    have_error = True
                    break
            except ValueError:
                have_error = True
                break

        return labeling_list, have_error

    def get_rid_unkonwn_word(self, text):
        text_rid = []
        for token in text:
            if token in self.token2id.keys():
                text_rid.append(token)
        return text_rid

    def get_type2id(self):
        self.token_type2id = {}
        for i, token_type in enumerate(self.config.token_types):
            self.token_type2id[token_type] = i

        # Add all possible B- and I- tags
        entity_types = ["资产", "攻击", "威胁", "协议", "漏洞", "缓解措施"]  # Add all entity types here
        for entity_type in entity_types:
            self.token_type2id["B-" + entity_type] = len(self.token_type2id)
            self.token_type2id["I-" + entity_type] = len(self.token_type2id)

        self.token2id = {}
        with open(self.config.vocab_file, 'r', encoding='utf-8') as f:
            cnt = 0
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 1:
                    continue
                self.token2id[parts[0]] = cnt
                cnt += 1
        self.token2id[' '] = cnt

    def get_data(self, file_path, is_test=False):
        data = []
        cnt = 0
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                data_json = json.load(f)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
                return []

        for data_item in data_json:
            cnt += 1
            if cnt > self.config.num_sample:
                break
            text = data_item['text']
            text_tokened = [c.lower() for c in text]
            text_tokened = self.get_rid_unkonwn_word(text_tokened)

            if not is_test:
                label_list = data_item['label']
                token_type_list, have_error = self.subject_object_labeling(
                    label_list=label_list, text=text, text_tokened=text_tokened
                )
                token_type_origin = token_type_list
                if have_error:
                    continue
            else:
                token_type_list = None
                token_type_origin = None

            item = {'text_tokened': text_tokened, 'token_type_list': token_type_list}
            item['text_tokened'] = [self.token2id[x] for x in item['text_tokened']]
            if not is_test:
                item['token_type_list'] = [self.token_type2id[x] for x in item['token_type_list']]
            item['text'] = ''.join(text_tokened)
            item['label_list'] = label_list
            item['token_type_origin'] = token_type_origin
            data.append(item)

        dataset = Dataset(data)
        if is_test:
            dataset.is_test = True
        data_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.config.batch_size,
            collate_fn=dataset.collate_fn,
            drop_last=True
        )
        return data_loader

    def get_train_dev_data(self, path_train=None, path_dev=None, path_test=None):
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

    def __getitem__(self, index):
        text_tokened = self.data[index]['text_tokened']
        token_type_list = self.data[index]['token_type_list']

        data_info = {}
        for key in self.data[0].keys():
            if key in locals():
                data_info[key] = locals()[key]

        data_info['text'] = self.data[index]['text']
        data_info['label_list'] = self.data[index]['label_list']
        data_info['token_type_origin'] = self.data[index]['token_type_origin']
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
        token_type_list = None
        text_tokened, mask_tokens = merge(item_info['text_tokened'])
        if not self.is_test:
            token_type_list, _ = merge(item_info['token_type_list'])
        # convert to contiguous and cuda
        if USE_CUDA:
            text_tokened = text_tokened.contiguous().cuda()
            mask_tokens = mask_tokens.contiguous().cuda()
        else:
            text_tokened = text_tokened.contiguous()
            mask_tokens = mask_tokens.contiguous()

        if not self.is_test:
            if USE_CUDA:
                token_type_list = token_type_list.contiguous().cuda()
            else:
                token_type_list = token_type_list.contiguous()

        data_info = {"mask_tokens": mask_tokens.to(torch.uint8)}
        data_info['text'] = item_info['text']
        data_info['label_list'] = item_info['label_list']
        data_info['token_type_origin'] = item_info['token_type_origin']
        for key in item_info.keys():
            if key in locals():
                data_info[key] = locals()[key]

        return data_info


if __name__ == '__main__':
    config = ConfigNer()
    process = ModelDataPreparation(config)
    train_loader, dev_loader, test_loader = process.get_train_dev_data('../data/mydata/mydata.json')
    print(train_loader)
    for item in train_loader:
        print(item)
