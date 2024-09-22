import os
import torch
import numpy as np
import pandas as pd
import json
import re
from scipy import io
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
from Models.interpretable_diffusion.model_utils import normalize_to_neg_one_to_one, unnormalize_to_zero_to_one
from Utils.masking_utils import noise_mask
from transformers import AutoTokenizer

def replace(text, numbers, num_token="[NUM]"):
    text = text.replace(num_token, "¬").replace("¬¬", "¬, ¬").replace("¬¬", "¬, ¬")
    for number in numbers:
        text = text.replace("¬", str(number), 1)
    return text

def compress_matrix(text):
    text = (
        text.replace("¬, ¬", "¬¬")
        .replace("¬, ¬", "¬¬")
        .replace("¬,¬", "¬¬")
        .replace("¬,¬", "¬¬")
    )
    return text

def extract(text, num_token="[NUM]"):
    import re
    pattern = r"(?<!\')-?\d+(\.\d+)?([eE][-+]?\d+)?(?!\'|\d)"
    numbers = []
    def replace(match):
        numbers.append(match.group())
        return "¬"
    nonum_text = re.sub(pattern, replace, text)
    return compress_matrix(nonum_text).replace("¬", num_token), numbers


def tokenize_fnc(sample, tokenizer, scaler,num_token="[NUM]"):
    if type(sample) != str:
        sample = sample["text"]
    sample = sample.replace(" ", "")
    num_token_id = tokenizer.convert_tokens_to_ids(num_token)
    nonum_text, numbers = extract(sample, num_token=num_token)
    out = tokenizer(nonum_text, max_length=256, padding="longest", return_tensors='pt')
    ids = np.array(out["input_ids"])
    ids = ids.flatten()
    locs = ids == num_token_id
    num_embed = np.ones(len(ids)).astype(np.float16)
    num_locs = np.sum(locs)
    if len(numbers) > num_locs:
        numbers = numbers[:num_locs]
    elif len(numbers) < num_locs:
        numbers = np.pad(numbers, (0, num_locs - len(numbers)), constant_values=1)
    num_embed[locs] = numbers
    return ids, num_embed

class CustomDataset(Dataset):
    def __init__(
        self, 
        name,
        data_root, 
        window=64, 
        proportion=0.8, 
        save2npy=True, 
        neg_one_to_one=True,
        seed=123,
        period='train',
        output_dir='./OUTPUT',
        predict_length=None,
        missing_ratio=None,
        style='separate', 
        distribution='geometric', 
        mean_mask_length=3
    ):
        super(CustomDataset, self).__init__()
        assert period in ['train', 'test'], 'period must be train or test.'
        if period == 'train':
            assert ~(predict_length is not None or missing_ratio is not None), ''
        self.name, self.pred_len, self.missing_ratio = name, predict_length, missing_ratio
        self.style, self.distribution, self.mean_mask_length = style, distribution, mean_mask_length

        # JSON数据相关初始化
        self.json_files = [os.path.join(data_root, f) for f in os.listdir(data_root) if f.endswith('.json')]
        self.json_data = []
        self.json_files.sort(key=lambda x: int(re.search(r'_(\d+)', x).group(1)))  # 按文件名中的数字部分排序
        for json_file in self.json_files:
            with open(json_file, 'r', encoding='utf-8') as f:
                self.json_data.append(json.load(f))
        self.tokenizer = AutoTokenizer.from_pretrained(r"././robertabase", local_files_only=True)
        self.tokenizer.add_tokens(["[NUM]"])
        self.max_length = 256  # 512 1256
        self.rawdata, self.scaler, self.text, self.text_num = self.read_data(self.json_data, self.tokenizer,
                                                                             self.max_length, self.name)  # read_data

        # self.rawdata, self.scaler = self.read_data(data_root, self.name)
        self.dir = os.path.join(output_dir, 'samples')
        os.makedirs(self.dir, exist_ok=True)

        self.window, self.period = window, period
        self.len, self.var_num = self.rawdata.shape[0], self.rawdata.shape[-1]
        self.sample_num_total = max(self.len - self.window + 1, 0)
        self.save2npy = save2npy
        self.auto_norm = neg_one_to_one

        self.data = self.__normalize(self.rawdata)
        self.normed_text_num = [self.__normalize(data) for data in self.text_num]
        self.text_num = [self.normed_text_num[i] for i in range(len(self.normed_text_num))]
        train_data, inference_data, train_text, test_text, train_text_num, test_text_num = self.__getsamples(self.data, self.text, self.text_num, proportion, seed, )
        self.samples = train_data if period == 'train' else inference_data
        self.text = train_text if period == 'train' else test_text
        self.text_num = train_text_num if period == 'train' else test_text_num
        if period == 'test':
            if missing_ratio is not None:
                self.masking = self.mask_data(seed)
            elif predict_length is not None:
                masks = np.ones(self.samples.shape)
                masks[:, -predict_length:, :] = 0
                self.masking = masks.astype(bool)
            else:
                raise NotImplementedError()
        self.sample_num = self.samples.shape[0]

    def __getsamples(self, data, text, text_num, proportion, seed):
        x = np.zeros((self.sample_num_total, self.window, self.var_num))
        for i in range(self.sample_num_total):
            start = i
            end = i + self.window
            x[i, :, :] = data[start:end, :]

        train_data, test_data, train_text, test_text, train_text_num, test_text_num = self.divide(x, text, text_num,
                                                                                                  proportion, seed)

        if self.save2npy:
            if 1 - proportion > 0:
                np.save(os.path.join(self.dir, f"{self.name}_ground_truth_{self.window}_test.npy"), self.unnormalize(test_data))
            np.save(os.path.join(self.dir, f"{self.name}_ground_truth_{self.window}_train.npy"), self.unnormalize(train_data))
            if self.auto_norm:
                if 1 - proportion > 0:
                    np.save(os.path.join(self.dir, f"{self.name}_norm_truth_{self.window}_test.npy"), unnormalize_to_zero_to_one(test_data))
                np.save(os.path.join(self.dir, f"{self.name}_norm_truth_{self.window}_train.npy"), unnormalize_to_zero_to_one(train_data))
            else:
                if 1 - proportion > 0:
                    np.save(os.path.join(self.dir, f"{self.name}_norm_truth_{self.window}_test.npy"), test_data)
                np.save(os.path.join(self.dir, f"{self.name}_norm_truth_{self.window}_train.npy"), train_data)

        return train_data, test_data, train_text, test_text, train_text_num, test_text_num

    def normalize(self, sq):
        d = sq.reshape(-1, self.var_num)
        d = self.scaler.transform(d)
        if self.auto_norm:
            d = normalize_to_neg_one_to_one(d)
        return d.reshape(-1, self.window, self.var_num)

    def unnormalize(self, sq):
        d = self.__unnormalize(sq.reshape(-1, self.var_num))
        return d.reshape(-1, self.window, self.var_num)

    def __normalize(self, rawdata):
        if rawdata.ndim == 1:
            rawdata = rawdata.reshape(-1, 1)
        data = self.scaler.transform(rawdata)
        if self.auto_norm:
            data = normalize_to_neg_one_to_one(data)
        return data

    def __unnormalize(self, data):
        if self.auto_norm:
            data = unnormalize_to_zero_to_one(data)
        x = data
        return self.scaler.inverse_transform(x)
    
    @staticmethod
    def divide(data, text, text_num, ratio, seed=2023):
        if not (data.shape[0] == len(text) == len(text_num)):
            raise ValueError("All inputs must have the same number of rows.")

        size = data.shape[0]
        # Store the state of the RNG to restore later.
        st0 = np.random.get_state()
        np.random.seed(seed)
        regular_train_num = int(np.ceil(size * ratio))
        id_rdm = np.random.permutation(size)
        regular_train_id = id_rdm[:regular_train_num]
        irregular_train_id = id_rdm[regular_train_num:]
        regular_data = data[regular_train_id, :]
        irregular_data = data[irregular_train_id, :]

        # Split text and text_num using the same indices
        regular_text = [text[i] for i in regular_train_id]
        irregular_text = [text[i] for i in irregular_train_id]
        regular_text_num = [text_num[i] for i in regular_train_id]
        irregular_text_num = [text_num[i] for i in irregular_train_id]

        # Restore RNG.
        np.random.set_state(st0)

        return regular_data, irregular_data, regular_text, irregular_text, regular_text_num, irregular_text_num


    @staticmethod
    def read_data(json_root_data, tokenizer, max_length, name=''):
        trend_texts = [sample.get('Trend Analysis', '') for sample in json_root_data]
        time_series = [value for i, sample in enumerate(json_root_data)
                       for value in (sample['sampled_time_series'] if i == 0 else [sample['sampled_time_series'][-1]])]
        time_series = np.array(time_series).reshape(-1, 1)
        scaler = MinMaxScaler().fit(time_series)
        trend_inputs, trend_inputs_num = zip(
            *map(lambda trend_text: tokenize_fnc(trend_text, tokenizer, scaler), trend_texts))
        trend_inputs, trend_inputs_num = list(trend_inputs), list(trend_inputs_num)
        text = trend_inputs
        text_num = trend_inputs_num

        return time_series, scaler, text, text_num
    
    def mask_data(self, seed=2023):
        masks = np.ones_like(self.samples)
        # Store the state of the RNG to restore later.
        st0 = np.random.get_state()
        np.random.seed(seed)

        for idx in range(self.samples.shape[0]):
            x = self.samples[idx, :, :]  # (seq_length, feat_dim) array
            mask = noise_mask(x, self.missing_ratio, self.mean_mask_length, self.style,
                              self.distribution)  # (seq_length, feat_dim) boolean array
            masks[idx, :, :] = mask

        if self.save2npy:
            np.save(os.path.join(self.dir, f"{self.name}_masking_{self.window}.npy"), masks)

        # Restore RNG.
        np.random.set_state(st0)
        return masks.astype(bool)

    def __getitem__(self, ind):
        if self.period == 'test':
            x = self.samples[ind, :, :]  # (seq_length, feat_dim) array
            m = self.masking[ind, :, :]  # (seq_length, feat_dim) boolean array
            return torch.from_numpy(x).float(), torch.from_numpy(m)
        x = self.samples[ind, :, :]  # (seq_length, feat_dim) array
        text = self.text[ind]
        text_num = self.text_num[ind]
        return text, text_num, torch.from_numpy(x).float()


    def __len__(self):
        return self.sample_num
    

class fMRIDataset(CustomDataset):
    def __init__(
        self, 
        proportion=1., 
        **kwargs
    ):
        super().__init__(proportion=proportion, **kwargs)

    @staticmethod
    def read_data(filepath, name=''):
        """Reads a single .csv
        """
        data = io.loadmat(filepath + '/sim4.mat')['ts']
        scaler = MinMaxScaler()
        scaler = scaler.fit(data)
        return data, scaler
