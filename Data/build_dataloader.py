import torch
from Utils.io_utils import instantiate_from_config
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(r"././robertabase", local_files_only=True)

def collate_fn(batch):
    texts, text_nums, datas = zip(*batch)
    # 处理数据的填充或裁剪
    # 将 tuple 转换为 tensor
    texts_tensors = [torch.tensor(t) for t in texts]
    text_nums_tensors = [torch.tensor(tn) for tn in text_nums]
    datas_tensors = [torch.tensor(d) for d in datas]

    # 使用 pad_sequence 进行填充
    texts_padded = pad_sequence(texts_tensors, batch_first=True)
    text_nums_padded = pad_sequence(text_nums_tensors, batch_first=True)
    datas_padded = pad_sequence(datas_tensors, batch_first=True)
    return texts_padded, text_nums_padded, datas_padded

def build_dataloader(config, args=None):
    batch_size = config['dataloader']['batch_size']
    jud = config['dataloader']['shuffle']
    config['dataloader']['train_dataset']['params']['output_dir'] = args.save_dir
    dataset = instantiate_from_config(config['dataloader']['train_dataset'])

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=jud,
                                             num_workers=0,
                                             pin_memory=True,
                                             sampler=None,
                                             drop_last=jud, collate_fn=collate_fn)

    dataload_info = {
        'dataloader': dataloader,
        'dataset': dataset
    }

    return dataload_info

def build_dataloader_cond(config, args=None):
    batch_size = config['dataloader']['sample_size']
    config['dataloader']['test_dataset']['params']['output_dir'] = args.save_dir
    if args.mode == 'infill':
        config['dataloader']['test_dataset']['params']['missing_ratio'] = args.missing_ratio
    elif args.mode == 'predict':
        config['dataloader']['test_dataset']['params']['predict_length'] = args.pred_len
    test_dataset = instantiate_from_config(config['dataloader']['test_dataset'])

    dataloader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=0,
                                             pin_memory=True,
                                             sampler=None,
                                             drop_last=False)

    dataload_info = {
        'dataloader': dataloader,
        'dataset': test_dataset
    }

    return dataload_info


if __name__ == '__main__':
    pass

