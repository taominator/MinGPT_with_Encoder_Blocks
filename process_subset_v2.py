import os
import numpy as np
import torch
from torch.utils.data import Dataset
from multiprocessing import Process, Queue, cpu_count
from queue import Empty
from mingpt.bpe import BPETokenizer


class TextFileDataset(Dataset):
    def __init__(self, base_folder, subset_str):
        self.file_list = []
        for folder in sorted(os.listdir(base_folder)):
            if not folder.startswith(f'urlsf_subset{subset_str.zfill(2)}'):
                continue
            full_path = os.path.join(base_folder, folder)
            self.file_list.extend([os.path.join(full_path, f) for f in os.listdir(full_path) if f.endswith('.txt')])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        with open(self.file_list[idx], 'r', encoding='utf-8') as file:
            return file.read()

def tokenize_generator(text, encoder, context_length=1024):
    tokenized_text = encoder.encode(text)
    if len(tokenized_text) >= context_length:
        for start_idx in range(0, len(tokenized_text) - context_length + 1, context_length):
            chunk = tokenized_text[start_idx:start_idx + context_length]
            for i in range(1, len(chunk)):
                yield np.array(chunk[:i], dtype=np.long), np.array(chunk[i], dtype=np.long)
    else:
        for i in range(1, len(tokenized_text)):
            yield np.array(tokenized_text[:i], dtype=np.long), np.array(tokenized_text[i], dtype=np.long)

def producer_task(dataset, queue, start_idx, end_idx):
    for idx in range(start_idx, end_idx):
        text = dataset[idx]
        queue.put(text)
    queue.put(None)

def consumer_task(queue, encoder, save_path, chunk_size=500000):
    x_list, y_list, chunk_count = [], [], 0
    while True:
        try:
            text = queue.get(timeout=10)
            if text is None:
                break
            for x_data, y_data in tokenize_generator(text, encoder):
                x_list.append(x_data)
                y_list.append(y_data)
                if len(x_list) >= chunk_size:
                    dataset_chunk = {'data': [torch.tensor(x) for x in x_list], 'labels': [torch.tensor(y) for y in y_list]}
                    torch.save(dataset_chunk, os.path.join(save_path, f'chunk_{chunk_count}.pth'))
                    x_list, y_list = [], []
                    chunk_count += 1

        except Empty:
            continue

    if x_list:
        dataset_chunk = {'data': [torch.tensor(x) for x in x_list], 'labels': [torch.tensor(y) for y in y_list]}
        torch.save(dataset_chunk, os.path.join(save_path, f'chunk_{chunk_count}.pth'))

def process_subset(major_index_str, encoder, chunk_size=500000):
    base_folder = 'D:/Machine_Learning/MinGPT/extracted_tar_openwebtext'
    dataset = TextFileDataset(base_folder, major_index_str)
    
    num_workers = cpu_count()
    num_producers = num_workers // 2
    num_consumers = num_workers - num_producers
    
    items_per_producer = len(dataset) // num_producers
    
    queue = Queue(maxsize=1000)
    
    producers = []
    for i in range(num_producers):
        start_idx = i * items_per_producer
        end_idx = start_idx + items_per_producer if i != num_producers - 1 else len(dataset)
        p = Process(target=producer_task, args=(dataset, queue, start_idx, end_idx))
        producers.append(p)
        p.start()
    
    current_directory = os.getcwd()
    save_path = os.path.join(current_directory, 'datasets', f'subset_{major_index_str}')
    os.makedirs(save_path, exist_ok=True)
    
    consumers = []
    for _ in range(num_consumers):
        c = Process(target=consumer_task, args=(queue, encoder, save_path, chunk_size))
        consumers.append(c)
        c.start()

    for p in producers:
        p.join()

    for c in consumers:
        c.join()

    print(f"Processing completed for subset {major_index_str}!")


if __name__ == '__main__':
    encoder = BPETokenizer().encoder

    import argparse

    parser = argparse.ArgumentParser(description='Process OpenWebText subset.')
    parser.add_argument('subset', type=str, help='The subset string, e.g., "01".')

    args = parser.parse_args()

    process_subset(args.subset, encoder)
