import os
import torch
from torch.utils.data import Dataset
from multiprocessing import Process, Queue, cpu_count
from queue import Empty

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

def tokenize_and_process(text, encoder, context_length=1024):
    tokenized_text = encoder.encode(text)
    x_list, y_list = [], []
    if len(tokenized_text) >= context_length:
        for start_idx in range(0, len(tokenized_text) - context_length + 1, context_length):
            chunk = tokenized_text[start_idx:start_idx + context_length]
            for i in range(1, len(chunk)):
                x_list.append(torch.tensor(chunk[:i], dtype=torch.long))
                y_list.append(torch.tensor(chunk[i], dtype=torch.long))
    else:
        for i in range(1, len(tokenized_text)):
            x_list.append(torch.tensor(tokenized_text[:i], dtype=torch.long))
            y_list.append(torch.tensor(tokenized_text[i], dtype=torch.long))
    return x_list, y_list

def producer_task(dataset, queue, start_idx, end_idx):
    for idx in range(start_idx, end_idx):
        text = dataset[idx]
        queue.put(text)
        if (idx - start_idx) % 100 == 0:  # Print every 100 processed files
            print(f"Producer processed {idx - start_idx + 1} files out of {end_idx - start_idx}")
    queue.put(None)  # Sentinel value indicating this producer is done
    print("Producer task completed.")

def consumer_task(queue, encoder, save_path, chunk_size=500000):
    x_list, y_list, chunk_count = [], [], 0
    processed_files = 0
    while True:
        try:
            text = queue.get(timeout=10)
            if text is None:
                break
            x_data, y_data = tokenize_and_process(text, encoder)
            x_list.extend(x_data)
            y_list.extend(y_data)
            processed_files += 1
            if processed_files % 100 == 0:  # Print every 100 processed files
                print(f"Consumer processed {processed_files} files.")

            if len(x_list) >= chunk_size:
                dataset_chunk = {'data': x_list, 'labels': y_list}
                torch.save(dataset_chunk, os.path.join(save_path, f'chunk_{chunk_count}.pth'))
                x_list, y_list = [], []
                chunk_count += 1
                print(f"Saved chunk_{chunk_count}")

        except Empty:
            logging.warning("Queue is empty. Consumer task is waiting for data.")
            continue

    if x_list:
        dataset_chunk = {'data': x_list, 'labels': y_list}
        torch.save(dataset_chunk, os.path.join(save_path, f'chunk_{chunk_count}.pth'))
        print(f"Saved final chunk_{chunk_count}")

    print("Consumer task completed.")

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

    queue.close()  # Close the queue after all tasks are done

    print(f"Processing completed for subset {major_index_str}!")

# ----------------------------------------------------------------------------------------------------------------------

from mingpt.bpe import BPETokenizer

encoder = BPETokenizer().encoder

# ----------------------------------------------------------------------------------------------------------------------

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process OpenWebText subset.')
    parser.add_argument('subset', type=str, help='The subset string, e.g., "01".')

    args = parser.parse_args()

    process_subset(args.subset, encoder)

