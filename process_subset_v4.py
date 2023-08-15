from mingpt.bpe import BPETokenizer
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from multiprocessing import Process, Queue, cpu_count
from queue import Empty
import hashlib
import time
from filelock import FileLock


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
    try:
        tokenized_text = encoder.encode(text)
        total_length = len(tokenized_text)
        
        # If the total length is less than the context length, we simply start from one token and increase.
        if total_length < context_length:
            for i in range(1, total_length):
                yield np.array(tokenized_text[:i], dtype=np.long), np.array(tokenized_text[i], dtype=np.long)
        else:
            # Here, we start from one token and keep increasing the window size until it's context_length-1
            for end_idx in range(1, context_length):
                yield np.array(tokenized_text[:end_idx], dtype=np.long), np.array(tokenized_text[end_idx], dtype=np.long)
            
            # Now, we slide the window one token at a time until the end of the text.
            for start_idx in range(1, total_length - context_length + 1):
                end_idx = start_idx + context_length
                yield np.array(tokenized_text[start_idx:end_idx-1], dtype=np.long), np.array(tokenized_text[end_idx-1], dtype=np.long)
                
    except Exception as e:
        print(f"Error during tokenization: {e}")



def producer_task(dataset, queue, start_idx, end_idx):
    try:
        for idx in range(start_idx, end_idx):
            text = dataset[idx]
            queue.put(text)
    except Exception as e:
        print(f"Error in producer task: {e}")
    finally:
        queue.put(None)


def consumer_task(queue, encoder, save_path, chunk_size=500000, max_retries=3):
    x_list, y_list, chunk_count = [], [], 0
    failure_count = 0  # Counter for the number of failures

    def compute_file_hash(file_path):
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for block in iter(lambda: f.read(4096), b''):
                sha256.update(block)
        return sha256.hexdigest()

    def save_and_verify(dataset_chunk, chunk_name):
        nonlocal failure_count
        max_retries = 3
        backoff_factor = 2
        wait_time = 1  # Initial wait time in seconds
        
        for retry_count in range(max_retries):
            temp_path = os.path.join(save_path, f'{chunk_name}_temp.pth')
            final_path = os.path.join(save_path, f'{chunk_name}.pth')
            
            # Locking the file before writing
            with FileLock(temp_path + ".lock"):
                # Save the dataset chunk
                torch.save(dataset_chunk, temp_path)
                
                # Calculate the checksum of the written file using the optimized function
                written_file_hash = compute_file_hash(temp_path)
        
                # Calculate the checksum of the data in memory
                data_hash = hashlib.sha256(repr(dataset_chunk).encode()).hexdigest()
                
                # Check if saved file's hash matches the in-memory data's hash
                if written_file_hash == data_hash:
                    os.rename(temp_path, final_path)  # Atomic move
                    return True
                else:
                    os.remove(temp_path)
                    failure_count += 1  # Increment failure count
                    if retry_count < max_retries - 1:
                        time.sleep(wait_time)
                        wait_time *= backoff_factor
                    else:
                        print(f"Failed to save and verify {chunk_name} after {max_retries} attempts due to checksum mismatch.")
                        return False

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
                    if not save_and_verify(dataset_chunk, f'chunk_{chunk_count}'):
                        print(f"Chunk {chunk_count} could not be saved after {max_retries} attempts!")
                    x_list, y_list = [], []
                    chunk_count += 1

        except Empty:
            continue
        except Exception as e:
            print(f"Error in consumer task: {e}")

    if x_list:
        dataset_chunk = {'data': [torch.tensor(x) for x in x_list], 'labels': [torch.tensor(y) for y in y_list]}
        save_and_verify(dataset_chunk, f'chunk_{chunk_count}')

    print(f"Total save failures encountered: {failure_count}")


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

    for _ in range(num_consumers):
        queue.put(None)  # Send termination signals equal to the number of consumers

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
