from mingpt.bpe import BPETokenizer
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from multiprocessing import Process, Queue, cpu_count, Value, Lock
from queue import Empty
import time
from filelock import FileLock
import shutil  # Added for shutil.move

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
        stride = context_length // 8

        # For very short texts
        if total_length <= stride:
            yield np.array(tokenized_text[:-1], dtype=np.long), np.array(tokenized_text[-1], dtype=np.long)
            
        # For texts longer than the stride but shorter than the context length
        elif total_length <= context_length:
            for end_idx in range(stride, total_length + 1, stride):
                yield np.array(tokenized_text[:end_idx-1], dtype=np.long), np.array(tokenized_text[end_idx-1], dtype=np.long)
            
            # If there are any tokens left at the end that couldn't fit a full stride window
            if (total_length - 1) % stride != 0:
                yield np.array(tokenized_text[-stride:-1], dtype=np.long), np.array(tokenized_text[-1], dtype=np.long)
                
        # For longer texts
        else:
            # Gradually increase the window size until it's the size of the context length
            for end_idx in range(stride, context_length + 1, stride):
                yield np.array(tokenized_text[:end_idx-1], dtype=np.long), np.array(tokenized_text[end_idx-1], dtype=np.long)
                
            # Now slide the window by the stride for the rest of the text
            for start_idx in range(stride, total_length - context_length + 1, stride):
                end_idx = start_idx + context_length
                yield np.array(tokenized_text[start_idx:end_idx-1], dtype=np.long), np.array(tokenized_text[end_idx-1], dtype=np.long)

            # If there are any tokens left at the end that couldn't fit a full context_length window
            if end_idx < total_length:
                yield np.array(tokenized_text[-context_length:-1], dtype=np.long), np.array(tokenized_text[-1], dtype=np.long)
                
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

def consumer_task(consumer_id, queue, encoder, save_path, chunk_counter, counter_lock, chunk_size=500000, max_retries=3):
    x_list, y_list = [], []

    def save_dataset_chunk(dataset_chunk):
        nonlocal x_list, y_list
        with counter_lock:  # Ensure exclusive access to the shared counter
            chunk_num = chunk_counter.value
            chunk_counter.value += 1
        chunk_name = f'chunk_{chunk_num}'
        temp_path = os.path.join(save_path, f'{chunk_name}_temp.pth')
        final_path = os.path.join(save_path, f'{chunk_name}.pth')
        backoff_factor = 2
        wait_time = 1  # Initial wait time in seconds
        
        for retry_count in range(max_retries):
            with FileLock(temp_path + ".lock"):
                try:
                    torch.save(dataset_chunk, temp_path)
                    shutil.move(temp_path, final_path)
                    x_list, y_list = [], []  # Clear the lists
                    return True
                except Exception as e:
                    print(f"Error saving chunk: {e}")
                    os.remove(temp_path)
                    if retry_count < max_retries - 1:
                        time.sleep(wait_time)
                        wait_time *= backoff_factor
                    else:
                        print(f"Failed to save {chunk_name} after {max_retries} attempts.")
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
                    save_dataset_chunk(dataset_chunk)
        except Empty:
            continue
        except Exception as e:
            print(f"Error in consumer task: {e}")

    if x_list:
        dataset_chunk = {'data': [torch.tensor(x) for x in x_list], 'labels': [torch.tensor(y) for y in y_list]}
        save_dataset_chunk(dataset_chunk)

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

    # Create a shared counter and a lock
    chunk_counter = Value('i', 0)
    counter_lock = Lock()

    consumers = []
    for i in range(num_consumers):
        c = Process(target=consumer_task, args=(i, queue, encoder, save_path, chunk_counter, counter_lock, chunk_size))
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
