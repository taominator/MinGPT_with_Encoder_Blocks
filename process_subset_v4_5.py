import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import logging
from filelock import FileLock
from concurrent.futures import ProcessPoolExecutor
from mingpt.bpe import BPETokenizer

logging.basicConfig(level=logging.INFO)


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
        logging.error(f"Error during tokenization: {e}")


def save_chunk(dataset_chunk, save_path, chunk_count, max_retries=3):
    file_path = os.path.join(save_path, f'chunk_{chunk_count}.pth')
    retries = 0
    while retries < max_retries:
        try:
            with FileLock(file_path + ".lock"):
                torch.save(dataset_chunk, file_path)
            logging.info(f"Saved chunk {chunk_count} at {file_path}")
            return
        except Exception as e:
            retries += 1
            logging.warning(f"Error saving chunk {chunk_count} at {file_path} on attempt {retries}. Error: {e}")
    logging.error(f"Failed to save chunk {chunk_count} at {file_path} after {max_retries} attempts.")


def process_texts(encoder, save_path, chunk_size, texts):
    x_list, y_list, chunk_count = [], [], 0
    try:
        for text in texts:
            for x_data, y_data in tokenize_generator(text, encoder):
                x_list.append(x_data)
                y_list.append(y_data)
                if len(x_list) >= chunk_size:
                    dataset_chunk = {
                        'data': torch.tensor(np.stack(x_list, axis=0)),
                        'labels': torch.tensor(np.stack(y_list, axis=0))
                    }
                    save_chunk(dataset_chunk, save_path, chunk_count)
                    x_list, y_list = [], []
                    chunk_count += 1
        return chunk_count
    except Exception as e:
        logging.error(f"Error processing texts. Error: {e}")
        return 0


def process_subset(major_index_str, encoder, chunk_size=500000):
    base_folder = os.path.join(os.getcwd(), 'extracted_tar_openwebtext')
    dataset = TextFileDataset(base_folder, major_index_str)
    data_loader = DataLoader(dataset, batch_size=chunk_size, shuffle=True, num_workers=os.cpu_count())

    current_directory = os.getcwd()
    save_path = os.path.join(current_directory, 'datasets', f'subset_{major_index_str}')
    os.makedirs(save_path, exist_ok=True)

    total_chunks = 0
    with ProcessPoolExecutor() as executor:
        for batch_texts in data_loader:
            total_chunks += executor.submit(process_texts, encoder, save_path, chunk_size, batch_texts).result()

    logging.info(f"Processing completed for subset {major_index_str} with {total_chunks} chunks!")


if __name__ == '__main__':
    encoder = BPETokenizer().encoder

    #import argparse

    #parser = argparse.ArgumentParser(description='Process OpenWebText subset.')
    #parser.add_argument('subset', type=str, help='The subset string, e.g., "01".')
    #parser.add_argument('--chunk-size', type=int, default=500000, help='Size of data chunk.')

    #args = parser.parse_args()

    #process_subset(args.subset, encoder, chunk_size=args.chunk_size)

    process_subset('0', encoder, chunk_size=500000)