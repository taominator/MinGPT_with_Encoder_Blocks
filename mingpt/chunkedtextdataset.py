import os
import torch
from torch.utils.data import IterableDataset
import random

class ChunkedTextDataset(IterableDataset):
    def __init__(self, major_index_str):
        current_directory = os.getcwd()
        self.base_path = os.path.join(current_directory, 'datasets')
        self.subset_folder = os.path.join(self.base_path, f'subset_{major_index_str}')
        
        # Figure out the number of chunks by counting files in the directory
        self.num_chunks = sum(1 for file in os.listdir(self.subset_folder) if file.startswith(f'chunk_') and file.endswith('.pth'))

    def __iter__(self):
        # Shuffle order of chunks
        chunk_indices = torch.randperm(self.num_chunks)
        for i in chunk_indices:
            chunk = torch.load(os.path.join(self.subset_folder, f'chunk_{i}.pth'))
    
            x_data = chunk['data']
            y_data = chunk['labels']
            
            # Using zip to pair x and y, then shuffling the pairs
            combined = list(zip(x_data, y_data))
            random.shuffle(combined)
            
            for x, y in combined:
                yield x, y