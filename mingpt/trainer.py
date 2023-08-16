"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""

import time
from collections import defaultdict

import torch
from torch.utils.data.dataloader import DataLoader
from torch.nn.utils.rnn import pad_sequence
from mingpt.utils import CfgNode as CN

class Trainer:

    @staticmethod
    def get_default_config():
        C = CN()
        # device to train on
        C.device = 'auto'
        # dataloder parameters
        C.num_workers = 4
        # optimizer parameters
        C.max_iters = None
        C.batch_size = 64
        C.learning_rate = 3e-4
        C.betas = (0.9, 0.95)
        C.weight_decay = 0.1 # only applied on matmul weights
        C.grad_norm_clip = 1.0
        return C

    def __init__(self, config, model, train_dataset):
        self.config = config
        self.model = model
        self.optimizer = None
        self.train_dataset = train_dataset
        self.callbacks = defaultdict(list)

        # determine the device we'll train on
        if config.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = config.device
        self.model = self.model.to(self.device)
        print("running on device", self.device)

        # variables that will be assigned to trainer class later for logging and etc
        self.iter_num = 0
        self.iter_time = 0.0
        self.iter_dt = 0.0

    def add_callback(self, onevent: str, callback):
        self.callbacks[onevent].append(callback)

    def set_callback(self, onevent: str, callback):
        self.callbacks[onevent] = [callback]

    def trigger_callbacks(self, onevent: str):
        for callback in self.callbacks.get(onevent, []):
            callback(self)


    def custom_collate_fn(batch):
        # Separate data and labels
        data, labels = zip(*batch)

        # Calculate the lengths of each sequence in the batch
        data_lengths = [len(seq) for seq in data]

        # Pad the data sequences
        data_padded = pad_sequence(data, batch_first=True, padding_value=50256)

        # Pad the label sequences
        labels_padded = pad_sequence(labels, batch_first=True, padding_value=-1)

        # Create attention masks for data based on the original lengths
        attention_masks = []
        for idx, seq_len in enumerate(data_lengths):
            mask = torch.ones(data_padded.size(1), dtype=torch.long)
            mask[seq_len:] = 0
            attention_masks.append(mask)
        attention_masks = torch.stack(attention_masks)

        return data_padded, labels_padded, attention_masks
    

    def run(self):
        model, config = self.model, self.config

        # setup the optimizer
        self.optimizer = model.configure_optimizers(config)

        # setup the dataloader for the iterable dataset
        train_loader = DataLoader(
            self.train_dataset,
            shuffle=False,
            pin_memory=True,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            collate_fn=Trainer.custom_collate_fn
        )

        model.train()
        self.iter_num = 0
        self.iter_time = time.time()
        data_iter = iter(train_loader)
        while True:
            try:
                batch = next(data_iter)
                batch = [t.to(self.device) for t in batch]
                x, y, attention_mask = batch

                # forward the model, now including the attention mask
                logits, self.loss = model(x, y, attention_mask=attention_mask)

                # backprop and update the parameters
                model.zero_grad(set_to_none=True)
                self.loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                self.optimizer.step()

                # Increment the iteration number
                self.iter_num += 1

                print(f'iteration {self.iter_num}:')
                print(f'x: \n{x}')
                print(f'y: \n{y}')
                print(f'attention_mask: \n{attention_mask}')

                # Print the current loss and time taken every 1000 iterations
                if self.iter_num % 1 == 0:
                    tnow = time.time()
                    self.iter_dt = tnow - self.iter_time
                    print(f"Iteration {self.iter_num}: Loss = {self.loss.item()}, Time Taken: {self.iter_dt:.2f} seconds")
                    self.iter_time = tnow

                self.trigger_callbacks('on_batch_end')

            except StopIteration:
                # Dataset has been exhausted
                break
