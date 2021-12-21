import torch

class SimpleCustomBatch:
    def __init__(self, data):
        transposed_data = list(zip(*data))
        self.input_ids = torch.stack(transposed_data[0], 0)
        self.token_type_ids = torch.stack(transposed_data[1], 0)
        self.attention_mask = torch.stack(transposed_data[2], 0)
        self.start_positions = torch.stack(transposed_data[3], 0)
        self.end_positions = torch.stack(transposed_data[4], 0)

    def pin_memory(self):
        self.input_ids = self.input_ids.pin_memory()
        self.token_type_ids = self.token_type_ids.pin_memory()
        self.attention_mask = self.attention_mask.pin_memory()
        self.start_positions = self.start_positions.pin_memory()
        self.end_positions = self.end_positions.pin_memory()
        return self

def collate_wrapper(batch):
    return SimpleCustomBatch(batch)