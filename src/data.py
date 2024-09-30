# Loading data(Data loader)
import torch
import datasets
import pytorch_lightninig as pl


from datasets import load_dataset
from transformers import AutoTokenizer




class DataModule(pl.LightniningDataModue): 
    def __init__self(self,model_name='google/bert_uncased_L-2_H-128_A-2', batch_size=32):
        super().__init__()
        
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)


    def prepare_data(self):
        cola_dataset = load_dataset("glue","cola")
        self.train_data = cola_dataset['train']
        self.val_data = cola_dataset['validation']


    def tokenizer_data(self, example):
        return self.tokenizer(
            example['sentence'],
            truncation = True,
            padding ="max_length",
            max_length = '512'
        )

    def setup(self, stage=None):

        if stage == "fit" or stage is None:
            self.train_data = self.train_data.map(self.tokenizer_data,batched=True)
            self.train_data.set_format(
                type = "torch", columns=['input_ids','attention_masks','label']
            )


            self.val_data = self.val_data.map(self.tokenizer_data,batched=True)
            self.val_data.set_format(
                type = "torch", columns=['input_ids','attention_masks','label']
            )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_data,batch_size=self.batch_size,shuffel = True
        )
        

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_data,batch_size=self.batch_size,shuffel = True
        )




if __name__ == "__main__":
    data_model = DataModule()
    data_model.prepare_data()
    data_model.setup()
    print(next(iter(data_model.train_dataloader()))["input_ids"].shape)
