import hydra
import torch
import logging


from omegaconf.omegaconf import OmegaConf

from model import ColaModel
from data import DataModule


logger = logging.getLogger(__name__)




@hydra.main("../configs",config_name="config.yaml")
def convert_model(cfg):
    root_dir = hydra.utils.get_original_cwd()
    model_path = f"{root_dir}/models/best-checkpoint-v4.ckpt"
    logger.info(f"Loading pre-trained model from:{model_path}")
    cola_model = ColaModel.load_from_checkpoint(model_path)

    data_module = DataModule(cfg.model.tokenizer, cfg.processing.batch_size, cfg.processing.max_length)


    data_module.prepare_data()
    data_module.setup()

    input_batch = next(iter(data_module.train_dataloader()))
    input_sample = {
        "input_ids": input_batch["input_ids"][0].unsqueeze(0).to(torch.int32),
        "attention_mask": input_batch["attention_mask"][0].unsqueeze(0).to(torch.int32),
    }

    #Export the model
    logger.info(f"Converting the model into ONNX format")
    torch.onnx.export(
        cola_model,
        (
        (input_sample["input_ids"]),
        input_sample['attention_mask']
    ),
    f"{root_dir}/models/model.onnx",
    export_params = True,
    opset_version = 14,
    input_names = ["input_ids","attention_mask"],
    output_names = ["outputs"],
     dynamic_axes={
            "input_ids": {0: "batch_size"},  # variable length axes
            "attention_mask": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
                     )

    logger.info(f"Model successfully converted, ONNX model format is at {root_dir}/models/model.onnx")


if __name__ =="__main__":
    convert_model()