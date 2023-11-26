# A code template for training DNN-based speech enhancement models.
A training code template is highly valuable for deep learning engineers as it can significantly enhance their work efficiency. Despite different individuals have varying coding styles, some are excellent while others may not be as good. My philosophy is to prioritize simplicity. In this context, I am sharing a practical organizational structure for training code files in speech enhancement (SE). The primary focus is on keeping it concise and intuitive rather than aiming for comprehensiveness.

## File Specification
For training:
* `config.toml`: Specifies the training configurations.
* `datasets.py`: Provides the dataset class for the dataloader.
* `distributed_utils.py`: Assists with Distributed Data Parallel (DDP) training.
* `model.py`: Defines the model.
* `loss_factory.py`: Provides various useful loss functions in SE.
* `train_sg.py`: Conducts the training process for a single GPU machine.
* `train.py`: Conducts the training process for multiple GPUs.
* `trainer_sg.py`: Encapsulates various functions during training for a single GPU machine.
* `trainer.py`: Encapsulates various functions during training for multiple GPUs.

For evaluation:
* `config.yaml`: Specifies evaluation paths.
* `infer_folder.py`: Conducts evaluation on a folder of WAV files.
* `infer_loader.py`: Conducts evaluation using a dataloader.
* `score_utils.py`: Provides calculations for various metrics.

## Usage
When starting a new SE project, you should follow these steps:
1. Modify `datasets.py`;
2. Define your own `model.py`;
3. Modify the `config.toml` to match your training setup;
4. Select a loss function in `loss_factory.py`, or create a new one if needed;
5. Probably do not need to modify `trainer.py` or `trainer_sg.py`;
6. Run the `train.py` or `train_sg.py` based on the number of available GPUs.
7. Before evaluation, remember to modify `config.yaml` to ensure that the paths are correctly configured.

## Note
The code is originally intended for Linux systems, and if you attempt to adapt it to the Windows platform, you may encounter certain issues:
* Incompatibility of paths: The file paths used in Linux systems may not be compatible with the file paths in Windows.

* Challenges in installing the pesq package: The process of installing the pesq package on Windows may not be straightforward and may require additional steps or configurations.

Please keep these considerations in mind when working with the code on the Windows platform.

## Acknowledgement
This code template heavily borrows from the excellent [Sheffield_Clarity_CEC1_Entry](https://github.com/TuZehai/Sheffield_Clarity_CEC1_Entry) reposity in many aspects.