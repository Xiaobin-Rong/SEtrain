# A code template for training DNN-based speech enhancement models.
A training code template is highly valuable for deep learning engineers as it can significantly enhance their work efficiency. Despite different programmers have varying coding styles, some are excellent while others may not be as good. My philosophy is to prioritize simplicity. In this context, I am sharing a practical organizational structure for training code files in speech enhancement (SE). The primary focus is on keeping it concise and intuitive rather than aiming for comprehensiveness.

## File Specification
For training:
* `cfg_train.toml`: Specifies the training configurations.
* `datasets.py`: Provides the dataset class for the dataloader.
* `distributed_utils.py`: Assists with Distributed Data Parallel (DDP) training.
* `loss_factory.py`: Provides various useful loss functions in SE.
* `model.py`: Defines the model.
* `train.py`: Conducts the training process, surpports both multiple-GPU and single-GPU conditions.
* `trainer.py`: Encapsulates various functions during training, surpports both multiple-GPU and single-GPU conditions.

For evaluation:
* `cfg_infer.yaml`: Specifies the evaluation configurations.
* `infer_folder.py`: Conducts evaluation on a folder of WAV files.
* `infer_loader.py`: Conducts evaluation using a dataloader.
* `score_utils.py`: Provides calculations for various metrics.

## Usage
When starting a new SE project, you should follow these steps:
1. Modify `datasets.py`;
2. Define your own `model.py`;
3. Modify the `config.toml` to match your training setup;
4. Select a loss function in `loss_factory.py`, or create a new one if needed;
5. Probably do not need to modify `trainer.py`;
6. Run the `train.py`:
   ```
   python train.py
   python train.py -D 1
   python train.py -C cfg_train.toml -D 1
   python train.py -C cfg_train.toml -D 0,1,2,3
   ```
8. Before evaluation, remember to modify `cfg_infer.yaml` to ensure that the paths are correctly configured.

## Note
1. The code is originally intended for Linux systems, and if you attempt to adapt it to the Windows platform, you may encounter certain issues:
* Incompatibility of paths: The file paths used in Linux systems may not be compatible with the file paths in Windows.
* Challenges in installing the pesq package: The process of installing the pesq package on Windows may not be straightforward and may require additional steps or configurations.

2. The code is merely provided as a template, and some negligible details are not included in the repository, such as the `INFO.csv` in  `datasets.py` and `DNSMOS` in the `infer_folder`/`infer_loader.py`.

## Acknowledgement
This code template heavily borrows from the excellent [Sheffield_Clarity_CEC1_Entry](https://github.com/TuZehai/Sheffield_Clarity_CEC1_Entry) repository in many aspects.
