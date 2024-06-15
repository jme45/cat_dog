import os
from datetime import datetime
from pathlib import Path
from typing import Tuple, Any

import pandas as pd
import torch
from ml_utils_jme45 import ml_utils
from torch import nn
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
import torchvision

from cat_dog_classifiers import classifiers

import parameters


def fit_model(
    model_type: str,
    n_epochs: int,
    output_path: str | Path = parameters.DEFAULT_RUNS_DIR,
    compile_model: bool = False,
    device="cpu",
    data_root_dir=None,  # if None, use default
    num_workers: int = 0,
    experiment_name: str = "test",
    print_progress_to_screen: bool = False,
    optimiser_class: str | Optimizer = "Adam",
    optimiser_kwargs: dict = {"lr": 1e-3},
    return_classifier: bool = False,
) -> Tuple[dict[str, list], dict[str, Any]]:
    """
    Fit a model of a particular type to an aircraft subset

    :param model_type: type of model, e.g. vit_b_16, effnet_b7
    :param n_epochs: number of epochs to train for
    :param output_path: path to save state dict and tensorboard files
    :param compile_model: whether to compile (apparently best on good GPUs)
    :param device: device to run on.
    :param data_root_dir: Root directory for the training/cv data
    :param num_workers: num workers for dataloader. 0 best on laptop.
    :param experiment_name:
    :param print_progress_to_screen:
    :param optimiser_class: e.g. "Adam" or "SGD"
    :param optimiser_kwargs: any arguments for the optimiser, e.g. "lr"
    :param return_classifier: If True, return the final classifier as an extra element in the list returned.
    :return:
    """
    # More workers are only useful if using CUDA (experimentally).
    # I won't ever have access to a Computer with more than one GPU,
    # so can cap number of workers at 2. Similarly pin_memory.
    if device == "cuda":
        num_workers = min(num_workers, 2)
        num_workers = min(num_workers, os.cpu_count())
        pin_memory = True
    else:
        num_workers = 0
        pin_memory = False

    # Set up output_path, consisting of experiment_name, etc.
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    output_path = Path(output_path) / experiment_name / model_type / timestamp

    classifier = classifiers.CatDogClassifier(
        model_type,
        ["cat", "dog"],
        load_classifier_pretrained_weights=False,
        data_augmentation_transforms=parameters.DATA_AUGMENTATION_TRANSFORMS,
    )

    # Compiling is allegedly useful on more powerful GPUs.
    if compile_model:
        classifier.model = torch.compile(classifier.model)

    # Calculate number of model parameters, mainly as general info.
    num_params = sum(torch.numel(param) for param in classifier.model.parameters())

    if data_root_dir is None:
        data_root_dir = parameters.DATA_ROOT_DIR
    else:
        data_root_dir = Path(data_root_dir)

    # Set up training and validation sets.
    train_set = torchvision.datasets.ImageFolder(
        root=data_root_dir / "train",
        transform=classifier.train_transform,
        target_transform=None,
    )
    val_set = torchvision.datasets.ImageFolder(
        root=data_root_dir / "cv",
        transform=classifier.predict_transform,
        target_transform=None,
    )

    train_dataloader = DataLoader(
        train_set,
        parameters.BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_dataloader = DataLoader(
        val_set,
        parameters.BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # Set up tensorboard logging.
    tensorboard_logger = ml_utils.TensorBoardLogger(True, root_dir=output_path)

    # Define a trainer, which will do the training on the model.
    trainer = ml_utils.ClassificationTrainer(
        model=classifier.model,
        train_dataloader=train_dataloader,
        test_dataloader=val_dataloader,
        optimiser_class=optimiser_class,
        optimiser_kwargs=optimiser_kwargs,
        loss_fn=nn.CrossEntropyLoss(),  # should really use nn.BCEWithLogitsLoss(), but would need to change output shape
        n_epochs=n_epochs,
        device=device,
        output_path=output_path,
        num_classes=classifier.num_classes,
        save_lowest_test_loss_model=True,
        save_final_model=True,
        tensorboard_logger=tensorboard_logger,
        disable_epoch_progress_bar=False,
        disable_within_epoch_progress_bar=False,
        print_progress_to_screen=print_progress_to_screen,
        state_dict_extractor=classifier.state_dict_extractor,
        trainable_parts=classifier.trainable_parts,
    )

    # Now run the training.
    all_results = trainer.train()

    # Obtain information about the model and training run.
    meta_info = dict(num_params=num_params, output_path=str(output_path))

    # List of items to return. If we want to also return the classifier, add it to the list.
    ret = [all_results, meta_info]
    if return_classifier:
        ret.append(classifier)

    return ret


if __name__ == "__main__":
    all_results, meta_info = fit_model(
        "trivial",
        2,
        data_root_dir="/Users/jonathan/programming/ml_work/custom_datasets/cats_dogs",
    )

    print("For run:")
    print(meta_info)
    print("\nOutput frame:")
    print(pd.DataFrame(all_results))
