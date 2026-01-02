import os
import torch
from torch.utils.data import DataLoader
from models.vmunet.vmunet import VMUNet
from datasets.dataset import ISIC17Dataset
from utils import get_logger, set_seed, log_config_info
from engine import test_one_epoch
from configs.config_setting import setting_config
import warnings
warnings.filterwarnings("ignore")


def clean_state_dict(state_dict):
    """Remove keys like total_ops / total_params because VMUNet checkpoints include them."""
    remove_keys = [k for k in state_dict.keys() if "total_ops" in k or "total_params" in k]
    for k in remove_keys:
        state_dict.pop(k)
    return state_dict


def main(config):
    # ------------------- GPU Setup -------------------
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    set_seed(config.seed)
    torch.cuda.empty_cache()

    # ------------------- Setup paths -------------------
    log_dir = os.path.join(config.work_dir, "log")
    out_dir = os.path.join(config.work_dir, "outputs")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    logger = get_logger("test", log_dir)
    log_config_info(config, logger)

    # ------------------- Dataset -------------------
    val_dataset = ISIC17Dataset(config.data_path, config, train=False)
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=config.num_workers
    )

    # ------------------- Model -------------------
    cfg = config.model_config
    model = VMUNet(
        num_classes=cfg["num_classes"],
        input_channels=cfg["input_channels"],
        depths=cfg["depths"],
        depths_decoder=cfg["depths_decoder"],
        drop_path_rate=cfg["drop_path_rate"],
        load_ckpt_path=None
    ).to(device)

    # برای مدل VM-UNet ضروری است
    model.load_from()
    model.eval()

    # ------------------- Load Weights -------------------
    ckpt_path = config.best_ckpt_path
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Weight not found: {ckpt_path}")
    print(f"Loading pretrained model weights from:\n {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    checkpoint = clean_state_dict(checkpoint)
    model.load_state_dict(checkpoint, strict=False)

    # ------------------- Loss -------------------
    criterion = config.criterion

    # ------------------- Testing -------------------
    print("Running inference on validation set...")
    loss = test_one_epoch(
        val_loader,
        model,
        criterion,
        logger,
        config,
        test_data_name="ISIC_val"
    )

    print(f"\nFinal Validation Loss = {loss:.4f}")


if __name__ == "__main__":
    config = setting_config
    main(config)
