from box import Box

config = {
    "num_devices": 1,
    "batch_size": 2,
    "num_workers": 4,
    "num_epochs": 3,
    "eval_interval": 1,
    "out_dir": "./runs/train",
    "opt": {
        "learning_rate": 1e-3,  # 1e-3
        "weight_decay": 1e-4,
        "decay_factor": 10,
        "steps": [2000, 4000],
        "warmup_steps": 400,
    },
    "model": {
        "type": 'vit_h',  # vit_h, vit_l, vit_b, vit_tiny
        "checkpoint": "weights/sam_vit_h.pth",
        "freeze": {
            "image_encoder": True,
            "prompt_encoder": False,
            "mask_decoder": False,
        },
    },
    "dataset": {
        "train": {
            "root_dir": "./datasets/train_images",
            "annotation_file": "./datasets/annotations/train_sam.json"
        },
        "val": {
            "root_dir": "./datasets/val_images",
            "annotation_file": "./datasets/annotations/val_sam.json"
        }
    }
}

cfg = Box(config)
