from box import Box

config = {
    "num_devices": 1,
    "out_dir": "./runs/train",
    "model": {
        "type": 'vit_h',  # vit_h, vit_l, vit_b, vit_tiny
        "checkpoint": "weights/sam_vit_h.pth",
        "freeze": {
            "image_encoder": True,
            "prompt_encoder": True,
            "mask_decoder": True,
        },
    },
}

teacher_cfg = Box(config)
