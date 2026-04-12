import os


PROJECT_ROOT = os.environ.get(
    "POLARIS_PROJECT_ROOT",
    "/home/b311/data2/25-zhangxizhe/code/PoLaRIS-Infrared-LiDAR-Detection",
)
DATA_ROOT = os.path.join(PROJECT_ROOT, "dataset", "NUAA-SIRST", "boxinstseg_coco") + "/"
WORK_DIR = os.path.join(PROJECT_ROOT, "baselines", "boxinstseg", "work_dirs", "box_levelset_nuaa_r50_fpn_3x")
CLASSES = ("target",)


model = dict(
    type="BoxLevelSet",
    backbone=dict(
        type="ResNet",
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        style="pytorch",
        init_cfg=dict(type="Pretrained", checkpoint="https://download.pytorch.org/models/resnet50-11ad3fa6.pth"),
    ),
    neck=dict(
        type="FPN",
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=0,
        num_outs=5,
    ),
    bbox_head=dict(
        type="BoxSOLOv2Head",
        num_classes=1,
        in_channels=256,
        stacked_convs=4,
        seg_feat_channels=256,
        strides=[8, 8, 16, 32, 32],
        scale_ranges=((1, 32), (16, 64), (32, 128), (64, 256), (128, 2048)),
        sigma=0.2,
        num_grids=[16, 12, 10, 8, 6],
        cate_down_pos=0,
        loss_cate=dict(
            type="FocalLoss",
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0,
        ),
        loss_boxpro=dict(type="BoxProjectionLoss", loss_weight=3.0),
        loss_levelset=dict(type="LevelsetLoss", loss_weight=1.0),
    ),
    train_cfg=dict(),
    test_cfg=dict(
        nms_pre=500,
        score_thr=0.05,
        mask_thr=0.55,
        filter_thr=0.025,
        kernel="gaussian",
        sigma=2.0,
        max_per_img=100,
    ),
)


dataset_type = "CocoDataset"
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True, with_mask=False),
    dict(type="GenerateBoxMask"),
    dict(type="Resize", img_scale=(256, 256), keep_ratio=True),
    dict(type="RandomFlip", flip_ratio=0.5),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size_divisor=32),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels", "gt_masks"]),
]
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(256, 256),
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="Pad", size_divisor=32),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        classes=CLASSES,
        ann_file=DATA_ROOT + "annotations/instances_train2017.json",
        img_prefix=DATA_ROOT + "train2017/",
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        classes=CLASSES,
        ann_file=DATA_ROOT + "annotations/instances_val2017.json",
        img_prefix=DATA_ROOT + "val2017/",
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        classes=CLASSES,
        ann_file=DATA_ROOT + "annotations/instances_val2017.json",
        img_prefix=DATA_ROOT + "val2017/",
        pipeline=test_pipeline,
    ),
)


optimizer = dict(
    type="AdamW",
    lr=0.00005,
    weight_decay=0.1,
    paramwise_cfg=dict(norm_decay_mult=0.0, bypass_duplicate=True),
)
optimizer_config = dict(grad_clip=dict(max_norm=1, norm_type=2))
lr_config = dict(
    policy="step",
    warmup="linear",
    warmup_iters=2000,
    warmup_ratio=0.001,
    step=[27, 33],
)

checkpoint_config = dict(interval=1)
log_config = dict(interval=50, hooks=[dict(type="TextLoggerHook")])
runner = dict(type="EpochBasedRunner", max_epochs=36)
evaluation = dict(interval=1, metric=["segm"])
device_ids = range(1)
dist_params = dict(backend="nccl")
log_level = "INFO"
work_dir = WORK_DIR
load_from = None
resume_from = None
workflow = [("train", 1)]
