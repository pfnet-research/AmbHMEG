custom_imports = dict(
    imports=[
        "gryphgen.data",
        "gryphgen.engine",
        "gryphgen.model",
        "gryphgen.score",
        "gryphgen.vocab",
    ]
)

len_tex = 150
num_workers = 0

model = dict(
    type="StableDiffusion",
    model_id="CompVis/stable-diffusion-v1-4",
    vocab=dict(
        type="FormulaBatcher",
        lexicon=dict(
            type="FormulaVocab",
            load="alphabet/mathwriting.txt",
            skip="<SKIP>",
            mask="<MASK>",
        ),
    ),
    text_encoder=None,
    graph_encoder=dict(
        type="GraphEncoder",
        vocab_load="~/data/mathwriting/pickles/graph/vocab.pkl",
        embedding_dim=768 // 2,  # default: 128
        gconv_dim=768 // 2,  # default: 128
        gconv_hidden_dim=512,
        gconv_pooling="avg",
        gconv_num_layers=5,
        mlp_normalization="none",
    ),
    p_uncond=0.1,
    guidance_scale=7.5,
)

# add for memory-efficient
train_pipeline = [
    dict(type="LoadFromPickle", key="pkl_path"),
    dict(type="ScaleInk", w=224, h=224),
    dict(type="PaintInk", w=224, h=224, fill=(1, 1, 1), line=1),
    dict(
        type="Annotate",
        keys=["img"],
        meta=["tex", "name", "node_types", "edge_types", "edges"],
    ),
]

valid_pipeline = [
    dict(type="ScaleInk", w=224, h=224),
    dict(type="PaintInk", w=224, h=224, fill=(1, 1, 1), line=1),
    dict(
        type="Annotate",
        keys=["img"],
        meta=["tex", "name", "node_types", "edge_types", "edges"],
    ),
]

train_dataloader = dict(
    batch_size=128,
    num_workers=num_workers,
    sampler=dict(
        type="DefaultSampler",
        shuffle=True,
    ),
    dataset=dict(
        type="FormulaPathDataset",
        ann_file="~/data/mathwriting/pickles/graph/train/",
        pipeline=train_pipeline,
        test_mode=False,
    ),
)

val_dataloader = dict(
    batch_size=32,
    num_workers=num_workers,
    sampler=dict(
        type="DefaultSampler",
        shuffle=False,
    ),
    dataset=dict(
        type="FormulaDataset",
        ann_file="~/data/mathwriting/pickles/graph/valid/",
        pipeline=valid_pipeline,
        test_mode=True,
        indices=range(9),
    ),
)

test_dataloader = dict(
    batch_size=32,
    num_workers=num_workers,
    sampler=dict(
        type="DefaultSampler",
        shuffle=False,
    ),
    dataset=dict(
        type="FormulaDataset",
        ann_file="~/data/mathwriting/pickles/graph/test/",
        pipeline=valid_pipeline,
        test_mode=True,
    ),
)

train_cfg = dict(
    type="EpochBasedTrainLoop",
    max_epochs=50,
)

val_cfg = dict(type="ValLoop")

test_cfg = dict(type="TestLoop")

optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(
        type="AdamW",
        lr=1e-4,
        betas=(0.9, 0.999),
        weight_decay=0.001,
    ),
    clip_grad=dict(
        max_norm=10,
        norm_type=2,
    ),
)

param_scheduler = [
    dict(
        type="LinearLR",
        start_factor=0.2,
        begin=0,
        end=100,
        by_epoch=False,
    ),
    dict(
        type="MultiStepLR",
        milestones=[50],
        gamma=0.1,
        by_epoch=True,
    ),
]

work_dir = "work-graph"

val_evaluator = dict(type="DumpImage", output_dir=f"{work_dir}/dump")

test_evaluator = dict(type="DumpImage", output_dir=f"{work_dir}/dump")

launcher = "none"

default_hooks = dict(
    checkpoint=dict(
        type="CheckpointHook",
        interval=1,
        max_keep_ckpts=1,
        by_epoch=True,
    )
)

custom_hooks = [dict(type="CheckInvalidLossHook", interval=1)]
