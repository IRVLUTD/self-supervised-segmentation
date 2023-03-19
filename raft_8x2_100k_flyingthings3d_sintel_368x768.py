model = dict(
    type='RAFT',
    num_levels=4,
    radius=4,
    cxt_channels=128,
    h_channels=128,
    encoder=dict(
        type='RAFTEncoder',
        in_channels=3,
        out_channels=256,
        net_type='Basic',
        norm_cfg=dict(type='IN'),
        init_cfg=[
            dict(
                type='Kaiming',
                layer=['Conv2d'],
                mode='fan_out',
                nonlinearity='relu'),
            dict(type='Constant', layer=['InstanceNorm2d'], val=1, bias=0)
        ]),
    cxt_encoder=dict(
        type='RAFTEncoder',
        in_channels=3,
        out_channels=256,
        net_type='Basic',
        norm_cfg=dict(type='SyncBN'),
        init_cfg=[
            dict(
                type='Kaiming',
                layer=['Conv2d'],
                mode='fan_out',
                nonlinearity='relu'),
            dict(type='Constant', layer=['SyncBatchNorm2d'], val=1, bias=0)
        ]),
    decoder=dict(
        type='RAFTDecoder',
        net_type='Basic',
        num_levels=4,
        radius=4,
        iters=12,
        corr_op_cfg=dict(type='CorrLookup', align_corners=True),
        gru_type='SeqConv',
        flow_loss=dict(type='SequenceLoss', gamma=0.85),
        act_cfg=dict(type='ReLU')),
    freeze_bn=True,
    train_cfg=dict(),
    test_cfg=dict(iters=32))
img_norm_cfg = dict(
    mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], to_rgb=False)
crop_size = (368, 768)
sintel_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        type='ColorJitter',
        asymmetric_prob=0.2,
        brightness=0.4,
        contrast=0.4,
        saturation=0.4,
        hue=0.1592356687898089),
    dict(type='Erase', prob=0.5, bounds=[50, 100], max_num=3),
    dict(
        type='SpacialTransform',
        spacial_prob=0.8,
        stretch_prob=0.8,
        crop_size=(368, 768),
        min_scale=-0.2,
        max_scale=0.6,
        max_stretch=0.2),
    dict(type='RandomCrop', crop_size=(368, 768)),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.1, direction='vertical'),
    dict(type='Validation', max_flow=1000.0),
    dict(
        type='Normalize',
        mean=[127.5, 127.5, 127.5],
        std=[127.5, 127.5, 127.5],
        to_rgb=False),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['imgs', 'flow_gt', 'valid'],
        meta_keys=[
            'filename1', 'filename2', 'ori_filename1', 'ori_filename2',
            'filename_flow', 'ori_filename_flow', 'ori_shape', 'img_shape',
            'erase_bounds', 'erase_num', 'scale_factor'
        ])
]
sintel_test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='InputPad', exponent=3),
    dict(
        type='Normalize',
        mean=[127.5, 127.5, 127.5],
        std=[127.5, 127.5, 127.5],
        to_rgb=False),
    dict(type='TestFormatBundle'),
    dict(
        type='Collect',
        keys=['imgs'],
        meta_keys=[
            'flow_gt', 'filename1', 'filename2', 'ori_filename1',
            'ori_filename2', 'ori_shape', 'img_shape', 'img_norm_cfg',
            'scale_factor', 'pad_shape', 'pad'
        ])
]
sintel_clean_train = dict(
    type='Sintel',
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations'),
        dict(
            type='ColorJitter',
            asymmetric_prob=0.2,
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
            hue=0.1592356687898089),
        dict(type='Erase', prob=0.5, bounds=[50, 100], max_num=3),
        dict(
            type='SpacialTransform',
            spacial_prob=0.8,
            stretch_prob=0.8,
            crop_size=(368, 768),
            min_scale=-0.2,
            max_scale=0.6,
            max_stretch=0.2),
        dict(type='RandomCrop', crop_size=(368, 768)),
        dict(type='RandomFlip', prob=0.5, direction='horizontal'),
        dict(type='RandomFlip', prob=0.1, direction='vertical'),
        dict(type='Validation', max_flow=1000.0),
        dict(
            type='Normalize',
            mean=[127.5, 127.5, 127.5],
            std=[127.5, 127.5, 127.5],
            to_rgb=False),
        dict(type='DefaultFormatBundle'),
        dict(
            type='Collect',
            keys=['imgs', 'flow_gt', 'valid'],
            meta_keys=[
                'filename1', 'filename2', 'ori_filename1', 'ori_filename2',
                'filename_flow', 'ori_filename_flow', 'ori_shape', 'img_shape',
                'erase_bounds', 'erase_num', 'scale_factor'
            ])
    ],
    data_root='data/Sintel',
    test_mode=False,
    pass_style='clean')
sintel_final_train = dict(
    type='Sintel',
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations'),
        dict(
            type='ColorJitter',
            asymmetric_prob=0.2,
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
            hue=0.1592356687898089),
        dict(type='Erase', prob=0.5, bounds=[50, 100], max_num=3),
        dict(
            type='SpacialTransform',
            spacial_prob=0.8,
            stretch_prob=0.8,
            crop_size=(368, 768),
            min_scale=-0.2,
            max_scale=0.6,
            max_stretch=0.2),
        dict(type='RandomCrop', crop_size=(368, 768)),
        dict(type='RandomFlip', prob=0.5, direction='horizontal'),
        dict(type='RandomFlip', prob=0.1, direction='vertical'),
        dict(type='Validation', max_flow=1000.0),
        dict(
            type='Normalize',
            mean=[127.5, 127.5, 127.5],
            std=[127.5, 127.5, 127.5],
            to_rgb=False),
        dict(type='DefaultFormatBundle'),
        dict(
            type='Collect',
            keys=['imgs', 'flow_gt', 'valid'],
            meta_keys=[
                'filename1', 'filename2', 'ori_filename1', 'ori_filename2',
                'filename_flow', 'ori_filename_flow', 'ori_shape', 'img_shape',
                'erase_bounds', 'erase_num', 'scale_factor'
            ])
    ],
    data_root='data/Sintel',
    test_mode=False,
    pass_style='final')
sintel_clean_train_x100 = dict(
    type='RepeatDataset',
    times=100,
    dataset=dict(
        type='Sintel',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(
                type='ColorJitter',
                asymmetric_prob=0.2,
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.1592356687898089),
            dict(type='Erase', prob=0.5, bounds=[50, 100], max_num=3),
            dict(
                type='SpacialTransform',
                spacial_prob=0.8,
                stretch_prob=0.8,
                crop_size=(368, 768),
                min_scale=-0.2,
                max_scale=0.6,
                max_stretch=0.2),
            dict(type='RandomCrop', crop_size=(368, 768)),
            dict(type='RandomFlip', prob=0.5, direction='horizontal'),
            dict(type='RandomFlip', prob=0.1, direction='vertical'),
            dict(type='Validation', max_flow=1000.0),
            dict(
                type='Normalize',
                mean=[127.5, 127.5, 127.5],
                std=[127.5, 127.5, 127.5],
                to_rgb=False),
            dict(type='DefaultFormatBundle'),
            dict(
                type='Collect',
                keys=['imgs', 'flow_gt', 'valid'],
                meta_keys=[
                    'filename1', 'filename2', 'ori_filename1', 'ori_filename2',
                    'filename_flow', 'ori_filename_flow', 'ori_shape',
                    'img_shape', 'erase_bounds', 'erase_num', 'scale_factor'
                ])
        ],
        data_root='data/Sintel',
        test_mode=False,
        pass_style='clean'))
sintel_final_train_x100 = dict(
    type='RepeatDataset',
    times=100,
    dataset=dict(
        type='Sintel',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(
                type='ColorJitter',
                asymmetric_prob=0.2,
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.1592356687898089),
            dict(type='Erase', prob=0.5, bounds=[50, 100], max_num=3),
            dict(
                type='SpacialTransform',
                spacial_prob=0.8,
                stretch_prob=0.8,
                crop_size=(368, 768),
                min_scale=-0.2,
                max_scale=0.6,
                max_stretch=0.2),
            dict(type='RandomCrop', crop_size=(368, 768)),
            dict(type='RandomFlip', prob=0.5, direction='horizontal'),
            dict(type='RandomFlip', prob=0.1, direction='vertical'),
            dict(type='Validation', max_flow=1000.0),
            dict(
                type='Normalize',
                mean=[127.5, 127.5, 127.5],
                std=[127.5, 127.5, 127.5],
                to_rgb=False),
            dict(type='DefaultFormatBundle'),
            dict(
                type='Collect',
                keys=['imgs', 'flow_gt', 'valid'],
                meta_keys=[
                    'filename1', 'filename2', 'ori_filename1', 'ori_filename2',
                    'filename_flow', 'ori_filename_flow', 'ori_shape',
                    'img_shape', 'erase_bounds', 'erase_num', 'scale_factor'
                ])
        ],
        data_root='data/Sintel',
        test_mode=False,
        pass_style='final'))
sintel_clean_test = dict(
    type='Sintel',
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations'),
        dict(type='InputPad', exponent=3),
        dict(
            type='Normalize',
            mean=[127.5, 127.5, 127.5],
            std=[127.5, 127.5, 127.5],
            to_rgb=False),
        dict(type='TestFormatBundle'),
        dict(
            type='Collect',
            keys=['imgs'],
            meta_keys=[
                'flow_gt', 'filename1', 'filename2', 'ori_filename1',
                'ori_filename2', 'ori_shape', 'img_shape', 'img_norm_cfg',
                'scale_factor', 'pad_shape', 'pad'
            ])
    ],
    data_root='data/Sintel',
    test_mode=True,
    pass_style='clean')
sintel_final_test = dict(
    type='Sintel',
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations'),
        dict(type='InputPad', exponent=3),
        dict(
            type='Normalize',
            mean=[127.5, 127.5, 127.5],
            std=[127.5, 127.5, 127.5],
            to_rgb=False),
        dict(type='TestFormatBundle'),
        dict(
            type='Collect',
            keys=['imgs'],
            meta_keys=[
                'flow_gt', 'filename1', 'filename2', 'ori_filename1',
                'ori_filename2', 'ori_shape', 'img_shape', 'img_norm_cfg',
                'scale_factor', 'pad_shape', 'pad'
            ])
    ],
    data_root='data/Sintel',
    test_mode=True,
    pass_style='final')
flyingthing3d_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        type='ColorJitter',
        asymmetric_prob=0.2,
        brightness=0.4,
        contrast=0.4,
        saturation=0.4,
        hue=0.1592356687898089),
    dict(type='Erase', prob=0.5, bounds=[50, 100], max_num=3),
    dict(
        type='SpacialTransform',
        spacial_prob=0.8,
        stretch_prob=0.8,
        crop_size=(368, 768),
        min_scale=-0.4,
        max_scale=0.8,
        max_stretch=0.2),
    dict(type='RandomCrop', crop_size=(368, 768)),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.1, direction='vertical'),
    dict(type='Validation', max_flow=1000.0),
    dict(
        type='Normalize',
        mean=[127.5, 127.5, 127.5],
        std=[127.5, 127.5, 127.5],
        to_rgb=False),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['imgs', 'flow_gt', 'valid'],
        meta_keys=[
            'filename1', 'filename2', 'ori_filename1', 'ori_filename2',
            'filename_flow', 'ori_filename_flow', 'ori_shape', 'img_shape',
            'erase_bounds', 'erase_num', 'scale_factor'
        ])
]
flyingthings3d_clean_train = dict(
    type='FlyingThings3D',
    data_root='data/flyingthings3d',
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations'),
        dict(
            type='ColorJitter',
            asymmetric_prob=0.2,
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
            hue=0.1592356687898089),
        dict(type='Erase', prob=0.5, bounds=[50, 100], max_num=3),
        dict(
            type='SpacialTransform',
            spacial_prob=0.8,
            stretch_prob=0.8,
            crop_size=(368, 768),
            min_scale=-0.4,
            max_scale=0.8,
            max_stretch=0.2),
        dict(type='RandomCrop', crop_size=(368, 768)),
        dict(type='RandomFlip', prob=0.5, direction='horizontal'),
        dict(type='RandomFlip', prob=0.1, direction='vertical'),
        dict(type='Validation', max_flow=1000.0),
        dict(
            type='Normalize',
            mean=[127.5, 127.5, 127.5],
            std=[127.5, 127.5, 127.5],
            to_rgb=False),
        dict(type='DefaultFormatBundle'),
        dict(
            type='Collect',
            keys=['imgs', 'flow_gt', 'valid'],
            meta_keys=[
                'filename1', 'filename2', 'ori_filename1', 'ori_filename2',
                'filename_flow', 'ori_filename_flow', 'ori_shape', 'img_shape',
                'erase_bounds', 'erase_num', 'scale_factor'
            ])
    ],
    test_mode=False,
    pass_style='clean',
    scene='left')
data = dict(
    train_dataloader=dict(
        samples_per_gpu=2,
        workers_per_gpu=2,
        drop_last=True,
        shuffle=True,
        persistent_workers=True),
    val_dataloader=dict(
        samples_per_gpu=1,
        workers_per_gpu=2,
        shuffle=False,
        persistent_workers=True),
    test_dataloader=dict(samples_per_gpu=1, workers_per_gpu=2, shuffle=False),
    train=[
        dict(
            type='RepeatDataset',
            times=100,
            dataset=dict(
                type='Sintel',
                pipeline=[
                    dict(type='LoadImageFromFile'),
                    dict(type='LoadAnnotations'),
                    dict(
                        type='ColorJitter',
                        asymmetric_prob=0.2,
                        brightness=0.4,
                        contrast=0.4,
                        saturation=0.4,
                        hue=0.1592356687898089),
                    dict(type='Erase', prob=0.5, bounds=[50, 100], max_num=3),
                    dict(
                        type='SpacialTransform',
                        spacial_prob=0.8,
                        stretch_prob=0.8,
                        crop_size=(368, 768),
                        min_scale=-0.2,
                        max_scale=0.6,
                        max_stretch=0.2),
                    dict(type='RandomCrop', crop_size=(368, 768)),
                    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
                    dict(type='RandomFlip', prob=0.1, direction='vertical'),
                    dict(type='Validation', max_flow=1000.0),
                    dict(
                        type='Normalize',
                        mean=[127.5, 127.5, 127.5],
                        std=[127.5, 127.5, 127.5],
                        to_rgb=False),
                    dict(type='DefaultFormatBundle'),
                    dict(
                        type='Collect',
                        keys=['imgs', 'flow_gt', 'valid'],
                        meta_keys=[
                            'filename1', 'filename2', 'ori_filename1',
                            'ori_filename2', 'filename_flow',
                            'ori_filename_flow', 'ori_shape', 'img_shape',
                            'erase_bounds', 'erase_num', 'scale_factor'
                        ])
                ],
                data_root='data/Sintel',
                test_mode=False,
                pass_style='clean')),
        dict(
            type='RepeatDataset',
            times=100,
            dataset=dict(
                type='Sintel',
                pipeline=[
                    dict(type='LoadImageFromFile'),
                    dict(type='LoadAnnotations'),
                    dict(
                        type='ColorJitter',
                        asymmetric_prob=0.2,
                        brightness=0.4,
                        contrast=0.4,
                        saturation=0.4,
                        hue=0.1592356687898089),
                    dict(type='Erase', prob=0.5, bounds=[50, 100], max_num=3),
                    dict(
                        type='SpacialTransform',
                        spacial_prob=0.8,
                        stretch_prob=0.8,
                        crop_size=(368, 768),
                        min_scale=-0.2,
                        max_scale=0.6,
                        max_stretch=0.2),
                    dict(type='RandomCrop', crop_size=(368, 768)),
                    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
                    dict(type='RandomFlip', prob=0.1, direction='vertical'),
                    dict(type='Validation', max_flow=1000.0),
                    dict(
                        type='Normalize',
                        mean=[127.5, 127.5, 127.5],
                        std=[127.5, 127.5, 127.5],
                        to_rgb=False),
                    dict(type='DefaultFormatBundle'),
                    dict(
                        type='Collect',
                        keys=['imgs', 'flow_gt', 'valid'],
                        meta_keys=[
                            'filename1', 'filename2', 'ori_filename1',
                            'ori_filename2', 'filename_flow',
                            'ori_filename_flow', 'ori_shape', 'img_shape',
                            'erase_bounds', 'erase_num', 'scale_factor'
                        ])
                ],
                data_root='data/Sintel',
                test_mode=False,
                pass_style='final')),
        dict(
            type='FlyingThings3D',
            data_root='data/flyingthings3d',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations'),
                dict(
                    type='ColorJitter',
                    asymmetric_prob=0.2,
                    brightness=0.4,
                    contrast=0.4,
                    saturation=0.4,
                    hue=0.1592356687898089),
                dict(type='Erase', prob=0.5, bounds=[50, 100], max_num=3),
                dict(
                    type='SpacialTransform',
                    spacial_prob=0.8,
                    stretch_prob=0.8,
                    crop_size=(368, 768),
                    min_scale=-0.4,
                    max_scale=0.8,
                    max_stretch=0.2),
                dict(type='RandomCrop', crop_size=(368, 768)),
                dict(type='RandomFlip', prob=0.5, direction='horizontal'),
                dict(type='RandomFlip', prob=0.1, direction='vertical'),
                dict(type='Validation', max_flow=1000.0),
                dict(
                    type='Normalize',
                    mean=[127.5, 127.5, 127.5],
                    std=[127.5, 127.5, 127.5],
                    to_rgb=False),
                dict(type='DefaultFormatBundle'),
                dict(
                    type='Collect',
                    keys=['imgs', 'flow_gt', 'valid'],
                    meta_keys=[
                        'filename1', 'filename2', 'ori_filename1',
                        'ori_filename2', 'filename_flow', 'ori_filename_flow',
                        'ori_shape', 'img_shape', 'erase_bounds', 'erase_num',
                        'scale_factor'
                    ])
            ],
            test_mode=False,
            pass_style='clean',
            scene='left')
    ],
    val=dict(
        type='ConcatDataset',
        datasets=[
            dict(
                type='Sintel',
                pipeline=[
                    dict(type='LoadImageFromFile'),
                    dict(type='LoadAnnotations'),
                    dict(type='InputPad', exponent=3),
                    dict(
                        type='Normalize',
                        mean=[127.5, 127.5, 127.5],
                        std=[127.5, 127.5, 127.5],
                        to_rgb=False),
                    dict(type='TestFormatBundle'),
                    dict(
                        type='Collect',
                        keys=['imgs'],
                        meta_keys=[
                            'flow_gt', 'filename1', 'filename2',
                            'ori_filename1', 'ori_filename2', 'ori_shape',
                            'img_shape', 'img_norm_cfg', 'scale_factor',
                            'pad_shape', 'pad'
                        ])
                ],
                data_root='data/Sintel',
                test_mode=True,
                pass_style='clean'),
            dict(
                type='Sintel',
                pipeline=[
                    dict(type='LoadImageFromFile'),
                    dict(type='LoadAnnotations'),
                    dict(type='InputPad', exponent=3),
                    dict(
                        type='Normalize',
                        mean=[127.5, 127.5, 127.5],
                        std=[127.5, 127.5, 127.5],
                        to_rgb=False),
                    dict(type='TestFormatBundle'),
                    dict(
                        type='Collect',
                        keys=['imgs'],
                        meta_keys=[
                            'flow_gt', 'filename1', 'filename2',
                            'ori_filename1', 'ori_filename2', 'ori_shape',
                            'img_shape', 'img_norm_cfg', 'scale_factor',
                            'pad_shape', 'pad'
                        ])
                ],
                data_root='data/Sintel',
                test_mode=True,
                pass_style='final')
        ],
        separate_eval=True),
    test=dict(
        type='ConcatDataset',
        datasets=[
            dict(
                type='Sintel',
                pipeline=[
                    dict(type='LoadImageFromFile'),
                    dict(type='LoadAnnotations'),
                    dict(type='InputPad', exponent=3),
                    dict(
                        type='Normalize',
                        mean=[127.5, 127.5, 127.5],
                        std=[127.5, 127.5, 127.5],
                        to_rgb=False),
                    dict(type='TestFormatBundle'),
                    dict(
                        type='Collect',
                        keys=['imgs'],
                        meta_keys=[
                            'flow_gt', 'filename1', 'filename2',
                            'ori_filename1', 'ori_filename2', 'ori_shape',
                            'img_shape', 'img_norm_cfg', 'scale_factor',
                            'pad_shape', 'pad'
                        ])
                ],
                data_root='data/Sintel',
                test_mode=True,
                pass_style='clean'),
            dict(
                type='Sintel',
                pipeline=[
                    dict(type='LoadImageFromFile'),
                    dict(type='LoadAnnotations'),
                    dict(type='InputPad', exponent=3),
                    dict(
                        type='Normalize',
                        mean=[127.5, 127.5, 127.5],
                        std=[127.5, 127.5, 127.5],
                        to_rgb=False),
                    dict(type='TestFormatBundle'),
                    dict(
                        type='Collect',
                        keys=['imgs'],
                        meta_keys=[
                            'flow_gt', 'filename1', 'filename2',
                            'ori_filename1', 'ori_filename2', 'ori_shape',
                            'img_shape', 'img_norm_cfg', 'scale_factor',
                            'pad_shape', 'pad'
                        ])
                ],
                data_root='data/Sintel',
                test_mode=True,
                pass_style='final')
        ],
        separate_eval=True))
log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = 'work_dir/raft/1228_raft_syncbn_things3d/iter_90000.pth'
resume_from = None
workflow = [('train', 1)]
optimizer = dict(
    type='AdamW',
    lr=0.000125,
    betas=(0.9, 0.999),
    eps=1e-08,
    weight_decay=1e-05,
    amsgrad=False)
optimizer_config = dict(grad_clip=dict(max_norm=1.0))
lr_config = dict(
    policy='OneCycle',
    max_lr=0.000125,
    total_steps=100100,
    pct_start=0.05,
    anneal_strategy='linear')
runner = dict(type='IterBasedRunner', max_iters=100000)
checkpoint_config = dict(by_epoch=False, interval=10000)
evaluation = dict(interval=10000, metric='EPE')
work_dir = 'work_dir/raft/1229_sintel_syncbn'
gpu_ids = range(0, 1)
