model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='EfficientNet',
        arch='b0',
        init_cfg=dict(
            type='Pretrained',
            checkpoint=
            'https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet-b0_3rdparty_8xb32_in1k_20220119-a7e2a0b1.pth',
            prefix='backbone')),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=94,
        in_channels=1280,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5)))
dataset_type = 'CustomDataset'
classes =['VW-TIGUAN',
         'SEAT-IBIZA',
         'PEUGEOT-206',
         'PEUGEOT-2008',
         'AUDI-A3',
         'MINI-COUNTRYMAN',
         'MERCEDES-A',
         'VOLVO-V40',
         'ALFA-ROMEO-147',
         'SKODA-OCTAVIA',
         'RENAULT-MEGANE-2005',
         'ALFA-ROMEO-156',
         'SKODA-OKTAVIA-2005',
         'TOYOTA-STARLET',
        'FIAT-PANDA',
         'PEUGEOT-107',
         'TOYOTA-C-HR',
         'HYUNDAI-I20',
         'VW-T-ROC',
         'SEAT-CORDOBA',
         'SEAT-LEON',
         'OPEL-CORSA',
         'NISSAN-MICRA',
         'NISSAN-NOTE',
         'SUZUKI-SWIFT',
         'FORD-FIESTA',
         'FIAT-STILO',
         'BMW-X1',
         'HONDA-JAZZ',
         'PEUGEOT-208',
         'KIA-RIO',
         'SKODA-FABIA-2005',
         'PEUGEOT-307',
         'CHEVROLET-MATIZ',
         'TOYOTA-COROLLA',
         'NISSAN-ALMERA',
         'MERCEDES-C',
         'HYUNDAI-ATOS',
         'VW-PASSAT',
         'KIA-CEED',
         'DACIA-SANTERO',
         'ALFA-ROMEO-146',
         'FORD-PUMA',
         'BMW-1-SERIES',
         'FIAT-PUNTO',
         'CITROEN-SAXO',
         'AUDI-TT',
         'HONDA-CIVIC-2000',
         'DAIHATSU-SIRION',
         'KIA-PICANTO',
         'FIAT-TIPO',
         'TOYOTA-YARIS',
         'PEUGEOT-106',
         'CITROEN-C3',
         'PORSCHE-CAYENNE',
         'VW-GOLF',
         'TOYOTA-RAV4',
         'RENAULT-MEGANE',
         'OPEL-INSIGNIA',
         'RENAULT-CLIO',
         'BMW-3-SERIES',
         'OPEL-VECTRA',
         'CITROEN-C4',
         'MINI-COOPER-CLUBMAN',
         'CHEVROLET-SPARK',
         'HYUNDAI-GETZ',
         'HYUNDAI-I30',
         'AUDI-A1',
         'PEUGEOT-207',
         'HYUNDAI-ACCENT',
         'TOYOTA-AVENSIS',
         'NISSAN-PRIMERA',
         'SUZUKI-VITARA',
         'OPEL-ASTRA',
         'AUDI-Q3',
         'TOYOTA-AURIS',
         'PEUGEOT-108',
         'TOYOTA-AYGO',
         'HYUNDAI-I10',
         'MERCEDES-GLA',
         'SEAT-IBIZA-2005',
         'DAIHATSU-TERIOS',
         'FIAT-500',
         'FIAT-SEICENTO',
         'DACIA-DUSTER',
         'CITROEN-XSARA',
         'VW-POLO',
         'SKODA-FABIA',
         'FORD-ESCORT',
         'NISSAN-QASHQAI',
         'SMART-FORTWO',
         'HONDA-CIVIC',
         'FORD-FOCUS',
         'SMART-FORTWO-2005']         
img_norm_cfg = dict(
    mean=[125.307, 122.961, 113.8575],
    std=[51.5865, 50.847, 51.255],
    to_rgb=False)
    
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(224, -1), adaptive_side='long'),
    dict(type='Pad', size=(224, 224)),
    dict(type='RandomGrayscale', gray_prob=0.2),
    dict(type='ColorJitter', brightness=0.5, contrast=0.5, saturation=0.5),
    dict(type='RandomFlip'),
    dict(
        type='Normalize',
        mean=[125.307, 122.961, 113.8575],
        std=[51.5865, 50.847, 51.255],
        to_rgb=False),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(224, -1), adaptive_side='long'),
    dict(type='Pad', size=(224, 224)),
    dict(
        type='Normalize',
        mean=[125.307, 122.961, 113.8575],
        std=[51.5865, 50.847, 51.255],
        to_rgb=False),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]

data = dict(
    samples_per_gpu=64,
    workers_per_gpu=2,
    train=dict(
        type='CustomDataset',
        data_prefix='/data/grubles/car_brand_recognition/custom_cars_dataset/datasets/dataset_models',
        pipeline=train_pipeline
        ),
    val=dict(
        type='CustomDataset',
        data_prefix='/data/grubles/car_brand_recognition/custom_cars_dataset/datasets/test_dataset_models',
        pipeline=test_pipeline
        ),
    test=dict(
        type='CustomDataset',
        data_prefix='/data/grubles/car_brand_recognition/custom_cars_dataset/datasets/test_dataset_models',
        pipeline=test_pipeline
        ))
        
evaluation = dict(interval=5, metric='accuracy')
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='step', step=[20])
runner = dict(type='EpochBasedRunner', max_epochs=100)
checkpoint_config = dict(interval=10)
log_config = dict(interval=150, hooks=[dict(type='TensorboardLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = '/data/grubles/car_brand_recognition/custom_cars_dataset/experiments/exp_5'
load_from = None
resume_from = None
workflow = [('train', 1)]
gpu_ids = [0]
