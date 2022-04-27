_base_ = ['./upernet_beit-refine-test-cnn-mix-rand-s1.py']

model = dict(backbone=dict(ref_segmentor=dict(init_cfg=None)))


