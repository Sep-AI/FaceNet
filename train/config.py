
image_size=160

graph_dir='../graph/'

model_dir='../model/'

validation_set_split_ratio=0.05

min_nrof_val_images_per_class=0.0

data_dir='../data/casia_mtcnn_182'

batch_size=90

keep_probability=0.8

embedding_size=512

weight_decay=5e-4

center_loss_alfa=0.6

center_loss_factor=1e-2

learning_rate=0.01

LR_EPOCH=[10,20,40]

learning_rate_decay_factor=0.98

moving_average_decay=0.999

max_nrof_epochs=150
quant_delay = 200000
