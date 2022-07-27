"""
Convert TFRecord data to pytorch data.
Dataset can be found here:
https://console.cloud.google.com/storage/browser/grid-cells-datasets
"""
import tensorflow as tf
import dataset_reader
import torch

data_reader = dataset_reader.DataReader(
        'square_room', '../data/', num_threads=8)
train_traj = data_reader.read(batch_size=10000)
in_pos, in_hd, ego_vel, target_pos, target_hd = train_traj
with tf.train.SingularMonitoredSession() as sess:

    for i in range(99):
        res = sess.run({
        "init_pos": in_pos,
        "init_hd" :in_hd,
        "ego_vel": ego_vel,
        "target_pos":target_pos,
        "target_hd":target_hd,
        })

        torch.save(res, '../data/torch/{}-99.pt'.format(i))
