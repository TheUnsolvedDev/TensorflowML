import tensorflow as tf
import numpy as np

mirrored_strategy = tf.distribute.MirroredStrategy()
print(
    f'using distribution strategy\nnumber of gpus:{mirrored_strategy.num_replicas_in_sync}')

dataset = tf.data.Dataset.from_tensor_slices(
    np.random.rand(64, 224, 224, 3)).batch(8)

# create distributed dataset
ds = mirrored_strategy.experimental_distribute_dataset(dataset)

# make variables mirrored
with mirrored_strategy.scope():
    resnet50 = tf.keras.applications.resnet50.ResNet50()


def step_fn(pre_images):
    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(pre_images)
        outputs = resnet50(pre_images)[:, 0:1]
    return tape.gradient(outputs,pre_images)

# define distributed step function using strategy.run and strategy.gather


@tf.function
def distributed_step_fn(pre_images):
    per_replica_grads = mirrored_strategy.run(step_fn, args=(pre_images,))
    # print(per_replica_grads)
    tf.print(per_replica_grads)
    return mirrored_strategy.gather(per_replica_grads, 0)


# loop over distributed dataset with distributed_step_fn
for result in map(distributed_step_fn, ds):
    print(result.numpy().shape)
