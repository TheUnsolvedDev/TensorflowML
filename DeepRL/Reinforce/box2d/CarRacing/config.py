NUM_TRAINING_STEPS = int(1e+6)
NUM_TEST_GAMES = 10

ALPHA = 1e-4
GAMMA = 0.99
BATCH_SIZE = 128

ENV_INPUT_SHAPE = (48,48,1)
ENV_OUTPUT_SHAPE = (5,)
ENV_NAME = 'CarRacing-v3'
NUM_ENVS = 8