import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from keras.models import load_model
import gym
import numpy as np
import tensorflow as tf

tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)

def step(model, observation):
    actions = model.predict(observation.reshape(1, len(observation)))[0]
    return np.argmax(actions)

def load(name):
    model = load_model(name)
    return model

print('\n\n')

#inp = input('name of model: ')
inp = 'cartpole_vanilla_dqn.h5'
model = load(inp)
if 'cartpole' in inp:
    env = 'CartPole-v1'
elif 'acrobot' in inp:
    env = 'Acrobot-v1'
else:
    print('Invalid model, not found.')
    exit()

env = gym.make(env)
done = False
observation = env.reset()
cr = 0
while not done:
    action = step(model, observation)
    #action = np.random.choice(env.action_space.n)
    observation, reward, done, info = env.step(action)
    cr += reward
    env.render()
print('Finished with a cumulative reward of %.3f!' % cr)