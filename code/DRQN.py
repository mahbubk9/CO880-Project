#IMPORTS
import gym 
import random
import numpy as np
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, TimeDistributed, Flatten, LSTM
from tensorflow.keras.optimizers import RMSprop
import datetime

#SET SEED
np.random.seed(168)
tf.random.set_seed(168)

#DRQN 
class DRQN():
    def __init__(self, env, batch_size=64, max_experiences=5000):
        self.env = env
        self.input_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.max_experiences = max_experiences
        self.memory = deque(maxlen=self.max_experiences)
        self.batch_size = batch_size
        self.gamma = 1.0
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
        self.model = self.build_model()
        self.target_model = self.build_model()
                
    def build_model(self):
        model = Sequential()
        model.add(TimeDistributed(Conv2D(32, 8, (4,4), activation='relu', padding='valid'), input_shape=(SEQUENCE, IMG_SIZE, IMG_SIZE, 1)))
        model.add(TimeDistributed(Conv2D(64, 4, (2,2), activation='relu', padding='valid')))
        model.add(TimeDistributed(Conv2D(64, 3, (1,1), activation='relu', padding='valid')))
        model.add(TimeDistributed(Flatten()))
        model.add(LSTM(512))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(self.action_size))
        model.compile(loss='mse', optimizer=RMSprop(lr=0.00025, epsilon=self.epsilon_min), metrics=['accuracy'])
        return model
            
    def get_action(self, state):
        if np.random.random() <= self.epsilon:
            return self.env.action_space.sample()
        else:
            pred = self.model.predict(tf.expand_dims(state, 0))
            return np.argmax(pred)

    def add_experience(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        self.update_epsilon()

    def replay(self, episode):
        x_batch, y_batch = [], []
        minibatch = random.sample(self.memory, min(len(self.memory), self.batch_size))
        
        for state, action, reward, next_state, done in minibatch:
            y_target = self.target_model.predict(tf.expand_dims(state, 0))
            y_target[0][action] = reward if done else reward + self.gamma * np.max(self.model.predict(tf.expand_dims(next_state, 0))[0])
            x_batch.append(state)
            y_batch.append(y_target[0])

        self.model.fit(np.array(x_batch), np.array(y_batch), batch_size=len(x_batch), verbose=0)   

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon_decay * self.epsilon)

#ENV INITIALISATION AND PREPROCEESSING STATE
def initialize_env(env):
  initial_state = env.reset()
  initial_done_flag = False
  initial_rewards = 0
  return initial_state, initial_done_flag, initial_rewards  


def preprocess_state(image, img_size):
    img_temp = image[31:195]
    img_temp = tf.image.rgb_to_grayscale(img_temp)
    img_temp = tf.image.resize(img_temp, [img_size, img_size],
                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    img_temp = tf.cast(img_temp, tf.float32)
    return img_temp        

def combine_images(new_img, prev_img, img_size, seq=4):
    if len(prev_img.shape) == 4 and prev_img.shape[0] == seq:
        im = np.concatenate((prev_img[1:, :, :], tf.reshape(new_img, [1, img_size, img_size, 1])), axis=0)
    else:
        im = np.stack([new_img] * seq, axis=0)
    return im

#GAME PLAY
def play_game(agent, state, done, rewards):    
    while not done:
        action = agent.get_action(state)
        next_state, reward, done, _ = env.step(action)

        next_state = preprocess_state(next_state, IMG_SIZE)
        next_state = combine_images(new_img=next_state, prev_img=state, img_size=IMG_SIZE, seq=SEQUENCE)
        
        agent.add_experience(state, action, reward, next_state, done)

        state = next_state
        rewards += reward   
    return rewards

#TRAINING METHOD
def train_agent(env, episodes, agent):
  from collections import deque
  

  scores = deque(maxlen=100)

  for episode in range(episodes):
    state, done, rewards = initialize_env(env) 
    state = preprocess_state(state, IMG_SIZE)
    state = combine_images(new_img=state, prev_img=state, img_size=IMG_SIZE, seq=SEQUENCE)

    rewards = play_game(agent, state, done, rewards)
    scores.append(rewards)
    mean_score = np.mean(scores)

    if episode % 50 == 0:
        print(f'[Episode {episode}] - Average Score: {mean_score}')
        agent.target_model.set_weights(agent.model.get_weights())
        agent.target_model.save_weights(f'drqn/drqn_model_weights_{episode}')

    agent.replay(episode)

  print(f"Average Score: {np.mean(scores)}")


if __name__ == "__main__":
  env = gym.make('PongNoFrameskip-v4')
  IMG_SIZE = 84
  SEQUENCE = 4
  agent = DRQN(env)
  print(agent.input_size)
  print(agent.action_size)
  episodes = 200
  train_agent(env, episodes, agent)  



