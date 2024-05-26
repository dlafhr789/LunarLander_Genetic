import gymnasium as gym
import numpy as np
import os

GEN_NUM = 512


# 인공 신경만 클래스 만들기
# [참고] https://airsbigdata.tistory.com/195
def generate_rand_weight():
    w = []

    for i in range(GEN_NUM):
        w1 = np.random.rand(8, 10)
        w2 = np.random.rand(10, 10)
        w3 = np.random.rand(10, 4)

        w.append([w1, w2, w3])

    w_array = np.array(w, dtype=object)
    np.savez('weights.npz', w=w_array)


# 시그모이드 함수
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


def softmax(x: np.array):
    max_x = np.max(x)
    exp_x = np.exp(x - max_x)
    exp_x_sum = np.sum(exp_x)
    y = exp_x / exp_x_sum
    return y


def forward(network, x):
    w1, w2, w3 = network[0], network[1], network[2]
    b1, b2, b3 = np.ones(10), np.ones(10), np.ones(4)

    # Layer
    a1 = np.matmul(x, w1) + b1
    z1 = sigmoid(a1)
    a2 = np.matmul(z1, w2) + b2
    z2 = sigmoid(a2)
    a3 = np.matmul(z2, w3) + b3

    y = softmax(a3)
    return y


# LunarLander 환경 생성
env = gym.make("LunarLander-v2", render_mode="None")

# 임의의 가중치 생성
generate_rand_weight()
network = np.load('weights.npz', allow_pickle=True)['w']

for i in range(GEN_NUM):
    observation, info = env.reset()
    total_reward = 0
    while True:
        action = np.argmax(forward(network[i], observation))
        observation, reward, done, truncated, info = env.step(action)
        # print(action)
        total_reward += reward
        if done or truncated:
            print(total_reward)
            break

