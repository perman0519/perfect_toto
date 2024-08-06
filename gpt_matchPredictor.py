import numpy as np
# coding: utf-8
import sys, os
# import pandas as pd
sys.path.append(os.pardir)

import csv
# from dataset.mnist import load_mnist

# 데이터 읽기
# (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
player_file = open('player_datasets/player19-20.csv', 'r')
match_file = open('match_datasets/match19-20.csv', 'r')
rank_file = open('rank_datasets/rank19-20.csv', 'r')
lineup_file = open('lineup_datasets/lineup19-20.csv', 'r')

player_data = list(csv.reader(player_file))
match_data = csv.reader(match_file)
rank_data = csv.reader(rank_file)
lineup_data = csv.reader(lineup_file)

# match_data : 첫번째 학습의 결과물
# lineup -> team(home / away) -> player_stat -> match_data (62)
# 선수 22명의 능력치가 하나의 데이터입력

def get_player_stat(player_num, player_team, name):
    for player_info in player_data:
        if player_num == player_info[1] and player_team == player_info[2]:
            return list(map(np.float64, player_info[4:]))
    return []

train_data = []
for idx, lineup in enumerate(lineup_data): # line_up 한줄 씩
    if idx == 0: continue
    lineup_stats = []
    home_team = lineup[1] # home team 이름
    away_team = lineup[2] # away team 이름
    for col in range(3, 25, 2):
        player_num = lineup[col]
        lineup_stats += get_player_stat(player_num, home_team, lineup[col + 1])
    for col in range(25, len(lineup), 2):
        player_num = lineup[col]
        lineup_stats += get_player_stat(player_num, away_team, lineup[col + 1])
    train_data.append(lineup_stats)

train_datas = np.array(train_data)
print(train_datas)
print(train_datas.shape)

answer_datas = list()
for idx, match in enumerate(match_data):
    if idx == 0: continue
    tmp = match[3:34] + match[36:]
    answer_datas.append(list(map(np.float64, tmp)))

answer_datas = np.array(answer_datas)
print(answer_datas)
print(answer_datas.shape)

class SoccerMatchPredictor:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size) / np.sqrt(input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, hidden_size) / np.sqrt(hidden_size)
        self.b2 = np.zeros((1, hidden_size))
        self.W3 = np.random.randn(hidden_size, output_size) / np.sqrt(hidden_size)
        self.b3 = np.zeros((1, output_size))

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return x > 0

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.relu(self.z2)
        self.z3 = np.dot(self.a2, self.W3) + self.b3
        return self.z3

    def loss(self, y_true, y_pred):
        return np.mean(np.square(y_true - y_pred))

    def backward(self, X, y, output, learning_rate):
        m = X.shape[0]

        dZ3 = 2 * (output - y) / m
        dW3 = np.dot(self.a2.T, dZ3)
        db3 = np.sum(dZ3, axis=0, keepdims=True)

        dZ2 = np.dot(dZ3, self.W3.T) * self.relu_derivative(self.z2)
        dW2 = np.dot(self.a1.T, dZ2)
        db2 = np.sum(dZ2, axis=0, keepdims=True)

        dZ1 = np.dot(dZ2, self.W2.T) * self.relu_derivative(self.z1)
        dW1 = np.dot(X.T, dZ1)
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        self.W3 -= learning_rate * dW3
        self.b3 -= learning_rate * db3
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

    def train(self, X, y, epochs, learning_rate, batch_size=31):
        for epoch in range(epochs):
            for i in range(0, len(X), batch_size):
                batch_X = X[i:i+batch_size]
                batch_y = y[i:i+batch_size]
                
                output = self.forward(batch_X)
                self.backward(batch_X, batch_y, output, learning_rate)
            
            if epoch % 10 == 0:
                current_loss = self.loss(y, self.forward(X))
                print(f'Epoch {epoch}, Loss: {current_loss}')

# 데이터 전처리
def preprocess_data(X, y):
    # 정규화
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    y = (y - np.mean(y, axis=0)) / np.std(y, axis=0)
    return X, y

# 데이터 로드 (이미 처리된 데이터 사용)
# train_datas = np.load('train_datas.npy')
# answer_datas = np.load('answer_datas.npy')

# 데이터 전처리
X, y = preprocess_data(train_datas, answer_datas)

# 모델 초기화 및 학습
input_size = X.shape[1]
output_size = y.shape[1]
model = SoccerMatchPredictor(input_size=input_size, hidden_size=256, output_size=output_size)
model.train(X, y, epochs=1000, learning_rate=0.001, batch_size=31)

# 새로운 경기에 대한 예측 (예시)
new_match_data = X[0:1]  # 첫 번째 경기 데이터를 예시로 사용
prediction = model.forward(new_match_data)
print("Predicted match stats:", prediction)

# 예측 결과 역정규화 (실제 스케일로 변환)
prediction_original_scale = prediction * np.std(answer_datas, axis=0) + np.mean(answer_datas, axis=0)
print("Predicted match stats (original scale):", prediction_original_scale)