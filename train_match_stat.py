# coding: utf-8
import sys, os
# import pandas as pd
sys.path.append(os.pardir)

import csv
import numpy as np
# from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet

# 데이터 읽기
# (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
player_file = open('player_datasets/player19-20.csv', 'r')
match_file = open('match_datasets/match19-20.csv', 'r')
rank_file = open('rank_datasets/rank19-20.csv', 'r')
lineup_file = open('lineup_datasets/lineup19-20.csv', 'r')

player_data = csv.reader(player_file)
player_data = list(player_data)
match_data = csv.reader(match_file)
rank_data = csv.reader(rank_file)
lineup_data = csv.reader(lineup_file)
# print(lineup_data.__format__)
# lineup2 = pd.read_csv(lineup_file)
# match_data : 첫번째 학습의 결과물
# lineup -> team(home / away) -> player_stat -> match_data (62)
# 선수 22명의 능력치가 하나의 데이터입력

def get_player_stat(player_num, player_team):
    for player_info in player_data:
        # print(player_info[1], player_info[2])
        if player_num == player_info[1] and player_team == player_info[2]:
            print(player_info[4:])
            return player_info[4:]
    print("[]")
    return []

train_data = []
for idx, lineup in enumerate(lineup_data): # line_up 한줄 씩
    if idx == 0: continue
    lineup_stats = []
    home_team = lineup[1] # home team 이름
    away_team = lineup[2] # away team 이름
    print(home_team, "vs", away_team)
    for col in range(3, 25, 2):
        print(lineup[col + 1], lineup[col], home_team)
        player_num = lineup[col]
        lineup_stats += get_player_stat(player_num, home_team)
    for col in range(25, len(lineup), 2):
        print(lineup[col + 1], lineup[col], away_team)
        player_num = lineup[col]
        lineup_stats += get_player_stat(player_num, away_team)
    # train_data.append(lineup_stats) # 경기당 선수 22명의 stats
    print(len(lineup_stats))
    train_data.append(lineup_stats)

train_datas = np.array(train_data)
print(train_datas)
print(train_datas.shape)
'''

network = TwoLayerNet(input_size=748, hidden_size=100, output_size=62)

#[380][748]

iters_num = 38
train_size = x_train.shape[0] # 380
batch_size = 10 # 1/38
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1) # 38

for i in range(iters_num): # 380개에서 랜덤으로 10개를 뽑아서 38번 랜덤 돌림
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    # 기울기 계산
    #grad = network.numerical_gradient(x_batch, t_batch) # 수치 미분 방식
    grad = network.gradient(x_batch, t_batch) # 오차역전파법 방식(훨씬 빠르다)
    
    # 갱신
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]
    
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    
    if i % iter_per_epoch == 0: # 한 배치의 계산된 정확도 
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(train_acc, test_acc)
'''
