# coding: utf-8
import sys, os
# import pandas as pd
sys.path.append(os.pardir)

import csv
import numpy as np
# from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet

def get_player_stat(player_num, player_team, name):
    for player_info in player_data:
        if player_num == player_info[1] and player_team == player_info[2]:
            return list(map(np.float64, player_info[4:]))
    return []

# 데이터 읽기
# (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
network = TwoLayerNet(input_size=748, hidden_size=100, output_size=62)

for year in range(19, 24):
    player_file = open(f'player_datasets/player{year}-{year + 1}.csv', 'r')
    match_file = open(f'match_datasets/match{year}-{year + 1}.csv', 'r')
    rank_file = open(f'rank_datasets/rank{year}-{year + 1}.csv', 'r')
    lineup_file = open(f'lineup_datasets/lineup{year}-{year + 1}.csv', 'r')
# params = open('./params.txt', 'rw')

# params_data = params.read(params)

    player_data = list(csv.reader(player_file))
    match_data = csv.reader(match_file)
    rank_data = csv.reader(rank_file)
    lineup_data = csv.reader(lineup_file)

# match_data : 첫번째 학습의 결과물
# lineup -> team(home / away) -> player_stat -> match_data (62)
# 선수 22명의 능력치가 하나의 데이터입력

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
# print(train_datas)
# print(train_datas.shape)

    answer_datas = list()
    for idx, match in enumerate(match_data):
        if idx == 0: continue
        tmp = match[3:34] + match[36:]
        answer_datas.append(list(map(np.float64, tmp)))

    answer_datas = np.array(answer_datas)
    # print(answer_datas)
    # print(answer_datas.shape)

    norm_answer = answer_datas.copy()
    norm_min = np.min(norm_answer, axis=0)
    norm_max = np.max(norm_answer, axis=0)
    norm_answer = (norm_answer - norm_min) / (norm_max - norm_min)

    # print(norm_answer)
    #[380][748]

    iters_num = 38
    train_size = norm_answer.shape[0] # 380
    batch_size = 10 # 1/38
    learning_rate = 0.1

    train_loss_list = []
    train_acc_list = []
    test_acc_list = []

    iter_per_epoch = max(train_size / batch_size, 1) # 38
# network.params = params_data
# for i in range(iters_num): # 380개에서 랜덤으로 10개를 뽑아서 38번 랜덤 돌림
    # batch_mask = np.random.choice(train_size, batch_size)
    # x_batch = x_train[batch_mask]
    # t_batch = t_train[batch_mask]
    
    # 기울기 계산
    #grad = network.numerical_gradient(x_batch, t_batch) # 수치 미분 방식
    grad = network.gradient(train_datas, norm_answer) # 오차역전파법 방식(훨씬 빠르다)

    # 갱신
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]
        with open(f"{key}.csv", "w") as file:
            writer = csv.writer(file)
            if (key == 'b1' or key == 'b2') :
                writer.writerows([network.params[key]])
            else:
                writer.writerows(network.params[key])

    # print(network.params)
    loss = network.loss(train_datas, norm_answer)
    train_loss_list.append(loss)
    test_acc = network.accuracy(train_datas, norm_answer)
    # print(test_acc)
    # test_acc_list.append(test_acc)

        # if i % iter_per_epoch == 0: # 한 배치의 계산된 정확도 
        #     train_acc = network.accuracy(train_datas, norm_answer)
            # test_acc = network.accuracy(x_test, t_test)
        #     train_acc_list.append(train_acc)
        #     # test_acc_list.append(test_acc)
        #     print(train_acc)


# print(test_acc_list)