import requests
import csv
from bs4 import BeautifulSoup

base_url = "https://www.fifacm.com"
# team_name = "manchester-united"

data = [
	['player_name', 'player_number', 'team', 'is_GK', "GK_Diving", "GK_Handling", "GK_Kicking", "GK_Positioning", "GK_Reflexes", "Crossing", "Finishing", "Heading_Acc", "Short_Pass", "Volleys", "Dribbling", "Curve", "FK_Acc", "Long_Pass", "Ball_Control", "Shot_Power", "Jumping", "Stamina", "Strength", "Long_Shots", "Acceleration", "Sprint_Speed", "Agility", "Reactions", "Balance", "Aggression", "Interceptions", "Att_Position", "Vision", "Penalties", "Composure", "Def_Awareness", "Stand_Tackle", "Slide_Tackle"],
]

# player_url_list = []

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

player_url = input("player_url: ")
response = requests.get(player_url, headers=headers)
print(response.status_code)
if response.status_code == 200:
	html = BeautifulSoup(response.text, "html.parser")

	position = html.select_one(".player-position").text.strip()
	is_GK = 0
	if position == "GK":
		is_GK = 1
	first_name = html.select_one(".player-firstname").text.strip()
	last_name = html.select_one(".player-lastname").text.strip()

	name = first_name + " " + last_name
	team_name = html.select_one(".d-inline > a").text.strip()
	player_num = html.select_one("div.row.player-left-area-2.mx-0.pb-2 > div.col-md-12.col-12 > div:nth-child(3) > div.d-inline.ml-3").text.strip()
	player_data = [name, player_num, team_name, is_GK]
	stats = html.select_one(".row.player-stats.mt-3.px-0")
	stats_list = stats.select(".sub-stat-rating")
	for (stat) in stats_list:
		player_data.append(int(stat.text.strip()))
	data.append(player_data)

print("crawling ok!")
print(data[1])