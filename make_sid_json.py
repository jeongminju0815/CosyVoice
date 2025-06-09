import json
sid_dict = {}
first = 1069

for i in range(10000, 12994):
    sid_dict[first] = f"Speaker{i}"
    first += 1

with open('eng_sid.json', 'w') as f:
    json.dump(sid_dict, f, indent=4)