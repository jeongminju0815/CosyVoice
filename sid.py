from tqdm import tqdm 

with open('/data/minju/TTS/CosyVoice_flow_cache/new_lj_vctk_libri_ray.txt', 'r') as f:
    lines = f.read().splitlines()

sid_dict = {}
sid_count = 10000
result = []
for line in tqdm(lines):
    wavpath, sid, text = line.split("||")
    if sid not in sid_dict.keys():
        new_sid = "Speaker" + str(sid_count)
        sid_dict[sid] = new_sid
        sid_count += 1
    new_text = text.replace(sid, sid_dict[sid])
    result.append((wavpath, sid_dict[sid], new_text))


with open('new_lj_vctk_libri_sid.txt', 'w') as f:
    for r in result:
        f.write(f"{r[0]}||{r[1]}||{r[2]}\n")

with open('sid_dict.txt', 'w') as f:
    for key, value in sid_dict.items():
        f.write(f"{key}|{value}\n")