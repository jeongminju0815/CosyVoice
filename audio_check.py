with open('/data/minju/TTS/CosyVoice_origin/examples/libritts/cosyvoice/data/long-custom/filelist.txt', 'r') as f:
    lines = f.read().splitlines()

for line in lines:
    wavpath, sid, text = line.split('|')
    if not sid in wavpath:
        print(line)