with open('/data/minju/TTS/CosyVoice_origin/examples/libritts/cosyvoice/SCRIPT_DATASET/multispeaker_24k_valid.txt', 'r') as f:
    lines = f.read().splitlines()

result = []
for line in lines:
    wavpath, sid, text = line.split("|")
    utt = wavpath.split("/")[-1].replace(".wav", '')
    #spk = utt.split("-")[1] #for semi
    if "SON_SECRETARY_24k" in wavpath:
        spk = wavpath.split("/")[-3] #for son
    if ("slx_speakers" in wavpath) or ("FREEMODEL" in wavpath):
        spk = wavpath.split("/")[-4] + "_" + wavpath.split("/")[-3] #for ploonet
    if "sentimental_AIHUB" in wavpath:
        spk = "-".join(wavpath.split("/")[-1].split("-")[:4])
    if "EMOTION_AIHUB" in wavpath:
        spk = wavpath.split("/")[-2]
    if "A-NX-E-005" in wavpath:
        spk = "sunbeacon"
    if "A-A2-A-028" in wavpath:
        spk = "chimneycon"
    if "A-A1-F-019" in wavpath:
        spk = "mzcon"
    if "A-H2-E-020" in wavpath:
        spk = "happycon"
    if "A-A3-F-031" in wavpath:
        spk = "grumblingcon"
    if "M-H3-E-004" in wavpath:
        spk = "sweetcon"
    if "K-H2-A-009" in wavpath:
        spk = "madamcon"
    if "KARDER" in wavpath:
        spk = "cardergarden"
    if "aihub_multispeaker" in wavpath:
        spk = wavpath.split("/")[-2]
    new_text = f"{spk}<|endofprompt|>{text}"
    result.append([wavpath, spk, new_text])

with open('/data/minju/TTS/CosyVoice_origin/examples/libritts/cosyvoice/SCRIPT_DATASET_SID/multispeaker_24k_valid.txt', 'w') as f:  
    for w, spk, t in result:
        f.write(f"{w}||{spk}||{t}\n")