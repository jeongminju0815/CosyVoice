import argparse
import logging
import glob
import os
from tqdm import tqdm


logger = logging.getLogger()


def main():
    with open(args.txt_dir, 'r') as f:
        lines = f.read().splitlines()

    utt2wav, utt2text, utt2spk, spk2utt = {}, {}, {}, {}
    
    #/data/DATASET/TTS/son_secretary/train/wavs/손비서 이예은 성우/230725-LYE-0001.wav|7|죄송합니다. 지금은 이경일 대표님께서 전화를 받지 못하는 상황이에요. 메모를 남겨주시면 제가 전달해드릴게요!
    for line in lines:
        wavpath, sid, text = line.split("||")
        utt = wavpath.split("/")[-1].replace(".wav", '')
        #spk = utt.split("-")[1] #for semi
        if "son_secretary" in wavpath:
            spk = wavpath.split("/")[-2] #for son
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
        spk = sid
        utt = f"{spk}_{utt}"
        print(spk, utt)
        utt2wav[utt] = wavpath
        utt2text[utt] = text
        utt2spk[utt] = spk
        if spk not in spk2utt:
            spk2utt[spk] = []
        spk2utt[spk].append(utt)


    with open('{}/wav.scp'.format(args.des_dir), 'w') as f:
        for k, v in utt2wav.items():
            f.write('{} {}\n'.format(k, v))
    with open('{}/text'.format(args.des_dir), 'w') as f:
        for k, v in utt2text.items():
            f.write('{} {}\n'.format(k, v))
    with open('{}/utt2spk'.format(args.des_dir), 'w') as f:
        for k, v in utt2spk.items():
            f.write('{} {}\n'.format(k, v))
    with open('{}/spk2utt'.format(args.des_dir), 'w') as f:
        for k, v in spk2utt.items():
            print(k, len(v))
            f.write('{} {}\n'.format(k, ' '.join(v)))
    print(spk2utt.keys())
    print(len(spk2utt.keys()))
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--txt_dir',
                        type=str)
    parser.add_argument('--des_dir',
                        type=str)
    args = parser.parse_args()
    main()
