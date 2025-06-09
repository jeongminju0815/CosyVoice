from cosyvoice.utils.frontend_utils import contains_chinese, replace_blank, replace_corner_mark, remove_bracket, spell_out_number, split_paragraph, is_only_punctuation
import inflect
import re
from hyperpyyaml import load_hyperpyyaml
# from tn.chinese.normalizer import Normalizer as ZhNormalizer
from tn.english.normalizer import Normalizer as EnNormalizer
from cosyvoice.tokenizer.tokenizer import get_qwen_tokenizer
from tqdm import tqdm
# zh_tn_model = ZhNormalizer(remove_erhua=False, full_to_half=False, overwrite_cache=True)
en_tn_model = EnNormalizer()
inflect_parser = inflect.engine()

def text_normalize(text, split=True, text_frontend=True):
    text = text.strip()


    text = en_tn_model.normalize(text)
    text = spell_out_number(text, inflect_parser)
    text = replace_blank(text)
    text = re.sub(r"[\\:#\"`;\\-]+", "", text)
    text = text.lower()
    text = text.strip("'")
    text = text.strip(" ")
    return text


with open('/data/minju/TTS/CosyVoice_flow_cache/examples/libritts/lj_vctk_libri_train.txt', 'r') as f:
    lines = f.read().splitlines()
result = []
count = 10000
sid_dict = {}
try:
    for line in tqdm(lines):
        wavpath, sid, text = line.split("|")
        # print("----")
        # print(text)
        text = text_normalize(text)
        # print("after:", text)
        # print("-----")
        if sid not in sid_dict.keys():
            sid_dict[sid] = "Speaker"+str(count)
            count += 1
        text = sid_dict[sid] + "<|endofprompt|>" + text
        result.append([wavpath, sid_dict[sid], text])

    with open('engsid.txt', 'w') as f:
        for key, value in sid_dict.items():
            f.write(f"{key}|{value}\n")

    with open('new_lj_vctk_libri.txt', 'w') as f:
        for r in result:
            f.write(f"{r[0]}||{r[1]}||{r[2]}\n")
except:
    pass