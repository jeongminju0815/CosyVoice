import ray
from cosyvoice.utils.frontend_utils import contains_chinese, replace_blank, replace_corner_mark, remove_bracket, spell_out_number, split_paragraph, is_only_punctuation
import inflect
import re
from hyperpyyaml import load_hyperpyyaml
from tn.english.normalizer import Normalizer as EnNormalizer
from cosyvoice.tokenizer.tokenizer import get_qwen_tokenizer
from tqdm import tqdm
import time

# Ray 초기화
ray.init()

# 전역 객체를 Ray actor로 만들어 프로세스 간 공유
@ray.remote
class TextNormalizer:
    def __init__(self):
        self.en_tn_model = EnNormalizer()
        self.inflect_parser = inflect.engine()
    
    def normalize(self, text):
        text = text.strip()
        text = self.en_tn_model.normalize(text)
        text = spell_out_number(text, self.inflect_parser)
        text = replace_blank(text)
        text = re.sub(r"[\\:#\"`;\\-]+", "", text)
        text = text.lower()
        text = text.strip("'")
        text = text.strip(" ")
        return text

# 배치로 처리하는 함수
@ray.remote
def process_batch(lines, start_idx, end_idx, normalizer_actor, count_start):
    batch_results = []
    sid_dict = {}
    count = count_start
    
    for i in range(start_idx, end_idx):
        if i >= len(lines):
            break
            
        line = lines[i]
        try:
            wavpath, sid, text = line.split("|")
            # 텍스트 정규화
            text = ray.get(normalizer_actor.normalize.remote(text))
            
            if sid not in sid_dict:
                sid_dict[sid] = f"Speaker{count}"
                count += 1
            
            text = sid_dict[sid] + "<|endofprompt|>" + text
            batch_results.append([wavpath, sid_dict[sid], text])
        except Exception as e:
            print(f"Error processing line {i}: {e}")
            continue
    
    return batch_results, sid_dict, count

def main():
    start_time = time.time()
    
    # 파일 읽기
    with open('/data/minju/TTS/CosyVoice_flow_cache/examples/libritts/lj_vctk_libri_train.txt', 'r') as f:
        lines = f.read().splitlines()

    # Ray actor 생성
    normalizer_actor = TextNormalizer.remote()
    
    # 배치 크기 설정
    batch_size = 1000
    num_batches = (len(lines) + batch_size - 1) // batch_size
    
    # 병렬 처리
    futures = []
    count_start = 10000
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(lines))
        
        # 각 배치마다 고유한 count 시작값 설정
        batch_count_start = count_start + i * 1000  # 충분한 여유를 두고 설정
        
        future = process_batch.remote(lines, start_idx, end_idx, normalizer_actor, batch_count_start)
        futures.append(future)
    
    # 결과 수집
    results = []
    all_sid_dict = {}
    
    # 진행률 표시
    for future in tqdm(futures, desc="Processing batches"):
        batch_results, batch_sid_dict, _ = ray.get(future)
        results.extend(batch_results)
        all_sid_dict.update(batch_sid_dict)
    
    # 결과 저장
    with open('engsid_ray.txt', 'w') as f:
        for key, value in all_sid_dict.items():
            f.write(f"{key}|{value}\n")
    
    with open('new_lj_vctk_libri_ray.txt', 'w') as f:
        for r in results:
            f.write(f"{r[0]}||{r[1]}||{r[2]}\n")
    
    # Ray 종료
    ray.shutdown()
    
    end_time = time.time()
    print(f"Total processing time: {end_time - start_time:.2f} seconds")
    print(f"Processed {len(results)} lines")

if __name__ == "__main__":
    main()