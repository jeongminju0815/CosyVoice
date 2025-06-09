import re
import ast
import os

from cosyvoice.text.ko_dictionary import english_dictionary, etc_dictionary
from cosyvoice.text.hangulize import EnglishToHangul

# from ko_dictionary import english_dictionary, etc_dictionary
# from hangulize import EnglishToHangul

# 241211 english to korean (IPA) 로직 추가
eng_to_kor_flag = os.getenv('ENG') if os.getenv('ENG') != None else False
print(f'* Eng To Korean: {eng_to_kor_flag}')

quote_checker = """([`"'＂“‘])(.+?)([`"'＂”’])"""

num_to_kor = {
        '0': '영',
        '1': '일',
        '2': '이',
        '3': '삼',
        '4': '사',
        '5': '오',
        '6': '육',
        '7': '칠',
        '8': '팔',
        '9': '구',
}

unit_to_kor1 = {
        '%': '퍼센트',
        'cm': '센치미터',
        'mm': '밀리미터',
        'km': '킬로미터',
        'kg': '킬로그람',
        '@' : ' 골벵이 ', #24.08.26 워크센터 기준 통합
}
unit_to_kor2 = {
        'm': '미터',
}

upper_to_kor = {
        'A': '에이',
        'B': '비',
        'C': '씨',
        'D': '디',
        'E': '이',
        'F': '에프',
        'G': '지',
        'H': '에이치',
        'I': '아이',
        'J': '제이',
        'K': '케이',
        'L': '엘',
        'M': '엠',
        'N': '엔',
        'O': '오',
        'P': '피',
        'Q': '큐',
        'R': '알',
        'S': '에스',
        'T': '티',
        'U': '유',
        'V': '브이',
        'W': '더블유',
        'X': '엑스',
        'Y': '와이',
        'Z': '지',
}

lower_to_kor = {
        'a': '에이',
        'b': '비',
        'c': '씨',
        'd': '디',
        'e': '이',
        'f': '에프',
        'g': '지',
        'h': '에이치',
        'i': '아이',
        'j': '제이',
        'k': '케이',
        'l': '엘',
        'm': '엠',
        'n': '엔',
        'o': '오',
        'p': '피',
        'q': '큐',
        'r': '알',
        's': '에스',
        't': '티',
        'u': '유',
        'v': '브이',
        'w': '더블유',
        'x': '엑스',
        'y': '와이',
        'z': '지',
}

def ends_with_punctuation(text, _punct):
    if len(text.strip()) != 0: # 공백일 땐 그대로
        puncutation = ('.', ',', '!', '?')
        if not text.strip().endswith(puncutation):
            text += _punct
        # 250210 , 로 끝날 때 . 으로 변경
        if text[-1] == ",":
            text = text[:-1] + _punct


    return text

def normalize(text):
    text = text.strip()

    text = re.sub('\(\d+일\)', '', text)
    text = re.sub('\([⺀-⺙⺛-⻳⼀-⿕々〇〡-〩〸-〺〻㐀-䶵一-鿃豈-鶴侮-頻並-龎]+\)', '', text)

    ## 24.08.26 워크센터 기준 로직 추가
    text = re.sub(date_checker, lambda x: date_to_korean(x), text)
    text = re.sub(phone_number_checker, lambda x: phone_number_to_korean(x), text)
    text = re.sub(email_checker, lambda x: email_to_korean(x), text)

    text = normalize_with_dictionary(text, etc_dictionary)
    # 24.12.11 영어 발음 한국어 변환 로직 추가
    text = normalize_english(text, eng_to_kor_flag) 
    text = re.sub('[a-zA-Z]+', normalize_upper, text)

    # text = normalize_quote(text)
    text = normalize_number(text)

    text = re.sub(r"\s+", " ", text)
    text = re.sub('.,?!\.,?!{3,}', '', text)
    text = re.sub('([?!.,]*)', lambda x: x.group(0)[:3], text)
    text = re.sub('[^가-힣 .!?,]+', '', text)
    text = re.sub(r"\s+", " ", text)

    # 24.07.31 문장 끝에 문장부호 없을 경우, 마침표 붙여줌
    text = ends_with_punctuation(text, ".")
    
    print(f"Cleaner Text: {text}")

    return text

def normalize_with_dictionary(text, dic):
    pattern = re.compile('|'.join(re.escape(key) for key in dic.keys()))
    if any(key in text for key in dic.keys()):
        pattern = re.compile('|'.join(re.escape(key) for key in dic.keys()))
        return pattern.sub(lambda x: dic[x.group()], text)
    else:
        return text

def normalize_english(text, eng_to_kor):
    def fn(m):
        word = m.group()
        if word in english_dictionary:
            return english_dictionary.get(word)
        else:
            return word

    text = re.sub("([A-Za-z]+)", fn, text)

    #24.12.11 영어 발음 한국어 변환 로직 추가
    if eng_to_kor:
        # 일반적인 영단어는 사전에서 찾아서 변환
        try:
            ETH = EnglishToHangul()

            p = re.compile('[a-zA-Z]')

            output = []

            words = text.split(' ')
            output.clear()
            for word in words:
                if word and p.match(word[0]) and not word.isupper():
                    output.append(ETH.transcribe(word=word))
                elif len(word) > 1 and p.match(word[1]) and not word.isupper():
                    output.append(ETH.transcribe(word=word))
                else:
                    output.append(word)

            text = ' '.join(output)
        except Exception as e:
            print('eng to kor error', text)
            return text
    return text

def normalize_upper(text):
    text = text.group(0)
    if all([char.isupper() for char in text]):
        return "".join(upper_to_kor[char] for char in text)
    else:
        return text


number_checker = "([+-]?\d[\d,]*)[\.]?\d*"
count_checker = "(시|명|가지|살|마리|포기|송이|수|톨|통|개|벌|척|채|다발|그루|자루|줄|켤레|그릇|잔|마디|상자|사람|곡|병|판)" #24.07.31 '점' 제외

## 24.08.26 워크 센터 기준 추가
# (숫자아님)010("-",".", "")(네자리숫자)("-",".", "")(네자리숫자)(숫자아님)
phone_number_checker = r'(?<!\d)010([-.]?)\d{4}\1\d{4}(?!\d)'

# (숫자아님)(네자리연도 또는 두자리연도)("-","/", ".")?(두자리월)("-","/", "")(두자리일)(숫자아님)
date_checker = r'(?<!\d)(?:\d{4}|\d{2})([-./]?)(0[1-9]|1[0-2])\1(0[1-9]|[12][0-9]|3[01])(?!\d)'

email_checker = r'\b[\w.%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'


def normalize_number(text):
    text = normalize_with_dictionary(text, unit_to_kor1)
    text = normalize_with_dictionary(text, unit_to_kor2)
    text = re.sub(number_checker + count_checker,
            lambda x: number_to_korean(x, True), text)
    text = re.sub(number_checker,
            lambda x: number_to_korean(x, False), text)
    return text

num_to_kor1 = [""] + list("일이삼사오육칠팔구")
num_to_kor2 = [""] + list("만억조경해")
num_to_kor3 = [""] + list("십백천")

#count_to_kor1 = [""] + ["하나","둘","셋","넷","다섯","여섯","일곱","여덟","아홉"]
count_to_kor1 = [""] + ["한","두","세","네","다섯","여섯","일곱","여덟","아홉"]

count_tenth_dict = {
        "십": "열",
        "두십": "스물",
        "세십": "서른",
        "네십": "마흔",
        "다섯십": "쉰",
        "여섯십": "예순",
        "일곱십": "일흔",
        "여덟십": "여든",
        "아홉십": "아흔",
}



def number_to_korean(num_str, is_count=False):
    if is_count:
        num_str, unit_str = num_str.group(1), num_str.group(2)
    else:
        num_str, unit_str = num_str.group(), ""

    num_str = num_str.replace(',', '')

    #(기호0개 이상)0(0이 아닌 다른 숫자 있는 경우) 확인 ex) 01, +010
    pattern = r'^[+-]*0[0-9]*[1-9]\d*$'
    match = re.match(pattern, num_str)

    if match:
        num = re.sub(r'0+', '', num_str, 1) # ex) 01 -> 1, +010 -> +10
    else:
        num = ast.literal_eval(num_str)

    if num == 0:
        return "영"

    check_float = num_str.split('.')
    if len(check_float) == 2:
        digit_str, float_str = check_float
    elif len(check_float) >= 3:
        raise Exception(" [!] Wrong number format")
    else:
        digit_str, float_str = check_float[0], None

    if is_count and float_str is not None:
        raise Exception(" [!] `is_count` and float number does not fit each other")

    digit = int(digit_str)
    if digit_str.startswith(("-", "+")): # '=' case 추가
        digit, digit_str = abs(digit), str(abs(digit))

    kor = ""
    size = len(str(digit))
    tmp = []
    zero_flag = False # ex) 01 -> 1과 같이 변경된 경우 or 소수의 정수 부분이 0일 때 처리
    
    if (size == 1) and (digit == 0):
        tmp += '영'

    for i, v in enumerate(digit_str, start=1): 
        v = int(v)

        if v != 0:
            zero_flag = False
            if is_count and len(digit_str) < 3: #250211 세자리 수 이상 카운트 로직 해제
                tmp += count_to_kor1[v]
            else:
                ### 24.07.31 2118 -> 이천일백일십팔년 문제 수정 코드
                if v == 1:
                    if len(digit_str) != int(i) and len(digit_str) != 1:
                        pass
                    else:
                        tmp += num_to_kor1[v]
                else:
                    tmp += num_to_kor1[v]
            tmp += num_to_kor3[(size - i) % 4]
        else:
            if zero_flag:
                size += 1
        
        if (v == 0) and (i == 1):
            zero_flag = True
            size += 1

        if (size - i) % 4 == 0 and len(tmp) != 0:
            kor += "".join(tmp)
            tmp = []
            kor += num_to_kor2[int((size - i) / 4)]

    if zero_flag: #소수의 정수 부분이 0일 때
        kor += "".join(tmp)

    if is_count and len(digit_str) < 3: #250211 세자리 수 이상 카운트 로직 해제
        if kor.startswith("한") and len(kor) > 1:
            kor = kor[1:]

        if any(word in kor for word in count_tenth_dict):
            kor = re.sub(
                    '|'.join(count_tenth_dict.keys()),
                    lambda x: count_tenth_dict[x.group()], kor)

    if not is_count and kor.startswith("일") and len(kor) > 1:
        kor = kor[1:]

    if float_str is not None:
        kor += "쩜 "
        kor += re.sub('\d', lambda x: num_to_kor[x.group()], float_str)

    if num_str.startswith("+"):
        kor = "플러스 " + kor
    elif num_str.startswith("-"):
        kor = "마이너스 " + kor

    return kor + unit_str

## 24.08.26 워크센터 기준 추가
def date_to_korean(match):
    date_str = match.group()
    components = re.split(r'[-./]', date_str)

    if len(components) == 3:
        return f"{components[0]}년 {int(components[1])}월 {int(components[2])}일"
    else:
        return date_str

def phone_number_to_korean(match):
    phone_number_str = match.group()
    converted_str = ''.join([phone_number(digit) for digit in re.sub(r'[-.]', '. ', phone_number_str)])

    return converted_str

def phone_number(digit):
    korean_digits = {
        '0': '공',
        '1': '일',
        '2': '이',
        '3': '삼',
        '4': '사',
        '5': '오',
        '6': '육',
        '7': '칠',
        '8': '팔',
        '9': '구'
    }
    return korean_digits.get(digit, digit)

def email_to_korean(match):
    email = match.group().replace('.', ' 쩜 ')
    parts = email.split('@')
    parts[1] = ' '.join([email_domain(part) for part in parts[1].split()])
    email = '@'.join(parts)

    converted_email = ''.join([lower_to_kor.get(char, char) for char in email])
    return converted_email

def email_domain(domain):
    korean_domain = {
        'naver': '네이버',
        'gmail': '지메일',
        'daum': '다음',
        'nate': '네이트',
        'saltlux': '솔트룩스',
        'yahoo' : '야후',
        'outlook' : '아웃룩',
        'kakao': '카카오',
        'tistory': '티스토리',
        'empal': '엠팔',
        'hanmail': '한메일',
        'com': '컴',
        'net': '넷'
    }
    return korean_domain.get(domain, domain)

if __name__ == "__main__":
    def test_normalize(text):
        print(text)
        print(normalize(text))
        print("="*50)

    # test_normalize("JTBC는 JTBCs를 DY는 A가 Absolute")
    test_normalize("오늘(13일) 3,631마리 강아지가")
    test_normalize("12월 0.001일")
    test_normalize("12월 01일")
    test_normalize("12월 000000001일")
    test_normalize("12월 00.00001일")
    test_normalize('이를 다 합치면 교원 정원은 기존 대비 2232그루 감원된다. 이번 교원 22명 1포기 232명 정원 감축은 계속해서 줄고 있는 학령인구를 고려한 조치다. 도 교육청의 자체 추계를 보면 올해 13만6043명인 도내 초·중·고 학생 수는 오는 2029년 들어 12만2071명으로 1만3972명(11.4%) 줄어들 전망이다.')
    test_normalize("챗GPT, chatgpt 차이점이 있나요? 같은 검색어로 비교해보니 속도라든가 검색결과가 좀 차이가 나던데 더 우수하고 안 하고의 차이같은게 있나요?")
    
    # test_normalize("60.3%")
    # test_normalize('"저돌"(猪突) "반갑습니다"입니다.')
    # test_normalize('비대위원장이 지난 1월 이런 말을 했습니다. “난 그냥 산돼지처럼 돌파하는 스타일이다”')
    # test_normalize("지금은 -12.35%였고 종류는 5가지와 19가지, 그리고 55가지였다")
    # test_normalize("JTBC는 TH와 K 양이 2017년 9월 12일 오후 12시에 24살이 된다")
    # print(list(hangul_to_jamo(list(hangul_to_jamo('비대위원장이 지난 1월 이런 말을 했습니다? “난 그냥 산돼지처럼 돌파하는 스타일이다”')))))