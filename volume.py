import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

# 1️⃣ 오디오 파일 로드
input_audio_path = "db_test.wav"  # 사용할 오디오 파일
y, sr = librosa.load(input_audio_path, sr=None)  # 샘플링 레이트 22050Hz

# 2️⃣ 멜 스펙트로그램 생성
S = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=512)

# 3️⃣ PCEN 적용
pcen_S, _ = librosa.pcen(S, sr=sr, hop_length=512, return_zf=True)


# 4️⃣ PCEN 적용된 신호를 시간 도메인으로 변환
y_pcen = librosa.feature.inverse.mel_to_audio(pcen_S, sr=sr, hop_length=512)

# 5️⃣ 결과 오디오 저장
output_audio_path = "output_pcen.wav"
sf.write(output_audio_path, y_pcen, sr)

print(f"PCEN 적용된 오디오 저장 완료: {output_audio_path}")