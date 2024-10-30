import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# 임베딩 모델 로드
encoder = SentenceTransformer('jhgan/ko-sroberta-multitask')

# 식당 관련 질문과 답변 데이터
questions = [
    "포트폴리오 주제가 뭔가요?",
    "모델은 어떤걸 썼나요?",
    "프로젝트 인원은 어떻게 되나요?",
    "프로젝트 기간은 어떻게 되나요?",
    "조장이 누구인가요?",
    "데이터의 출처는 어디인가요?",
    "어떤 데이터를 이용하였나요?",
    "프로젝트 하는데 어려움은 없었나요?"
]


answers = [
    "위/대장 내시경 실시간 병변 탐지 모델 개발입니다.",
    "YOLOv8m-seg 모델을 사용하였습니다.",
    "유동현,박두연,조혜정 입니다.",
    "총 3주입니다. 기획과 구현, 발표준비 등으로 구성했습니다.",
    "유동현 입니다.",
    "AI Hub에서 데이터를 가져왔습니다.",
    "위/대장 내시경 합성 이미지 데이터를 사용하였습니다.",
    "모델링 과정에서 대용량을 사용하여 구현하는데 어려움이 있었지만 어떻게 잘 해결했습니다."
]

# 질문 임베딩과 답변 데이터프레임 생성
question_embeddings = encoder.encode(questions)
df = pd.DataFrame({'question': questions, '챗봇': answers, 'embedding': list(question_embeddings)})

# 대화 이력을 저장하기 위한 Streamlit 상태 설정
if 'history' not in st.session_state:
    st.session_state.history = []

# 챗봇 함수 정의
def get_response(user_input):
    # 사용자 입력 임베딩
    embedding = encoder.encode(user_input)
    
    # 유사도 계산하여 가장 유사한 응답 찾기
    df['distance'] = df['embedding'].map(lambda x: cosine_similarity([embedding], [x]).squeeze())
    answer = df.loc[df['distance'].idxmax()]

    # 대화 이력에 추가
    st.session_state.history.append({"user": user_input, "bot": answer['챗봇']})

# Streamlit 인터페이스
st.title("위/대장 내시경 병변 탐지 모델")

# 이미지 표시
st.image("내시경썸네일.png", caption="Welcome to the Portfolio Chatbot", use_column_width=True)

st.write("포트폴리오에 관한 질문을 입력해보세요. 예: 주제가 무엇인가요?")

user_input = st.text_input("user", "")

if st.button("Submit"):
    if user_input:
        get_response(user_input)
        user_input = ""  # 입력 초기화

# 대화 이력 표시
for message in st.session_state.history:
    st.write(f"**사용자**: {message['user']}")
    st.write(f"**챗봇**: {message['bot']}")



