import dash
from dash import dcc, html, Input, Output, State, ctx, MATCH, ALL
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import os
from datetime import datetime
from scipy.spatial import ConvexHull
import google.generativeai as genai
from dotenv import load_dotenv

# ----------------------------------------------------------------------------------
# 0. 설정 및 보안 (API Key)
# ----------------------------------------------------------------------------------
pd.set_option('future.no_silent_downcasting', True)

# [설정] api.env 파일 로드
env_path = os.path.join(os.getcwd(), 'api.env')
load_dotenv(env_path)

GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")

# Gemini 모델 초기화
model = None
if not GOOGLE_API_KEY:
    print(f"⚠️ 경고: '{env_path}' 에서 GEMINI_API_KEY를 찾을 수 없습니다. 챗봇이 작동하지 않습니다.")
else:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        # [최종] 속도와 안정성을 위해 'gemini-2.0-flash' 사용
        model = genai.GenerativeModel('gemini-2.0-flash')
        print("✅ Gemini API 연결 성공 (Model: gemini-2.0-flash)")
    except Exception as e:
        print(f"⚠️ API 설정 오류: {e}")

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.LUMEN, "https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.5/font/bootstrap-icons.css"],
    meta_tags=[{'name': 'viewport', 'content': 'width=device-width, initial-scale=1'}],
    suppress_callback_exceptions=True
)
server = app.server

# ----------------------------------------------------------------------------------
# 1. 데이터 정의 (Full Data)
# ----------------------------------------------------------------------------------
TEAM_COLORS = {
    '강원': ['#DD5828', '#006058', '#FDB813', '#FFF3E0'],
    '광주': ['#5F0E0D', '#F5BC00', '#1D2F5E', '#FFFDE7'],
    '김천': ['#B81C22', '#002649', '#C9A96F', '#FFEBEE'],
    '대구': ['#0A50A1', '#99CEE3', '#FF7F00', '#E3F2FD'],
    '대전': ['#8E253F', '#007A6C', '#D3D3D3', '#E8F5E9'],
    '서울': ['#000000', '#EB3A2D', '#C29330', '#F5F5F5'],
    '수원': ['#00396F', '#EB0028', '#FFC627', '#E8EAF6'],
    '울산': ['#102FDD', '#FFCC00', '#D3D3D3', '#E0F7FA'],
    '인천': ['#276FB8', '#000000', '#FFD700', '#E3F2FD'],
    '전북': ['#00523D', '#FFD200', '#224F85', '#E8F5E9'],
    '제주': ['#F57F25', '#D72631', '#0A1E3A', '#FFF3E0'],
    '포항': ['#EF4641', '#000000', '#D3D3D3', '#FFEBEE'],
    'Default': ['#2c3e50', '#95a5a6', '#ecf0f1', '#D3D3D3']
}

INJURY_TEAM_MAP = {
    'Gangwon FC': '강원', 'Gwangju FC': '광주', 'Gimcheon Sangmu': '김천',
    'Daegu FC': '대구', 'Daejeon Hana Citizen': '대전', 'FC Seoul': '서울',
    'Suwon FC': '수원', 'Ulsan HD FC': '울산', 'Ulsan Hyundai': '울산',
    'Incheon United': '인천', 'Jeonbuk Hyundai Motors': '전북',
    'Jeju United': '제주', 'Pohang Steelers': '포항'
}

MANAGER_HISTORY = {
    '전북': [{'end': '2024-04-06', 'name': '단 페트레스쿠'}, {'start': '2024-04-07', 'end': '2024-05-26', 'name': '박원재(대행)'}, {'start': '2024-05-27', 'name': '김두현'}],
    '대구': [{'end': '2024-04-19', 'name': '최원권'}, {'start': '2024-04-20', 'end': '2024-04-22', 'name': '정선호(대행)'}, {'start': '2024-04-23', 'name': '박창현'}],
    '대전': [{'end': '2024-05-21', 'name': '이민성'}, {'start': '2024-05-22', 'end': '2024-06-02', 'name': '정광석(대행)'}, {'start': '2024-06-03', 'name': '황선홍'}],
    '인천': [{'end': '2024-07-05', 'name': '조성환'}, {'start': '2024-07-06', 'end': '2024-07-31', 'name': '변재섭(대행)'}, {'start': '2024-08-01', 'name': '최영근'}],
    '울산': [{'end': '2024-07-10', 'name': '홍명보'}, {'start': '2024-07-11', 'end': '2024-07-27', 'name': '이경수(대행)'}, {'start': '2024-07-28', 'name': '김판곤'}]
}

TEAM_INFO = {
    '강원': {'founded': '2008', 'stadium': '강릉/춘천', 'manager': '윤정환', 'slogan': 'Great Union', 'captains': [{'role': 'Captain', 'name': '윤석영'}, {'role': 'Vice-Captain', 'name': '김영빈'}], 'legends': '이을용, 김영후', 'trophies': [], 'records': [['최다 이적료 영입', '가브리엘 - 120만 달러 (약 15억 원)'], ['최다 이적료 방출', '양민혁 - 400만 유로 (약 60억 원)'], ['최다 출장', '김오규 - 222경기'], ['최다 득점', '김영후 - 39득점'], ['최다 도움', '김대원 - 26도움'], ['최다 공격P', '김영후 - 59개 (33득점 26도움)'], ['최다 무실점', '이광연 - 17경기'], ['최연소 출장', '김형진 : 17세 7개월 18일'], ['최연소 득점', '양민혁 : 17세 10개월 23일'], ['최고령 출장', '박호진 : 36세 10개월 2일'], ['최고령 득점', '이을용 : 35세 9개월 24일'] ]},
    '광주': {'founded': '2010', 'stadium': '광주축구전용구장', 'manager': '이정효', 'slogan': 'Yellow Spirit', 'captains': [{'role': 'Captain', 'name': '안영규'}, {'role': 'Vice-Captain', 'name': '이민기'}], 'legends': '펠리페, 여름', 'trophies': [{'name': 'K리그2', 'count': '2회 (2019, 2022)'}], 'records': [['최다 이적료 영입', '아사니 (약 9억 원)'], ['최다 이적료 방출', '펠리페 (약 20억 원)'], ['최다 출장', '안영규 - 190경기'], ['최다 득점', '펠리페 - 41골'], ['최다 도움', '이으뜸 - 34도움'], ['최다 공격P', '-'], ['최다 무실점', '-'], ['최연소 출장', '김윤호: 17세 4개월 17일'], ['최연소 득점', '엄지성: 18세 10개월 26일'], ['최고령 출장', '권정혁: 36세 11개월 9일'], ['최고령 득점', '김효기: 34세 12일'] ]},
    '김천': {'founded': '1984(상무)', 'stadium': '김천종합운동장', 'manager': '정정용', 'slogan': 'Happy Kimcheon', 'captains': [{'role': 'Captain', 'name': '김민덕'}, {'role': 'Vice-Captain', 'name': '김진규'}], 'legends': '-', 'trophies': [{'name': 'K리그2', 'count': '2회 (2021, 2023)'}], 'records': [['최다 이적료 영입', '-'], ['최다 이적료 방출', '-'], ['최다 출장', '-'], ['최다 득점', '-'], ['최다 도움', '-'], ['최다 공격P', '-'], ['최다 무실점', '-'], ['최연소 출장', '-'], ['최연소 득점', '-'], ['최고령 출장', '-'], ['최고령 득점', '-'] ]},
    '대구': {'founded': '2002', 'stadium': 'DGB대구은행파크', 'manager': '박창현', 'slogan': 'We are Daegu', 'captains': [{'role': 'Captain', 'name': '홍철'}, {'role': 'Vice-Captain', 'name': '장성원'}], 'legends': '세징야, 이근호', 'trophies': [{'name': '코리아컵', 'count': '1회 (2018)'}], 'records': [['최다 이적료 영입', '-'], ['최다 이적료 방출', '오장은 (약 28억원)'], ['최다 출장', '세징야 - 291경기'], ['최다 득점', '세징야 - 115득점'], ['최다 도움', '세징야 - 75도움'], ['최다 공격P', '세징야 - 190개'], ['최다 무실점', '조현우 - 70경기'], ['최연소 출장', '박한빈: 18세 8개월 4일'], ['최연소 득점', '박세진: 19세 1개월 24일'], ['최고령 출장', '데얀: 39세 3개월 5일'], ['최고령 득점', '데얀: 39세 2개월 28일'] ]},
    '대전': {'founded': '1997', 'stadium': '대전월드컵경기장', 'manager': '황선홍', 'slogan': 'Daejeon is U', 'captains': [{'role': 'Captain', 'name': '주세종'}, {'role': 'Vice-Captain', 'name': '이창근'}], 'legends': '최은성, 김은중', 'trophies': [{'name': 'K리그2', 'count': '1회 (2014)'}, {'name': '코리아컵', 'count': '1회 (2001)'}], 'records': [['최다 이적료 영입', '김동준 (약 15억 원)'], ['최다 이적료 방출', '윤도영 (약 35억 원)'], ['최다 출장', '최은성 - 495경기'], ['최다 득점', '김은중 - 50득점'], ['최다 도움', '장철우 - 22도움'], ['최다 공격P', '김은중 - 61개'], ['최다 무실점', '최은성 - 130경기'], ['최연소 출장', '윤도영: 17세 6개월 27일'], ['최연소 득점', '김현오: 17세 7개월 21일'], ['최고령 출장', '최은성: 40세 6개월 25일'], ['최고령 득점', '김은중: 35세 7개월'] ]},
    '서울': {'founded': '1983', 'stadium': '서울월드컵경기장', 'manager': '김기동', 'slogan': 'Seoul, My Soul', 'captains': [{'role': 'Captain', 'name': '기성용'}, {'role': 'Vice-Captain', 'name': '조영욱'}], 'legends': '기성용, 박주영, 아디, 데얀', 'trophies': [{'name': 'K리그1', 'count': '6회'}, {'name': '코리아컵', 'count': '2회'}, {'name': '리그컵', 'count': '2회'}, {'name': '슈퍼컵', 'count': '1회'}, {'name': '전국축구선수권', 'count': '1회'}], 'records': [['최다 이적료 영입', '-'], ['최다 이적료 방출', '-'], ['최다 출장', '고요한 - 446경기'], ['최다 득점', '데얀 - 184득점'], ['최다 도움', '몰리나 - 67도움'], ['최다 공격P', '데얀 - 230개'], ['최다 무실점', '김용대 - 71경기'], ['최연소 출장', '한동원 : 16세 25일'], ['최연소 득점', '강성진 : 18세 7개월 8일'], ['최고령 출장', '신의손 : 44세 7개월 9일'], ['최고령 득점', '아디 : 37세 2개월 22일'] ]},
    '수원': {'founded': '2003', 'stadium': '수원종합운동장', 'manager': '김은중', 'slogan': 'Suwon FC', 'captains': [{'role': 'Captain', 'name': '이용'}, {'role': 'Vice-Captain', 'name': '윤빛가람'}], 'legends': '박배종', 'trophies': [{'name': '내셔널리그', 'count': '1회 (2010)'}, {'name': '내셔널선수권', 'count': '3회'}, {'name': '대통령배', 'count': '2회'}], 'records': [['최다 이적료 영입', '-'], ['최다 이적료 방출', '-'], ['최다 출장', '박배종 - 178경기'], ['최다 득점', '라스 - 40득점'], ['최다 도움', '권용현, 라스 - 21도움'], ['최다 공격P', '라스 - 61개'], ['최다 무실점', '박배종 - 40경기'], ['최연소 출장', '안치우 : 17세 9개월 13일'], ['최연소 득점', '하정우 : 18세 9개월 17일'], ['최고령 출장', '이용 : 37세 11개월 9일'], ['최고령 득점', '이용 : 37세 3개월 27일'] ]},
    '울산': {'founded': '1983', 'stadium': '울산문수축구경기장', 'manager': '김판곤', 'slogan': 'My Team ULSAN', 'captains': [{'role': 'Captain', 'name': '김기희'}, {'role': 'Vice-Captain', 'name': '주민규'}], 'legends': '이천수, 김병지, 유상철, 김현석', 'trophies': [{'name': 'K리그1', 'count': '5회 (1996, 05, 22-24)'}, {'name': '코리아컵', 'count': '1회 (2017)'}, {'name': '리그컵', 'count': '5회'}, {'name': '슈퍼컵', 'count': '1회 (2006)'}, {'name': 'ACLE', 'count': '2회 (2012, 2020)'}, {'name': 'A3챔피언스컵', 'count': '1회 (2006)'}], 'records': [['최다 이적료 영입', '오장은 (27억 원)'], ['최다 이적료 방출', '이천수 (42억 원)'], ['최다 출장', '김현석 - 400경기'], ['최다 득점', '김현석 - 120득점'], ['최다 도움', '김현석 - 64도움'], ['최다 공격P', '김현석 - 184개'], ['최다 무실점', '김영광 - 76경기'], ['최연소 출장', '정성빈: 17세 9개월'], ['최연소 득점', '이호: 18세 5개월 22일'], ['최고령 출장', '박주영: 39세 9개월 11일'], ['최고령 득점', '박주영: 39세 9개월 11일'] ]},
    '인천': {'founded': '2003', 'stadium': '인천축구전용경기장', 'manager': '최영근', 'slogan': 'United We Stand', 'captains': [{'role': 'Captain', 'name': '이명주'}, {'role': 'Vice-Captain', 'name': '김도혁'}], 'legends': '임중용, 김도훈', 'trophies': [], 'records': [['최다 이적료 영입', '-'], ['최다 이적료 방출', '-'], ['최다 출장', '김도혁 - 293경기'], ['최다 득점', '무고사 - 108득점'], ['최다 도움', '제르소 - 25도움'], ['최다 공격P', '무고사 - 126개'], ['최다 무실점', '김이섭 - 46경기'], ['최연소 출장', '진성욱: 18세 3개월 2일'], ['최연소 득점', '최우진: 19세 3개월 10일'], ['최고령 출장', '김광석: 39세 7개월 19일'], ['최고령 득점', '김광석: 38세 25일'] ]},
    '전북': {'founded': '1994', 'stadium': '전주월드컵경기장', 'manager': '김두현', 'slogan': '전북이여 영원하라', 'captains': [{'role': 'Captain', 'name': '박진섭'}, {'role': 'Vice-Captain', 'name': '김진수'}], 'legends': '이동국, 최강희, 최진철, 최철순', 'trophies': [{'name': 'K리그1', 'count': '10회 (최다)'}, {'name': '코리아컵', 'count': '6회 (최다)'}, {'name': '슈퍼컵', 'count': '1회 (2004)'}, {'name': 'ACLE', 'count': '2회 (2006, 2016)'}], 'records': [['최다 이적료 영입', '송민규 (약 20억 원)'], ['최다 이적료 방출', '로페즈 (약 74억 원)'], ['최다 출장', '최철순 - 500경기'], ['최다 득점', '이동국 - 210득점'], ['최다 도움', '이동국 - 61도움'], ['최다 공격P', '이동국 - 271개'], ['최다 무실점', '권순태 - 103경기'], ['최연소 출장', '한석진: 16세 9개월'], ['최연소 득점', '이현승: 17세 4개월 26일'], ['최고령 출장', '최은성: 43세 3개월 15일'], ['최고령 득점', '이동국: 41세 1개월 15일'] ]},
    '제주': {'founded': '1982', 'stadium': '제주월드컵경기장', 'manager': '김학범', 'slogan': 'I Love Jeju', 'captains': [{'role': 'Captain', 'name': '임채민'}, {'role': 'Vice-Captain', 'name': '김동준'}], 'legends': '구자철, 윤정환', 'trophies': [{'name': 'K리그1', 'count': '1회 (1989)'}, {'name': 'K리그2', 'count': '1회 (2020)'}, {'name': '리그컵', 'count': '3회'}], 'records': [['최다 이적료 영입', '-'], ['최다 이적료 방출', '-'], ['최다 출장', '김기동 - 289경기'], ['최다 득점', '이원식 - 69득점'], ['최다 도움', '윤정환 - 28도움'], ['최다 공격P', '이원식 - 86개'], ['최다 무실점', '김호준 - 57경기'], ['최연소 출장', '차희철: 17세 5개월 19일'], ['최연소 득점', '차희철: 17세 5개월 25일'], ['최고령 출장', '김근배: 37세 3개월 4일'], ['최고령 득점', '정조국: 36세 2개월 8일'] ]},
    '포항': {'founded': '1973', 'stadium': '포항스틸야드', 'manager': '박태하', 'slogan': 'We are Steelers', 'captains': [{'role': 'Captain', 'name': '완델손'}, {'role': 'Vice-Captain', 'name': '허용준'}], 'legends': '황선홍, 김기동, 김광석', 'trophies': [{'name': 'K리그1', 'count': '5회'}, {'name': '코리아컵', 'count': '6회 (최다)'}, {'name': '리그컵', 'count': '2회'}, {'name': 'ACLE', 'count': '3회 (최다)'}, {'name': '실업연맹전', 'count': '5회'}, {'name': '대통령배', 'count': '1회'}, {'name': '홍콩구정컵', 'count': '1회'}], 'records': [['최다 이적료 영입', '지쿠 (약 15억 원)'], ['최다 이적료 방출', '이명주 (약 50억 원)'], ['최다 출장', '김광석 - 462경기'], ['최다 득점', '라데 - 63득점'], ['최다 도움', '황진성 - 63도움'], ['최다 공격P', '황진성 - 119개'], ['최다 무실점', '신화용 - 124경기'], ['최연소 출장', '최문식: 18세 2개월 19일'], ['최연소 득점', '최문식: 18세 7개월 20일'], ['최고령 출장', '김기동: 39세 9개월 18일'], ['최고령 득점', '김기동: 39세 5개월 28일'] ]}
}

TACTICAL_METRICS = {
    '강원': {'style': '밸런스 빌드업', 'desc': '짧은 패스와 전진 패스의 조화', 'top_stat': '재압박 효율 2위'},
    '광주': {'style': '토탈 사커 & 게겐프레싱', 'desc': '압도적인 점유율과 강한 전방 압박', 'top_stat': '경기 주도권 1위 (56%)'},
    '김천': {'style': '기동력 축구', 'desc': '많은 활동량과 빠른 공수 전환', 'top_stat': '숏패스 비중 2위'},
    '대구': {'style': '선수비 후역습 (딸깍)', 'desc': '내려앉은 수비 후 긴 패스로 한방', 'top_stat': '롱패스 비중 1위 (10%)'},
    '대전': {'style': '다이렉트 어택', 'desc': '직선적이고 빠른 측면 돌파', 'top_stat': '공중볼 경합 1위'},
    '서울': {'style': '규율 잡힌 밸런스', 'desc': '김기동식 공간 활용과 실리 축구', 'top_stat': '지능적 수비 수치 높음'},
    '수원': {'style': '실리적 공격 축구', 'desc': '효율적인 공격 전개와 마무리', 'top_stat': '전진 패스 비중 상위권'},
    '울산': {'style': '주도적 지배 (티키타카)', 'desc': '높은 점유율과 짧은 패스 위주 운영', 'top_stat': '압박 강도(PPDA) 1위'},
    '인천': {'style': '질식 수비 (늪 축구)', 'desc': '강한 수비 블록과 거친 압박', 'top_stat': '텐백 지수 상위권'},
    '전북': {'style': '닥공 (닥치고 공격)', 'desc': '높은 라인과 공격적인 운영', 'top_stat': '크로스 의존도 높음'},
    '제주': {'style': '질식 압박 & 활동량', 'desc': '많이 뛰며 상대를 괴롭히는 축구', 'top_stat': '전진 패스 비중 1위'},
    '포항': {'style': '측면 파괴 & 크로스', 'desc': '측면을 넓게 쓰는 직선적인 공격', 'top_stat': '크로스 의존도 2위'}
}

def get_tactical_tooltip(team_name):
    clean_name = clean_team_name(team_name)
    data = TACTICAL_METRICS.get(clean_name, {'style': '정보 없음', 'desc': '-', 'top_stat': '-'})
    
    return [
        html.H6(f"⚽ {data['style']}", style={'fontWeight': 'bold', 'marginBottom': '5px', 'color': '#ffcc00'}),
        html.P(data['desc'], style={'marginBottom': '5px', 'fontSize': '0.9em'}),
        html.Small(f"📊 핵심: {data['top_stat']}", style={'color': '#ddd'})
    ]

MANAGER_SPEECH_PROFILES = {
    "이정효": {
        "sentence_style": "차분하지만 뼈 있는 직설 화법, 확신에 찬 단정적 어조",
        "perspective": "결과보다는 훈련 과정과 선수의 성장, 한계를 깨는 도전 중심",
        "frequent_phrases": ["성장해야 합니다", "과정이 중요합니다", "버티십시오", "책임감보다는 사명감", "실패를 두려워하지 마라", "우리는 하나다"],
        "avoid": ["부담감(느낄 새도 없다)", "방어적인 태도", "적당히", "핑계", "안주하는 모습"]
    },
    "윤정환": {
        "sentence_style": "차분하고 겸손하며, 신중하게 단어를 고르는 정중한 어조 (말끝을 흐리거나 '음...'하며 생각을 정리함)",
        "perspective": "나보다는 선수와 스태프의 헌신, 그리고 '팀 분위기'와 '믿음'을 최우선으로 여기는 덕장 스타일",
        "frequent_phrases": ["선수들에게 공을 돌리고 싶다", "보이지 않는 곳에서의 헌신", "팀 분위기가 가장 중요합니다", "최선을 다했습니다"],
        "avoid": ["자극적인 도발", "지나친 설레발", "감독 개인의 성과 강조", "선수 탓"]
    },
    "정정용": {
        "sentence_style": "차분하고 논리적이며, 선생님처럼 설명하는 부드러운 경어체",
        "perspective": "당장의 결과보다 '과정'과 '시스템', 그리고 선수의 '성장'을 중시하는 육성가적 관점",
        "frequent_phrases": ["저는 그렇게 생각합니다", "과정이 중요합니다", "결국은 성장해야 합니다", "버티다 보면 기회가 옵니다", "시스템적으로"],
        "avoid": ["감정적인 화풀이", "무조건적인 결과 지상주의 발언", "선수 탓", "거만한 태도"]
    },
    "박창현": {
        "sentence_style": "솔직하고 소탈한 대화체, 권위적이지 않고 겸손하며 경험을 이야기하듯 편안한 어조 ('~같아요', '뭐' 같은 추임새 사용)",
        "perspective": "나이와 경력을 불문하고 배울 점은 흡수(Copy)하여 내 것으로 만드는 유연함과 끊임없는 실험 정신",
        "frequent_phrases": ["좋은 건 미안하지만 갖다 씁니다", "후배들에게도 배웁니다", "실험을 많이 해봤어요", "나 같은 사람도 해냈습니다"],
        "avoid": ["권위적인 태도", "고정관념", "변화를 거부하는 고집", "체면 치레"]
    },
    "황선홍": {
        "sentence_style": "차분하고 신중하며, 묵직하고 진중한 어조 (문장 호흡이 다소 길고 '어...', '음...' 하며 생각을 고르는 편)",
        "perspective": "팀의 안정화와 절실함, 그리고 실패를 딛고 일어서는 도전 정신 중심",
        "frequent_phrases": ["절실한 마음으로", "책임감을 가지고", "차근차근 만들어 나가겠습니다", "운동장에서 증명하겠습니다", "팬들의 기대에 부응하도록"],
        "avoid": ["가벼운 농담", "즉각적인 성과 장담", "책임 회피", "감정적인 대응"]
    },
    "김기동": {
        "sentence_style": "자신감 넘치고 논리적인 어조 (질문에 바로 답하기보다 반문하거나 구체적 근거를 들어 설명하며, '어...', '음...' 보다는 명확하게 끊어 말함)",
        "perspective": "철저한 분석과 데이터 기반의 준비, 그리고 결과에 대한 확실한 책임감 ('나는 마술사가 아니라 준비하는 사람이다')",
        "frequent_phrases": ["제 자신을 믿습니다", "운동장에서 증명하겠습니다", "준비한 만큼 나옵니다", "핑계 대고 싶지 않습니다", "결국 결과로 보여줘야죠"],
        "avoid": ["근거 없는 낙관", "모호한 답변", "약한 모습", "지루한 설명"]
    },
    "김은중": {
        "sentence_style": "차분하고 겸손하며, 선생님처럼 자상하지만 단호한 어조 ('우리 선수들'을 주어로 자주 사용하며 공을 선수들에게 돌림)",
        "perspective": "선수의 성장과 잠재력 발견, 그리고 '원팀'으로서의 헌신과 희생 강조 ('스타는 없지만 팀은 있다')",
        "frequent_phrases": ["우리 선수들이 대견하고", "끝이 아닌 시작입니다", "자만하지 않고", "묵묵히 준비했습니다", "한국 축구의 미래를 위해"],
        "avoid": ["특정 선수 편애/비난", "감정적인 흥분", "나(감독)를 내세우기", "자극적인 언행"]
    },
    "김판곤": {
        "sentence_style": "논리적이고 설득력 있는 어조. '인포메이션', '서비스', '플랜' 등 영어 단어를 자연스럽게 섞어 쓰며, 체계적인 시스템을 강조함.",
        "perspective": "감독은 선수에게 '서비스'를 제공하는 사람이라는 조력자 마인드, 그리고 '주도적이고 능동적인' 축구 철학 중심.",
        "frequent_phrases": ["주도적이고 능동적인 축구를", "여러분들에게 가장 좋은 인포메이션을", "자율 속의 책임", "우리가 지배하는 경기", "나의 약함을 자랑할 수 있어야"],
        "avoid": ["무논리적인 호통", "단순한 정신력 강조", "감정적인 비난", "비체계적인 지시"]
    },
    "홍명보": {
        "sentence_style": "차분하고 분석적이며 다소 건조한 인터뷰 톤. '~라는 생각이 듭니다', '~라고 생각하고 있습니다'로 문장을 맺으며 '전체적인', '측면에서' 같은 연결어를 자주 사용.",
        "perspective": "팀 전체의 밸런스와 선수 개개인의 '역할' 수행, 그리고 월드컵 등 큰 목표를 위한 '과정'과 '조합'을 중시.",
        "frequent_phrases": ["전체적인 측면에서", "선수들이 각자의 역할을", "어떤 식으로든 결과를 가져오는", "조합을 맞춰가는 과정", "경기력을 유지하고"],
        "avoid": ["특정 선수 공개 비난", "즉흥적이고 감정적인 발언", "지나친 설레발", "구체적이지 않은 변명"]
    },
    "단 페트레스쿠": {
        "sentence_style": "직설적이고 단순 명료함, 열정적이고 에너지가 넘치는 단문 위주",
        "perspective": "무조건적인 결과와 승리, 공격적이고 빠른 전진(Direct) 중심",
        "frequent_phrases": ["결과가 가장 중요하다", "우리는 전북이다", "싸워야 한다", "변명은 필요 없다", "공격 앞으로(Go Forward)"],
        "avoid": ["패배를 인정하는 태도", "복잡하고 모호한 전술 설명", "점유율만을 위한 축구(가로 패스 싫어함)"]
    },
    "박태하": {
        "sentence_style": "차분하고 논리적이며 신사적인 경어체, 분석적인 어조",
        "perspective": "현대 축구 트렌드와 데이터 중시, 선수의 장점을 극대화하는 실리 추구",
        "frequent_phrases": ["선수들의 장점을 최대한", "최선을 다해 준비하겠습니다", "성공과 실패는 50대 50", "물러서고 싶지 않습니다"],
        "avoid": ["감정적인 흥분", "근거 없는 호언장담", "선수 탓"]
    },
    # 데이터에 없는 감독을 위한 기본값
    "Default": {
        "sentence_style": "정중하지만 단호한 전문가의 어조",
        "perspective": "데이터와 사실에 기반한 분석",
        "frequent_phrases": ["선수들을 믿습니다", "최선을 다했습니다", "다음 경기를 위해"],
        "avoid": ["책임 회피", "비속어", "AI스러운 기계적 답변"]
    }
}

FIXED_FORMATIONS = {
    '4-3-3': [(5, 34), (25, 10), (25, 26), (25, 42), (25, 58), (50, 15), (50, 34), (50, 53), (75, 15), (75, 34), (75, 53)],
    '4-4-2': [(5, 34), (25, 10), (25, 26), (25, 42), (25, 58), (50, 10), (50, 26), (50, 42), (50, 58), (75, 26), (75, 42)],
    '4-2-3-1': [(5, 34), (25, 10), (25, 26), (25, 42), (25, 58), (45, 26), (45, 42), (70, 10), (70, 34), (70, 58), (90, 34)],
    '3-4-3': [(5, 34), (25, 17), (25, 34), (25, 51), (50, 10), (50, 26), (50, 42), (50, 58), (75, 15), (75, 34), (75, 53)],
    '3-5-2': [(5, 34), (25, 17), (25, 34), (25, 51), (50, 10), (50, 22), (50, 34), (50, 46), (50, 58), (75, 26), (75, 42)],
    'Default': [(5, 34), (25, 10), (25, 26), (25, 42), (25, 58), (50, 15), (50, 34), (50, 53), (75, 15), (75, 34), (75, 53)]
}

def clean_team_name(name):
    if not name: return "Default"
    remove_list = ["유나이티드", "모터스", "스틸러스", "시티즌", "현대", "하나", "FC", "HD", " ", "상무"]
    clean = name
    for word in remove_list: clean = clean.replace(word, "")
    if '김천' in clean: return '김천'
    if '제주' in clean: return '제주'
    return clean

def get_team_colors(team_name):
    simple_name = clean_team_name(team_name)
    for key in TEAM_COLORS:
        if key == simple_name: return TEAM_COLORS[key]
    for key in TEAM_COLORS:
        if key in simple_name: return TEAM_COLORS[key]
    return TEAM_COLORS['Default']

def get_manager_for_date(team_name, match_date_str=None):
    simple_name = clean_team_name(team_name)
    current_manager = TEAM_INFO.get(simple_name, {}).get('manager', '-')
    if not match_date_str or simple_name not in MANAGER_HISTORY: return current_manager
    try:
        match_date = pd.to_datetime(str(match_date_str)).date()
        for period in MANAGER_HISTORY[simple_name]:
            start = datetime.strptime(period['start'], '%Y-%m-%d').date() if 'start' in period else None
            end = datetime.strptime(period['end'], '%Y-%m-%d').date() if 'end' in period else None
            if end:
                if start:
                    if start <= match_date <= end: return period['name']
                else: 
                    if match_date <= end: return period['name']
            elif start: 
                if match_date >= start: return period['name']
        return current_manager
    except: return current_manager

def get_team_metadata(team_name, match_date=None):
    simple_name = clean_team_name(team_name)
    info = TEAM_INFO.get(simple_name, {'founded': '-', 'stadium': '-', 'manager': '-', 'slogan': '-', 'captains': [], 'trophies': [], 'legends': '-', 'records': []})
    manager_name = get_manager_for_date(simple_name, match_date)

    def check_path(folder, name):
        base_name = str(name).replace("(대행)", "").strip()
        candidates = [f"{base_name}.jpg", f"{base_name}.png", f"{base_name.replace(' ', '')}.jpg", f"{base_name.replace(' ', '')}.png"]
        if folder == 'logos' or folder == 'logoflags':
            if base_name == '김천': candidates.extend(['김천상무.jpg', '김천상무.png'])
            if base_name == '제주': candidates.extend(['제주유나이티드.jpg', '제주유나이티드.png'])
        for f_name in candidates:
            abs_path = os.path.join(os.getcwd(), 'assets', 'pictures', folder, f_name)
            if os.path.exists(abs_path): return f"/assets/pictures/{folder}/{f_name}"
        return f"/assets/pictures/{folder}/{base_name}.jpg"

    trophies_data = []
    if isinstance(info['trophies'], list):
        for trp in info['trophies']:
            if isinstance(trp, dict):
                img_name = trp['name'].split()[0] 
                trophies_data.append({**trp, 'img': check_path('trophies', img_name)})
    
    return {
        **info, 'simple_name': simple_name, 'manager': manager_name,
        'img_logo': check_path('logos', simple_name),
        'img_kit_h': f"/assets/pictures/kits/{simple_name}H.png",
        'img_kit_a': f"/assets/pictures/kits/{simple_name}A.png",
        'img_flag': check_path('logoflags', simple_name),
        'img_stadium': check_path('stadiums', simple_name),
        'img_manager': check_path('managers', manager_name),
        'img_legend': check_path('legends', simple_name),
        'captains_data': [{**cap, 'img': check_path('players', cap['name'])} for cap in info['captains']],
        'trophies_data': trophies_data
    }

def hex_to_rgba(hex_code, opacity):
    hex_code = hex_code.lstrip('#')
    return f"rgba({int(hex_code[0:2], 16)}, {int(hex_code[2:4], 16)}, {int(hex_code[4:6], 16)}, {opacity})"

def get_contrasting_text_color(hex_color):
    hex_color = hex_color.lstrip('#')
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
    return 'white' if luminance < 0.5 else 'black'

# ----------------------------------------------------------------------------------
# 3. 데이터 로드
# ----------------------------------------------------------------------------------
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, 'data')

def load_data():
    files = {'raw': 'raw_data.csv', 'match': 'match_info.csv', 'stats': '2024_하나은행_K리그1_경기기록.csv', 'injury': 'k_league_2024_integrated.csv'}
    dfs = {}
    if not os.path.exists(DATA_DIR): os.makedirs(DATA_DIR, exist_ok=True)
    for k, v in files.items():
        path = os.path.join(DATA_DIR, v)
        if os.path.exists(path):
            try: dfs[k] = pd.read_csv(path, encoding='utf-8')
            except: 
                try: dfs[k] = pd.read_csv(path, encoding='cp949')
                except: dfs[k] = pd.DataFrame()
        else: dfs[k] = pd.DataFrame()
    return dfs['raw'], dfs['match'], dfs['stats'], dfs['injury']

raw_df, match_df, stats_df, injury_df = load_data()
all_teams = sorted(raw_df['team_name_ko'].unique()) if not raw_df.empty else []

if not match_df.empty and not raw_df.empty:
    match_df['game_id'] = match_df['game_id'].astype(str)
    raw_df['game_id'] = raw_df['game_id'].astype(str)
    if 'game_date' in match_df.columns:
        cols_to_merge = ['game_id', 'game_date', 'game_day', 'home_team_name_ko', 'away_team_name_ko']
        available_cols = [c for c in cols_to_merge if c in match_df.columns]
        info_map = match_df[available_cols].drop_duplicates(subset=['game_id'])
        cols_in_raw = [c for c in available_cols if c in raw_df.columns and c != 'game_id']
        if cols_in_raw: raw_df = raw_df.drop(columns=cols_in_raw)
        raw_df = raw_df.merge(info_map, on='game_id', how='left')

if 'period_id' in raw_df.columns:
    raw_df['period_id'] = raw_df['period_id'].fillna(1).infer_objects(copy=False)

if not stats_df.empty:
    if '라운드' in stats_df.columns:
        stats_df['라운드_숫자'] = stats_df['라운드'].astype(str).str.extract(r'(\d+)').fillna(0).astype(int)
    if '출전시간(분)' in stats_df.columns:
        stats_df['출전시간(분)'] = pd.to_numeric(stats_df['출전시간(분)'], errors='coerce').fillna(0)

def calculate_league_averages(df):
    if df.empty: return {'xG': 0, 'Shots': 0, 'Passes': 0}
    match_stats = df.groupby(['game_id', 'team_name_ko']).agg({'xG': 'sum'}).reset_index()
    shots = df[df['type_name'].isin(['Shot', 'Goal'])].groupby(['game_id', 'team_name_ko']).size().reset_index(name='shots')
    passes = df[df['type_name']=='Pass'].groupby(['game_id', 'team_name_ko']).size().reset_index(name='passes')
    stats = match_stats.merge(shots, on=['game_id','team_name_ko'], how='left').merge(passes, on=['game_id','team_name_ko'], how='left').fillna(0)
    return {'xG': stats['xG'].mean(), 'Shots': stats['shots'].mean(), 'Passes': stats['passes'].mean()}

def preprocess_data(df):
    if df.empty: return df
    df = df.copy()
    for col in ['start_x', 'start_y', 'end_x', 'end_y']:
        if col not in df.columns: df[col] = 50
        
    df['norm_x'] = df['start_x']
    df['norm_y'] = df['start_y']
    df['norm_end_x'] = df['end_x']
    df['norm_end_y'] = df['end_y']
    
    mask_2nd = (df['period_id'] == 2) & (df['time_seconds'] < 2700)
    df.loc[mask_2nd, 'time_seconds'] += 2700
    
    if 'xT' not in df.columns:
        df['xT'] = 0.0
        mask = (df['type_name'] == 'Pass') & (df['norm_end_x'] > df['norm_x'])
        df.loc[mask, 'xT'] = (df.loc[mask, 'norm_end_x'] - df.loc[mask, 'norm_x']) * 0.002
    
    df['dist_to_goal'] = np.sqrt((105 - df['norm_x'])**2 + (34 - df['norm_y'])**2)
    dy1 = 30.34 - df['norm_y']
    dy2 = 37.66 - df['norm_y']
    dx = 105 - df['norm_x']
    angle1 = np.arctan2(dy1, dx)
    angle2 = np.arctan2(dy2, dx)
    df['shot_angle'] = np.abs(angle1 - angle2)
    
    logit = -1.5 - 0.12 * df['dist_to_goal'] + 2.0 * df['shot_angle']
    df['xG'] = np.where(df['type_name'].isin(['Shot', 'Goal']), 1 / (1 + np.exp(-logit)), 0)

    df['angle_rad'] = np.arctan2(df['norm_end_y'] - df['norm_y'], df['norm_end_x'] - df['norm_x'])
    df['angle_deg'] = np.degrees(df['angle_rad'])
    df['angle_bin'] = (np.round(df['angle_deg'] / 45) * 45).fillna(0).astype(int)
    
    return df

raw_df = preprocess_data(raw_df)
LEAGUE_AVG = calculate_league_averages(raw_df)

def get_match_players_info(game_id, team_name):
    if raw_df.empty or stats_df.empty: return {}
    game_row = raw_df[raw_df['game_id'] == str(game_id)]
    if game_row.empty: return {}
    round_val = game_row.iloc[0].get('game_day', 0)
    try: round_num = int(round_val)
    except: round_num = 0
    target_team = clean_team_name(team_name)
    mask = (stats_df['라운드_숫자'] == round_num) & (stats_df['팀명'].str.contains(target_team))
    match_roster = stats_df[mask]
    if match_roster.empty: return {}
    return dict(zip(match_roster['선수명'], match_roster['포지션']))

def infer_formation(roster_info):
    if not roster_info: return '4-3-3'
    pos_counts = {'DF': 0, 'MF': 0, 'FW': 0}
    for pos in roster_info.values():
        if pos in pos_counts: pos_counts[pos] += 1
    
    df_n, mf_n, fw_n = pos_counts['DF'], pos_counts['MF'], pos_counts['FW']
    
    if df_n == 3:
        if mf_n == 5: return '3-5-2'
        if mf_n >= 4: return '3-4-3'
    elif df_n == 4:
        if mf_n == 5 and fw_n == 1: return '4-2-3-1'
        if mf_n == 4: return '4-4-2'
        return '4-3-3'
        
    return '4-3-3'

# [New] Icon Helper Function
def get_absence_icon(reason):
    r = str(reason).lower()
    if any(x in r for x in ['card', 'suspension', 'red']):
        return html.I(className="bi bi-file-fill text-danger me-2", title="Suspension")
    elif any(x in r for x in ['international', 'national', 'selection']):
        return html.I(className="bi bi-airplane-fill text-primary me-2", title="International Duty")
    elif 'fitness' in r:
        return html.I(className="bi bi-activity text-warning me-2", title="Fitness")
    else: # Default to Injury
        return html.I(className="bi bi-plus-square-fill text-danger me-2", title="Injury")

# [Fix] Height Adjustment for Injury Card
def generate_injury_card(team_name, colors):
    if injury_df.empty: return html.Div()
    
    target_eng_teams = [eng for eng, kor in INJURY_TEAM_MAP.items() if kor == clean_team_name(team_name)]
    if not target_eng_teams: return html.Div()
    
    team_injuries = injury_df[injury_df['Team'].isin(target_eng_teams)].copy()
    if team_injuries.empty: return html.Div()
    
    team_injuries['Games_Missed'] = team_injuries['Games_Missed'].fillna(0)
    team_injuries = team_injuries.sort_values(by='Games_Missed', ascending=False)
    
    rows = []
    for _, row in team_injuries.iterrows():
        icon = get_absence_icon(row['Reason'])
        rows.append(html.Tr([
            html.Td(row['Ko_name'], style={'fontWeight': 'bold'}),
            html.Td([icon, row['Reason']], style={'fontSize': '0.9em', 'color': 'gray'}),
            html.Td(f"{int(row['Games_Missed'])}경기", className="text-center")
        ]))
        
    header_style = {
        'background': f'linear-gradient(90deg, {colors[0]}, {colors[1]})',
        'color': get_contrasting_text_color(colors[0]),
        'fontWeight': 'bold',
        'borderBottom': 'none'
    }
    
    return dbc.Card([
        dbc.CardHeader("Major Absences (Season)", style=header_style),
        dbc.CardBody(
            dbc.Table([
                html.Thead(html.Tr([html.Th("선수명"), html.Th("사유"), html.Th("결장")])),
                html.Tbody(rows)
            ], hover=True, striped=True, size='sm'),
            style={'maxHeight': '600px', 'overflowY': 'auto'} # Match height with Best 11
        )
    ], style={'border': 'none', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)', 'marginBottom': '20px', 'backgroundColor': 'white', 'height': '100%'})


def generate_match_header_card(df, team_home, team_away, colors_h, colors_b, meta_h, meta_b, match_date):
    if df.empty: return html.Div()
    
    # Calculate Score
    score_h = df[df['team_name_ko'] == team_home]['result_name'].apply(lambda x: 1 if x == 'Goal' else 0).sum()
    score_a = df[df['team_name_ko'] == team_away]['result_name'].apply(lambda x: 1 if x == 'Goal' else 0).sum()
    
    return dbc.Card([
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Img(src=meta_h['img_logo'], style={'height': '80px', 'marginBottom': '10px'}),
                    html.H4(team_home, className="bold", style={'color': colors_h[0]})
                ], width=3, className="text-center"),
                
                dbc.Col([
                    html.H6(f"{match_date} | K League 1", className="text-muted mb-2"),
                    html.H1(f"{score_h} : {score_a}", className="bold display-4"),
                    html.H6(meta_h['stadium'], className="text-muted")
                ], width=6, className="text-center align-self-center"),
                
                dbc.Col([
                    html.Img(src=meta_b['img_logo'], style={'height': '80px', 'marginBottom': '10px'}),
                    html.H4(team_away, className="bold", style={'color': colors_b[0]})
                ], width=3, className="text-center")
            ])
        ], className="p-2")
    ], style={'border': 'none', 'boxShadow': '0 4px 8px rgba(0,0,0,0.1)', 'marginBottom': '10px', 'borderRadius': '0px'})

# [Fix] Height Match for Injury Card in Summary
def generate_match_injury_card(team_name, match_date, colors, title="Absences"):
    if injury_df.empty or not match_date: return html.Div()
    
    target_eng_teams = [eng for eng, kor in INJURY_TEAM_MAP.items() if kor == clean_team_name(team_name)]
    if not target_eng_teams: return html.Div()
    
    try:
        match_dt = pd.to_datetime(match_date)
        injury_df['Start_DT'] = pd.to_datetime(injury_df['Start_Date'], dayfirst=True, errors='coerce')
        injury_df['End_DT'] = pd.to_datetime(injury_df['End_Date'], dayfirst=True, errors='coerce')
        
        mask = (injury_df['Team'].isin(target_eng_teams)) & \
               (injury_df['Start_DT'] <= match_dt) & \
               ((injury_df['End_DT'] >= match_dt) | (injury_df['End_DT'].isna()))
        
        team_injuries = injury_df[mask].copy()
    except: return html.Div()
    
    if team_injuries.empty:
        content = html.P("-", className="text-muted text-center m-0 small")
    else:
        rows = []
        for _, row in team_injuries.iterrows():
            icon = get_absence_icon(row['Reason'])
            rows.append(html.Tr([
                html.Td(row['Ko_name'], style={'fontWeight': 'bold', 'padding': '2px', 'fontSize': '0.8rem'}),
                html.Td([icon, row['Reason']], style={'fontSize': '0.7rem', 'color': 'gray', 'padding': '2px'})
            ]))
        content = dbc.Table([html.Tbody(rows)], hover=True, striped=True, borderless=True, size='sm', className="m-0")

    header_style = {
        'background': f'linear-gradient(90deg, {colors[0]}, {colors[1]})',
        'color': get_contrasting_text_color(colors[0]),
        'fontWeight': 'bold',
        'borderBottom': 'none',
        'fontSize': '0.8rem',
        'textAlign': 'center',
    }
    
    return dbc.Card([
        dbc.CardHeader(f"{title}", style=header_style, className="py-1"),
        dbc.CardBody(content, style={'height': '100%', 'overflowY': 'auto', 'padding': '5px'})
    ], style={'border': '1px solid #e0e0e0', 'boxShadow': 'none', 'borderRadius': '0px'}, className="h-100")


# ----------------------------------------------------------------------------------
# 4. 시각화 엔진
# ----------------------------------------------------------------------------------
# [수정] Vertical Pitch Helper (for Heatmap, Shot Map)
def create_vertical_pitch_figure(title, colors, line_color="black"):
    fig = go.Figure()
    bg_color = "rgba(0,0,0,0)" # Transparent background for centering
    
    # [Fix] Padded Ranges to prevent clipping (-5 to 73, -5 to 110)
    shapes = [
        dict(type="rect", x0=0, y0=0, x1=68, y1=105, line=dict(color=line_color), layer="below"),
        dict(type="line", x0=0, y0=52.5, x1=68, y1=52.5, line=dict(color=line_color), layer="below"),
        dict(type="circle", x0=34-9.15, y0=52.5-9.15, x1=34+9.15, y1=52.5+9.15, line=dict(color=line_color), layer="below"),
        dict(type="rect", x0=34-20.16, y0=0, x1=34+20.16, y1=16.5, line=dict(color=line_color), layer="below"),
        dict(type="rect", x0=34-9.16, y0=0, x1=34+9.16, y1=5.5, line=dict(color=line_color), layer="below"),
        dict(type="rect", x0=34-20.16, y0=105-16.5, x1=34+20.16, y1=105, line=dict(color=line_color), layer="below"),
        dict(type="rect", x0=34-9.16, y0=105-5.5, x1=34+9.16, y1=105, line=dict(color=line_color), layer="below"),
    ]
    
    fig.update_layout(
        # [Fix] Title handled by Card Header
        xaxis=dict(visible=False, range=[-5, 73], fixedrange=True, constrain='domain'),
        yaxis=dict(visible=False, range=[-5, 110], fixedrange=True, scaleanchor="x", scaleratio=1),
        shapes=shapes, plot_bgcolor=bg_color, paper_bgcolor=bg_color,
        # [Fix] Minimal margins to center graph in card
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        autosize=True
    )
    return fig

def create_pitch_figure(title, colors, line_color="black"):
    fig = go.Figure()
    bg_color = "rgba(0,0,0,0)"
    main_color = colors[0]
    # [Fix] Padded Ranges (-5 to 110, -5 to 73)
    shapes = [
        dict(type="rect", x0=0, y0=0, x1=105, y1=68, line=dict(color=main_color), layer="below"),
        dict(type="line", x0=52.5, y0=0, x1=52.5, y1=68, line=dict(color=main_color), layer="below"),
        dict(type="circle", x0=52.5-9.15, y0=34-9.15, x1=52.5+9.15, y1=34+9.15, line=dict(color=main_color), layer="below"),
        dict(type="rect", x0=0, y0=34-20.16, x1=16.5, y1=34+20.16, line=dict(color=main_color), layer="below"),
        dict(type="rect", x0=105-16.5, y0=34-20.16, x1=105, y1=34+20.16, line=dict(color=main_color), layer="below"),
        dict(type="rect", x0=0, y0=34-9.16, x1=5.5, y1=34+9.16, line=dict(color=main_color), layer="below"),
        dict(type="rect", x0=105-5.5, y0=34-9.16, x1=105, y1=34+9.16, line=dict(color=main_color), layer="below"),
    ]
    fig.update_layout(
        # title removed from here
        xaxis=dict(visible=False, range=[-10, 115], fixedrange=True, constrain='domain'),
        yaxis=dict(visible=False, range=[-10, 78], fixedrange=True, scaleanchor="x", scaleratio=1),
        shapes=shapes, plot_bgcolor=bg_color, paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=10, r=120, t=10, b=10), # Reduced Top Margin
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
        autosize=False, height=450 # [FIX] 높이 고정
    )
    return fig

# [Fix] Height Reduced to 380px for Balance
def fig_match_lineup(game_id, team_name, colors):
    if raw_df.empty or stats_df.empty: return go.Figure()
    
    game_row = raw_df[raw_df['game_id'] == str(game_id)]
    if game_row.empty: return go.Figure()
    
    round_val = game_row.iloc[0].get('game_day', 0)
    try: round_num = int(round_val)
    except: round_num = 0
    
    target_team = clean_team_name(team_name)
    mask = (stats_df['라운드_숫자'] == round_num) & (stats_df['팀명'].str.contains(target_team)) & (stats_df['포지션'] != '대기')
    lineup_df = stats_df[mask]
    
    if lineup_df.empty: return go.Figure()
    
    roster_info = dict(zip(lineup_df['선수명'], lineup_df['포지션']))
    formation = infer_formation(roster_info)
    
    pos_order = {'GK': 0, 'DF': 1, 'MF': 2, 'FW': 3}
    lineup_sorted = lineup_df.sort_values(by='포지션', key=lambda col: col.map(pos_order))
    
    fixed_h_coords = FIXED_FORMATIONS.get(formation, FIXED_FORMATIONS['Default'])
    vertical_coords = [(y, x) for (x, y) in fixed_h_coords]
    
    fig = go.Figure()
    bg_color = '#2E8B57'
    line_color = 'rgba(255, 255, 255, 0.7)'
    
    shapes = [
        dict(type="rect", x0=0, y0=0, x1=68, y1=105, line=dict(color=line_color), layer="below"),
        dict(type="line", x0=0, y0=52.5, x1=68, y1=52.5, line=dict(color=line_color), layer="below"),
        dict(type="circle", x0=34-9.15, y0=52.5-9.15, x1=34+9.15, y1=52.5+9.15, line=dict(color=line_color), layer="below"),
        dict(type="rect", x0=34-20.16, y0=0, x1=34+20.16, y1=16.5, line=dict(color=line_color), layer="below"),
        dict(type="rect", x0=34-9.16, y0=0, x1=34+9.16, y1=5.5, line=dict(color=line_color), layer="below"),
        dict(type="rect", x0=34-20.16, y0=105-16.5, x1=34+20.16, y1=105, line=dict(color=line_color), layer="below"),
        dict(type="rect", x0=34-9.16, y0=105-5.5, x1=34+9.16, y1=105, line=dict(color=line_color), layer="below"),
    ]
    
    for i, (_, row) in enumerate(lineup_sorted.iterrows()):
        if i >= len(vertical_coords): break
        x, y = vertical_coords[i]
        
        fig.add_trace(go.Scatter(
            x=[x], y=[y], mode='markers+text',
            marker=dict(size=22, color=colors[0], line=dict(color='white', width=2)),
            text=[f"<b>{row['선수명']}</b>"],
            textposition="bottom center",
            textfont=dict(color='white', size=13, family="sans-serif"),
            hoverinfo='text',
            hovertext=f"{row['선수명']}<br>{row['포지션']} | No.{row['등번호']}"
        ))

    fig.update_layout(
        xaxis=dict(visible=False, range=[-5, 73], fixedrange=True),
        yaxis=dict(visible=False, range=[-5, 110], fixedrange=True, scaleanchor="x", scaleratio=1),
        shapes=shapes,
        plot_bgcolor=bg_color,
        paper_bgcolor=bg_color,
        margin=dict(l=0, r=0, t=10, b=10), # 여백 제거
        showlegend=False,
        autosize=True,
        uirevision=f"lineup-{game_id}" 
    )
    return fig

# [수정] autosize=False, uirevision 추가, width/height 고정
def fig_best11_vertical(team_name, colors):
    if stats_df.empty: return go.Figure()

    clean_name = clean_team_name(team_name)
    team_stats = stats_df[stats_df['팀명'].str.contains(clean_name, na=False)]
    
    if team_stats.empty: return go.Figure()
    
    pos_minutes = team_stats.groupby(['선수명', '포지션'])['출전시간(분)'].sum().reset_index()
    pos_minutes = pos_minutes.sort_values('출전시간(분)', ascending=False)
    dominant_pos = pos_minutes.drop_duplicates(subset=['선수명'], keep='first')[['선수명', '포지션']]
    
    total_minutes = team_stats.groupby('선수명')['출전시간(분)'].sum().reset_index()
    final_stats = total_minutes.merge(dominant_pos, on='선수명')
    final_stats = final_stats.sort_values('출전시간(분)', ascending=False)
    
    best_gk = final_stats[final_stats['포지션'] == 'GK'].head(1)
    field_players = final_stats[final_stats['포지션'] != 'GK'].head(10)
    
    if best_gk.empty and len(field_players) < 10: return go.Figure() 

    best_11 = pd.concat([best_gk, field_players])
    
    roster_info = dict(zip(best_11['선수명'], best_11['포지션']))
    formation = infer_formation(roster_info)
    
    fixed_h_coords = FIXED_FORMATIONS.get(formation, FIXED_FORMATIONS['Default'])
    vertical_coords = [(y, x) for (x, y) in fixed_h_coords]

    pos_order = {'GK': 0, 'DF': 1, 'MF': 2, 'FW': 3}
    best_11_sorted = best_11.sort_values(by='포지션', key=lambda col: col.map(pos_order))
    
    fig = go.Figure()
    bg_color = '#2E8B57'
    line_color = 'white'
    
    shapes = [
        dict(type="rect", x0=0, y0=0, x1=68, y1=105, line=dict(color=line_color), layer="below"),
        dict(type="line", x0=0, y0=52.5, x1=68, y1=52.5, line=dict(color=line_color), layer="below"),
        dict(type="circle", x0=34-9.15, y0=52.5-9.15, x1=34+9.15, y1=52.5+9.15, line=dict(color=line_color), layer="below"),
        dict(type="rect", x0=34-20.16, y0=0, x1=34+20.16, y1=16.5, line=dict(color=line_color), layer="below"),
        dict(type="rect", x0=34-9.16, y0=0, x1=34+9.16, y1=5.5, line=dict(color=line_color), layer="below"),
        dict(type="rect", x0=34-20.16, y0=105-16.5, x1=34+20.16, y1=105, line=dict(color=line_color), layer="below"),
        dict(type="rect", x0=34-9.16, y0=105-5.5, x1=34+9.16, y1=105, line=dict(color=line_color), layer="below"),
    ]
    
    for i, (_, row) in enumerate(best_11_sorted.iterrows()):
        if i >= len(vertical_coords): break
        x, y = vertical_coords[i]
        
        fig.add_trace(go.Scatter(
            x=[x], y=[y], mode='markers+text',
            marker=dict(size=25, color=colors[0], line=dict(color='white', width=2)),
            text=[f"<b>{row['선수명']}</b>"],
            textposition="bottom center",
            textfont=dict(color='white', size=12, family="Arial Black"),
            hoverinfo='text',
            hovertext=f"{row['선수명']}<br>{row['포지션']} | {int(row['출전시간(분)'])}분"
        ))
    
    fig.update_layout(
        xaxis=dict(visible=False, range=[-5, 73], fixedrange=True),
        yaxis=dict(visible=False, range=[-5, 110], fixedrange=True, scaleanchor="x", scaleratio=1),
        shapes=shapes,
        plot_bgcolor=bg_color,
        paper_bgcolor=bg_color,
        margin=dict(l=0, r=0, t=10, b=10),
        showlegend=False,
        autosize=False, height=600,
        uirevision=f"best11-{team_name}"
    )
    return fig

# --- Graphs ---
def fig_goals_xg_trend(df, team, colors):
    if df.empty: return go.Figure()
    stats = df.groupby('game_id').agg({'xG': 'sum', 'result_name': lambda x: (x == 'Goal').sum(), 'time_seconds': 'max'}).reset_index().sort_values('time_seconds')
    stats['Match'] = range(1, len(stats)+1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=stats['Match'], y=stats['xG'], mode='lines+markers', name='xG', line=dict(color='gray', width=2, dash='dot')))
    fig.add_trace(go.Scatter(x=stats['Match'], y=stats['result_name'], mode='lines+markers', name='Goals', line=dict(color=colors[0], width=4), marker=dict(size=10, color=colors[1])))
    fig.update_layout(
        # title removed
        xaxis=dict(title="Matches", dtick=1), yaxis=dict(title="Count"), margin=dict(l=20,r=20,t=20,b=20), height=300, legend=dict(orientation="v", y=1, x=1.02), autosize=False)
    return fig

def fig_action_zones(df, team, colors):
    if df.empty: return go.Figure()
    fig = go.Figure(go.Histogram2d(x=df['norm_x'], y=df['norm_y'], xbins=dict(start=0, end=105, size=35), ybins=dict(start=0, end=68, size=22.6), colorscale=[[0, '#f8f9fa'], [1, colors[0]]], opacity=0.8, texttemplate="%{z}"))
    fig.update_layout(
        # title removed
        xaxis=dict(visible=False, range=[0, 105]), yaxis=dict(visible=False, range=[0, 68]), margin=dict(l=10,r=10,t=10,b=10), height=300, autosize=False)
    return fig

def fig_attack_direction(df, team, colors):
    if df.empty: return go.Figure()
    final_third = df[df['norm_x'] > 70]
    left = len(final_third[final_third['norm_y'] < 22.6])
    center = len(final_third[(final_third['norm_y'] >= 22.6) & (final_third['norm_y'] <= 45.4)])
    right = len(final_third[final_third['norm_y'] > 45.4])
    total = max(left + center + right, 1)
    percs = [left/total, center/total, right/total]
    labels = ['Left', 'Center', 'Right']
    fig = go.Figure(go.Bar(x=percs, y=labels, orientation='h', marker=dict(color=[colors[1], colors[0], colors[1]]), text=[f"{p:.1%}" for p in percs], textposition='inside'))
    
    # [수정] Margin 0 to fill the card
    fig.update_layout(
        # title 제거 후 card header 사용
        xaxis=dict(visible=False), 
        yaxis=dict(autorange="reversed"), 
        margin=dict(l=10,r=10,t=0,b=10), # Top margin removed
        bargap=0.4, # Thinner bars
        autosize=True,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    return fig

def fig_shot_map(df, team, colors):
    # [수정] Vertical Shot Map (Same logic as lineup)
    fig = create_vertical_pitch_figure(f"{team} Shot Map", colors)
    if df.empty: return fig
    sub = df[(df['team_name_ko'] == team) & (df['type_name'].isin(['Shot', 'Goal']))]
    
    # Transform Coordinates for Vertical (x->y, y->x)
    # Original: x=0-105 (Length), y=0-68 (Width)
    # Vertical: x=0-68 (Width), y=0-105 (Length)
    # So Vertical X = Original Y, Vertical Y = Original X
    
    goals = sub[sub['result_name'] == 'Goal']
    misses = sub[sub['result_name'] != 'Goal']
    
    fig.add_trace(go.Scatter(x=misses['norm_y'], y=misses['norm_x'], mode='markers', marker=dict(size=10, color='gray', opacity=0.6, symbol='x'), name='Miss'))
    fig.add_trace(go.Scatter(x=goals['norm_y'], y=goals['norm_x'], mode='markers', marker=dict(size=15, color=colors[1], symbol='circle', line=dict(width=2, color='black')), name='Goal'))
    
    return fig

def fig_pass_network(df, team, colors):
    fig = create_pitch_figure(f"{team} Pass Network", colors)
    if df.empty: return fig
    sub = df[(df['team_name_ko'] == team) & (df['type_name'] == 'Pass')]
    if sub.empty: return fig
    if df['game_id'].nunique() > 1:
        fig.add_trace(go.Histogram2dContour(x=sub['norm_x'], y=sub['norm_y'], colorscale=[[0, '#f8f9fa'], [1, colors[0]]], contours=dict(coloring='heatmap'), showscale=False, opacity=0.6))
        return fig
    
    avg_loc = sub.groupby('player_name_ko')[['norm_x', 'norm_y']].mean()
    game_id = df['game_id'].iloc[0]
    roster_info = get_match_players_info(game_id, team)
    
    sub['next_player'] = sub['player_name_ko'].shift(-1)
    pass_conn = sub.groupby(['player_name_ko', 'next_player']).size().reset_index(name='count')
    line_rgba = hex_to_rgba(colors[0], 0.4)
    for _, row in pass_conn.iterrows():
        p1, p2 = row['player_name_ko'], row['next_player']
        if p1 in avg_loc.index and p2 in avg_loc.index and row['count'] >= 3:
            x0, y0 = avg_loc.loc[p1]
            x1, y1 = avg_loc.loc[p2]
            fig.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1], mode='lines', line=dict(color=line_rgba, width=min(row['count']*0.3, 5)), showlegend=False))
            
    for p in avg_loc.index:
        x, y = avg_loc.loc[p]
        pos = roster_info.get(p, '')
        symbol = 'square' if pos == '대기' else 'circle'
        fig.add_trace(go.Scatter(x=[x], y=[y], mode='markers+text', 
                                 marker=dict(size=15, color=colors[1], symbol=symbol, line=dict(color='white', width=1)), 
                                 text=[p], textposition="bottom center", textfont=dict(color='black', size=10), name=p, showlegend=False))
    return fig

def fig_momentum(df, team_home, team_away, colors_h, colors_a):
    if df.empty: return go.Figure()
    df = df.copy()
    df['min'] = (df['time_seconds'] // 60).astype(int)
    mom = df.groupby(['min', 'team_name_ko'])['xT'].sum().unstack(fill_value=0)
    if team_home in mom.columns and team_away in mom.columns: mom['diff'] = mom[team_home] - mom[team_away]
    else: mom['diff'] = 0
    mom['smooth'] = mom['diff'].rolling(window=3, min_periods=1).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=mom.index, y=mom['smooth'].clip(lower=0), mode='lines', fill='tozeroy', line=dict(width=0), fillcolor=colors_h[0], name=team_home))
    fig.add_trace(go.Scatter(x=mom.index, y=mom['smooth'].clip(upper=0), mode='lines', fill='tozeroy', line=dict(width=0), fillcolor=colors_a[0], name=team_away))
    fig.update_layout(
        # title removed
        xaxis=dict(title="Minutes"),
        yaxis=dict(visible=False),
        margin=dict(l=20,r=20,t=20,b=20),
        legend=dict(orientation="h", y=1.1),
        autosize=True
    )
    return fig

def fig_team_radar(df, team_home, team_away, colors_h, colors_a):
    def get_stats(t):
        d = df[df['team_name_ko']==t]
        if d.empty: return [0,0,0,0,0]
        return [len(d[d['type_name']=='Pass']), len(d[d['type_name']=='Shot'])*10, len(d[d['type_name']=='Duel']), len(d[(d['type_name']=='Pass') & (d['norm_end_x'] > d['norm_x'])]), d['norm_y'].std()*5]
    categories = ['Pass Volume', 'Attack', 'Physical', 'Directness', 'Width']
    home_vals = get_stats(team_home)
    away_vals = get_stats(team_away)
    max_vals = [max(h, a) if max(h, a) > 0 else 1 for h, a in zip(home_vals, away_vals)]
    home_norm = [(h/m)*100 for h, m in zip(home_vals, max_vals)]
    away_norm = [(a/m)*100 for a, m in zip(away_vals, max_vals)]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=home_norm, theta=categories, fill='toself', name=team_home, line=dict(color=colors_h[0], width=2)))
    fig.add_trace(go.Scatterpolar(r=away_norm, theta=categories, fill='toself', name=team_away, line=dict(color=colors_a[0], width=2)))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], showticklabels=False, ticks=''),
            angularaxis=dict(tickfont=dict(size=10, color='gray'))
        ),
        margin=dict(l=30, r=30, t=10, b=10),
        legend=dict(orientation="h", y=-0.1),
        autosize=True
    )
    return fig

def generate_stats_table(df, team_home, team_away, colors_h, colors_a):
    def get_metrics(t):
        d = df[df['team_name_ko'] == t]
        goals = len(d[d['result_name'] == 'Goal'])
        xg = d['xG'].sum()
        shots = len(d[d['type_name'].isin(['Shot', 'Goal'])])
        ontarget = len(d[d['result_name'].isin(['Goal', 'Saved'])])
        pass_tot = len(d[d['type_name']=='Pass'])
        pass_succ = len(d[(d['type_name']=='Pass') & (d['result_name']=='Successful')])
        pass_acc = int(pass_succ / pass_tot * 100) if pass_tot else 0
        return [goals, f"{xg:.2f}", shots, ontarget, pass_tot, f"{pass_acc}%"]
    team_m = get_metrics(team_home)
    opp_m = get_metrics(team_away)
    rows = []
    metrics = ["Goals", "xG", "Shots", "On Target", "Passes", "Pass Accuracy"]
    for m, t_val, o_val in zip(metrics, team_m, opp_m):
        rows.append(html.Tr([
            html.Td(t_val, className="text-center", style={'fontWeight': 'bold', 'color': colors_h[0]}),
            html.Td(m, className="text-center text-muted"),
            html.Td(o_val, className="text-center", style={'fontWeight': 'bold', 'color': colors_a[0]})
        ]))
    return dbc.Table([html.Thead(html.Tr([html.Th(team_home), html.Th("VS"), html.Th(team_away)])), html.Tbody(rows)], bordered=True, hover=True)

def generate_recent_stats_table(df, team, n_games, colors):
    d = df[df['team_name_ko'] == team]
    games_count = d['game_id'].nunique()
    if games_count == 0: return html.Div("No Data")
    avg_goals = len(d[d['result_name'] == 'Goal']) / games_count
    avg_xg = d['xG'].sum() / games_count
    avg_shots = len(d[d['type_name'].isin(['Shot', 'Goal'])]) / games_count
    avg_passes = len(d[d['type_name']=='Pass']) / games_count
    lg = LEAGUE_AVG
    rows = []
    metrics = ["Goals/Game", "xG/Game", "Shots/Game", "Passes/Game"]
    team_vals = [f"{avg_goals:.2f}", f"{avg_xg:.2f}", f"{avg_shots:.1f}", f"{int(avg_passes)}"]
    league_vals = [f"-", f"{lg['xG']:.2f}", f"{lg['Shots']:.1f}", f"{int(lg['Passes'])}"]
    for m, t_v, l_v in zip(metrics, team_vals, league_vals):
        rows.append(html.Tr([
            html.Td(m, className="text-center text-muted"),
            html.Td(t_v, className="text-center", style={'fontWeight': 'bold', 'color': colors[0]}),
            html.Td(l_v, className="text-center", style={'color': 'gray'})
        ]))
    return dbc.Table([html.Thead(html.Tr([html.Th("Metric"), html.Th(f"{team} (Avg)"), html.Th("League Avg")])), html.Tbody(rows)], bordered=True, hover=True)

def fig_xg_timeline(df, team_home, team_away, colors_h, colors_a):
    if df.empty: return go.Figure()
    def get_cum_xg(t):
        d = df[(df['team_name_ko']==t) & (df['type_name'].isin(['Shot','Goal']))].copy()
        d = d.sort_values('time_seconds')
        d['cum_xg'] = d['xG'].cumsum()
        return pd.concat([pd.DataFrame({'time_seconds':[0], 'cum_xg':[0]}), d])
    h_xg = get_cum_xg(team_home)
    a_xg = get_cum_xg(team_away)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=h_xg['time_seconds']/60, y=h_xg['cum_xg'], mode='lines+markers', name=team_home, line=dict(color=colors_h[0], width=3, shape='hv')))
    fig.add_trace(go.Scatter(x=a_xg['time_seconds']/60, y=a_xg['cum_xg'], mode='lines+markers', name=team_away, line=dict(color=colors_a[0], width=3, shape='hv')))
    fig.update_layout(
        # title removed
        xaxis=dict(title="Minutes"), yaxis=dict(title="xG"), margin=dict(l=20,r=120,t=20,b=20), height=300, legend=dict(orientation="v", y=1, x=1.02), autosize=False)
    return fig

def fig_zone14(df, team, colors):
    sub = df[df['team_name_ko'] == team]
    fig = go.Figure(go.Histogram2d(x=sub['norm_x'], y=sub['norm_y'], xbins=dict(start=0,end=105,size=5.25), ybins=dict(start=0,end=68,size=3.4), colorscale=[[0, '#f8f9fa'], [1, colors[0]]], zsmooth=False, opacity=0.7))
    shapes = [dict(type="rect", x0=0, y0=0, x1=105, y1=68, line=dict(color="black")), dict(type="line", x0=52.5, y0=0, x1=52.5, y1=68, line=dict(color="black")), dict(type="rect", x0=70, y0=22.6, x1=87.5, y1=45.3, line=dict(color="blue", width=3))]
    fig.add_annotation(x=78.75, y=50, text="Zone 14", showarrow=False, font=dict(color="blue", size=12, weight="bold"))
    fig.update_layout(title=f"{team} Zone 14", xaxis=dict(visible=False,range=[-2,107], constrain='domain'), yaxis=dict(visible=False,range=[-2,70], scaleanchor="x", scaleratio=1), shapes=shapes, margin=dict(l=10,r=120,t=30,b=10), height=430, autosize=False)
    return fig

def fig_pass_flow(df, team, colors):
    fig = create_pitch_figure(f"{team} Pass Flow", colors)
    sub = df[(df['team_name_ko'] == team) & (df['type_name'] == 'Pass')]
    if sub.empty: return fig
    prog_passes = sub[sub['norm_end_x'] > sub['norm_x'] + 10]
    arrow_rgba = hex_to_rgba(colors[2], 0.6)
    for _, row in prog_passes.iterrows():
        fig.add_annotation(x=row['norm_end_x'], y=row['norm_end_y'], ax=row['norm_x'], ay=row['norm_y'], xref="x", yref="y", axref="x", ayref="y", showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.5, arrowcolor=arrow_rgba)
    return fig

def fig_pass_sonar(df, team, colors):
    if df['game_id'].nunique() > 1: return create_pitch_figure("Sonar N/A (Multi-Game)", colors)
    fig = create_pitch_figure(f"{team} Pass Sonar", colors)
    sub = df[(df['team_name_ko'] == team) & (df['type_name'] == 'Pass')]
    
    game_id = df['game_id'].iloc[0]
    roster_info = get_match_players_info(game_id, team)
    
    if not roster_info:
        top11 = sub['player_name_ko'].value_counts().head(11).index.tolist()
        formation_key = 'Default'
    else:
        formation_key = infer_formation(roster_info)
        pos_order = {'GK': 0, 'DF': 1, 'MF': 2, 'FW': 3, '대기': 99}
        active_players = sub['player_name_ko'].unique()
        starters = [p for p in active_players if roster_info.get(p, '대기') != '대기']
        starters.sort(key=lambda x: (pos_order.get(roster_info.get(x, 'MF'), 4), x))
        top11 = starters[:11]

    fixed_positions = FIXED_FORMATIONS.get(formation_key, FIXED_FORMATIONS['Default'])
    fill_rgba = hex_to_rgba(colors[2], 0.7)
    
    for i, p in enumerate(top11):
        if i >= len(fixed_positions): break
        cx, cy = fixed_positions[i]
        
        p_data = sub[sub['player_name_ko'] == p]
        sonar = p_data.groupby('angle_bin').size()
        max_val = sonar.max()
        if max_val == 0: continue
        for angle, count in sonar.items():
            r = (count / max_val) * 7
            rad = np.radians(angle)
            x_wedge = [cx, cx + r*np.cos(rad-0.2), cx + r*np.cos(rad+0.2), cx]
            y_wedge = [cy, cy + r*np.sin(rad-0.2), cy + r*np.sin(rad+0.2), cy]
            fig.add_trace(go.Scatter(x=x_wedge, y=y_wedge, fill='toself', mode='lines', line=dict(color=colors[2], width=1), fillcolor=fill_rgba, showlegend=False, hoverinfo='skip'))
        fig.add_trace(go.Scatter(x=[cx], y=[cy-5], mode='text', text=[p], textfont=dict(color='black', size=10, weight='bold'), showlegend=False))
    
    fig.layout.title.text = f"{team} Pass Sonar ({formation_key})"
    return fig

def fig_defensive(df, team, colors):
    fig = create_pitch_figure(f"{team} Defensive Lines", colors)
    sub = df[(df['team_name_ko'] == team) & (df['type_name'].isin(['Recovery', 'Interception', 'Duel']))]
    if not sub.empty:
        avg_x = sub['norm_x'].mean()
        high_x = sub['norm_x'].quantile(0.75)
        low_x = sub['norm_x'].quantile(0.25)
        def add_line(x_val, color, name):
            y_curve = np.linspace(0, 68, 20)
            x_curve = [x_val - (abs(y-34)/20) for y in y_curve]
            fig.add_trace(go.Scatter(x=x_curve, y=y_curve, mode='lines', line=dict(color=color, width=4, dash='dash'), name=name))
        add_line(high_x, colors[2], 'High Press')
        add_line(avg_x, colors[0], 'Avg Line')
        add_line(low_x, colors[1], 'Low Block')
    fig.update_layout(legend=dict(title=dict(text="Lines")))
    return fig

def fig_defensive_actions(df, team, colors):
    fig = create_pitch_figure(f"{team} Def Actions", colors)
    sub = df[(df['team_name_ko'] == team) & (df['type_name'].isin(['Tackle', 'Interception', 'Recovery']))]
    for action in ['Tackle', 'Interception', 'Recovery']:
        act_data = sub[sub['type_name'] == action]
        fig.add_trace(go.Scatter(x=act_data['norm_x'], y=act_data['norm_y'], mode='markers', name=action, marker=dict(size=8)))
    return fig

# ----------------------------------------------------------------------------------
# 6. 레이아웃
# ----------------------------------------------------------------------------------
sidebar_content = html.Div([
    html.H4("P.P.P", style={'fontWeight': 'bold', 'color': 'black'}),
    html.Hr(),
    html.Label("1. Team", style={'fontWeight': 'bold'}),
    dcc.Dropdown(id='team-select', options=[{'label': t, 'value': t} for t in all_teams], value=all_teams[0] if all_teams else None, clearable=False),
    html.Br(),
    html.Label("2. Mode", style={'fontWeight': 'bold'}),
    dcc.Dropdown(id='mode-select', options=[{'label': 'Specific Match', 'value': 'specific'}, {'label': 'Recent Form', 'value': 'recent'}], value='specific', clearable=False),
    html.Div([html.Label("Count", className="mt-2"), dcc.Dropdown(id='recent-count', options=[{'label': i, 'value': i} for i in range(1,11)], value=5)], id='recent-count-container', style={'display': 'none'}),
    html.Div([html.Br(), html.Label("3. Match", style={'fontWeight': 'bold'}), dcc.Dropdown(id='match-select', clearable=False)], id='match-select-container'),
])

offcanvas = dbc.Offcanvas(
    sidebar_content, id="offcanvas", is_open=False, placement="start",
    style={'background': 'rgba(255, 255, 255, 0.9)', 'backdropFilter': 'blur(10px)'}
)

floating_btn = dbc.Button(html.I(className="bi bi-list"), id="open-offcanvas", n_clicks=0, style={'position': 'fixed', 'top': '20px', 'left': '20px', 'zIndex': 1050, 'width': '50px', 'height': '50px', 'borderRadius': '0px', 'backgroundColor': 'transparent', 'color': 'black', 'border': 'none', 'fontSize': '2rem', 'transition': 'opacity 0.3s'})

app.layout = html.Div([
    floating_btn, offcanvas,
    dbc.Container(id="page-content", fluid=True, style={'paddingTop': '80px', 'transition': '0.5s', 'minHeight': '100vh'})
])

# ----------------------------------------------------------------------------------
# 7. 콜백
# ----------------------------------------------------------------------------------
@app.callback(Output("open-offcanvas", "style"), Input("offcanvas", "is_open"))
def toggle_button_visibility(is_open):
    base_style = {'position': 'fixed', 'top': '20px', 'left': '20px', 'zIndex': 1050, 'width': '50px', 'height': '50px', 'borderRadius': '0px', 'backgroundColor': 'transparent', 'color': 'black', 'border': 'none', 'fontSize': '2rem', 'transition': 'opacity 0.3s'}
    if is_open:
        base_style['opacity'] = '0'
        base_style['pointerEvents'] = 'none'
    else:
        base_style['opacity'] = '1'
        base_style['pointerEvents'] = 'auto'
    return base_style

@app.callback(Output("offcanvas", "is_open"), Input("open-offcanvas", "n_clicks"), [State("offcanvas", "is_open")])
def toggle_offcanvas(n1, is_open): return not is_open if n1 else is_open

@app.callback([Output('recent-count-container', 'style'), Output('match-select-container', 'style')], Input('mode-select', 'value'))
def toggle_inputs(mode):
    return ({'display': 'block'}, {'display': 'none'}) if mode == 'recent' else ({'display': 'none'}, {'display': 'block'})

@app.callback([Output('match-select', 'options'), Output('match-select', 'value')], [Input('team-select', 'value'), Input('mode-select', 'value')])
def update_matches(team, mode):
    if not team or mode == 'recent': return [], None
    
    team_games_df = raw_df[raw_df['team_name_ko'] == team][['game_id', 'game_date', 'game_day', 'home_team_name_ko', 'away_team_name_ko']].drop_duplicates()

    try:
        team_games_df['game_day_int'] = team_games_df['game_day'].astype(int)
    except:
        team_games_df['game_day_int'] = 0 
        
    team_games_df = team_games_df.sort_values(by=['game_day_int', 'game_date'], ascending=[True, True])
    
    options = []
    
    for _, row in team_games_df.iterrows():
        gid = row['game_id']
        date_str = str(row['game_date']).split(' ')[0] 
        round_info = f"{row['game_day']}R" if pd.notnull(row['game_day']) else ""
        
        if row['home_team_name_ko'] == team:
            ha_info = "홈"
            opp = row['away_team_name_ko']
        else:
            ha_info = "원정" 
            opp = row['home_team_name_ko']

        label_parts = [part for part in [ha_info, round_info, date_str] if part]
        label = f"vs {opp} ({', '.join(label_parts)})"
        
        options.append({'label': label, 'value': gid})
        
    return options, options[0]['value'] if options else None

@app.callback(
    [Output("page-content", "children"), Output("page-content", "style")],
    [Input('mode-select', 'value'), Input('match-select', 'value'), Input('team-select', 'value'), Input('recent-count', 'value')]
)
def render_page(mode, match_id, team, count):
    if not team: return html.Div(), {}
    
    selected_date = None
    if mode == 'specific' and match_id:
        try:
            date_val = raw_df[raw_df['game_id'] == str(match_id)]['game_date'].iloc[0]
            selected_date = str(date_val)
        except: pass
        
    colors = get_team_colors(team)
    meta = get_team_metadata(team, selected_date)
    
    header_text_color = get_contrasting_text_color(colors[0])
    
    bg_gradient = f'linear-gradient(to bottom, {colors[0]}33, {colors[1]}33, #f8f9fa)'
    page_style = {
        'paddingTop': '80px',
        'transition': '0.5s',
        'backgroundImage': bg_gradient,
        'minHeight': '100vh',
        'backgroundAttachment': 'fixed'
    }

    header_div = html.Div(
        html.H2(f"{team} Analytics Dashboard", className="text-center m-0", style={'color': header_text_color, 'fontWeight': 'bold', 'letterSpacing': '-1px'}),
        style={
            'background': f'linear-gradient(90deg, {colors[0]}, {colors[1]})',
            'padding': '20px', 'borderRadius': '8px', 'marginBottom': '20px', 'boxShadow': '0 4px 6px rgba(0,0,0,0.1)'
        }
    )
    
    # [수정] 탭 구성에 Talk 추가
    tabs = dcc.Tabs(id='analysis-tabs', value='tab-0', children=[
        dcc.Tab(label='📝 Profile', value='tab-0', selected_style={'borderTop': f'4px solid {colors[0]}', 'fontWeight': 'bold', 'color': colors[0]}),
        dcc.Tab(label='🏠 Summary', value='tab-1', selected_style={'borderTop': f'4px solid {colors[0]}', 'fontWeight': 'bold', 'color': colors[0]}),
        dcc.Tab(label='⚔️ Attack', value='tab-2', selected_style={'borderTop': f'4px solid {colors[0]}', 'fontWeight': 'bold', 'color': colors[0]}),
        dcc.Tab(label='⚽ Pass', value='tab-3', selected_style={'borderTop': f'4px solid {colors[0]}', 'fontWeight': 'bold', 'color': colors[0]}),
        dcc.Tab(label='🛡️ Defense', value='tab-4', selected_style={'borderTop': f'4px solid {colors[0]}', 'fontWeight': 'bold', 'color': colors[0]}),
        dcc.Tab(label='💬 Talk', value='tab-chat', selected_style={'borderTop': f'4px solid {colors[0]}', 'fontWeight': 'bold', 'color': colors[0]}),
    ])
    
    return html.Div([header_div, tabs, html.Div(id='tabs-content', style={'paddingTop': '20px'})]), page_style

@app.callback(Output('tabs-content', 'children'), [Input('analysis-tabs', 'value'), State('mode-select', 'value'), State('match-select', 'value'), State('team-select', 'value'), State('recent-count', 'value')])
def render_tab_content(tab, mode, match_id, team, count):
    if not team: return html.Div()
    
    selected_date = None
    if mode == 'specific' and match_id:
        try:
            date_val = raw_df[raw_df['game_id'] == str(match_id)]['game_date'].iloc[0]
            selected_date = str(date_val)
        except: pass

    colors = get_team_colors(team)
    meta = get_team_metadata(team, selected_date)
    header_text_color = get_contrasting_text_color(colors[0])
    
    header_style = {
        'background': f'linear-gradient(90deg, {colors[0]}, {colors[1]})',
        'color': get_contrasting_text_color(colors[0]),
        'fontWeight': 'bold',
        'borderBottom': 'none',
        'padding': '8px 15px', 
        'fontSize': '0.95rem'
    }
    header_style_b = header_style.copy()
    
    card_style = {'border': '1px solid #e0e0e0', 'boxShadow': '0 1px 3px rgba(0,0,0,0.1)', 'marginBottom': '0px', 'backgroundColor': 'white', 'borderRadius': '4px'}
    
    # [Chat Tab]
    if tab == 'tab-chat':
        manager_name = meta.get('manager', '감독')
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(f"💬 Talk with {manager_name}", style=header_style),
                        dbc.CardBody([
                            html.Div(id="chat-history-display", style={'height': '500px', 'overflowY': 'auto', 'padding': '15px', 'backgroundColor': '#f8f9fa', 'border': '1px solid #dee2e6', 'borderRadius': '5px', 'marginBottom': '15px'}),
                            dbc.InputGroup([
                                dbc.Input(id="chat-input", placeholder=f"{manager_name} 감독님께 질문하세요...", type="text"),
                                dbc.Button("전송", id="chat-send-btn", color="primary", n_clicks=0, style={'backgroundColor': colors[0], 'borderColor': colors[0]})
                            ]),
                            dcc.Store(id="chat-store", data=[])
                        ])
                    ], style=card_style, className="h-100")
                ], width={'size': 8, 'offset': 2})
            ], className="mt-4")
        ], fluid=True)

    # [Tab 0: Profile]
    if tab == 'tab-0': 
        cap_data = [c for c in meta['captains_data'] if 'Captain' in c['role'] and 'Vice' not in c['role']]
        vice_data = [c for c in meta['captains_data'] if 'Vice' in c['role']]
        
        def create_captain_col(c):
            return dbc.Col(html.Div([
                        html.Img(src=c['img'], style={'width': '70px', 'height': '70px', 'borderRadius': '50%', 'objectFit': 'cover', 'border': f'2px solid {colors[0]}', 'marginBottom': '5px'}),
                        html.H6(c['name'], className="bold mb-0"),
                        html.Span(c['role'], className="badge rounded-pill mt-1", style={'backgroundColor': colors[0], 'color': 'white', 'fontSize': '0.7em'})
                    ], className="d-flex flex-column align-items-center justify-content-center h-100"), width=6)

        captain_content = []
        if cap_data: captain_content.append(create_captain_col(cap_data[0]))
        if vice_data: captain_content.append(create_captain_col(vice_data[0]))
        
        trp_cards = [dbc.Col(html.Div([html.Img(src=t['img'], style={'height': '110px', 'marginBottom': '5px'}), html.H6(t['name'], className="bold small m-0"), html.Small(t['count'], className="text-muted")], className="text-center p-2"), width=3) for t in meta['trophies_data']]
        rec_rows = [html.Tr([html.Td(r[0], style={'fontWeight': 'bold', 'color': colors[0], 'width': '40%'}), html.Td(r[1])]) for r in meta['records']]
        records_card = dbc.Card([dbc.CardHeader("Club Records", style=header_style), dbc.CardBody(dbc.Table(html.Tbody(rec_rows), bordered=True, hover=True, striped=True, size='sm', className="m-0"))], style=card_style, className="h-100")
        honors_card = dbc.Card([dbc.CardHeader("Honors", style=header_style), dbc.CardBody(dbc.Row(trp_cards))], style=card_style, className="h-100") if trp_cards else html.Div()
        
        best11_card = dbc.Card([
                dbc.CardHeader(f"2024 Season Best 11 - {team}", style=header_style),
                dbc.CardBody(dcc.Graph(figure=fig_best11_vertical(team, colors), config={'responsive': False}, style={'height': '600px'}), className="p-0 d-flex justify-content-center align-items-center")
        ], style=card_style, className="h-100")

        return dbc.Container([
            dbc.Row([
                # Left Column: Team Info
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.Div([
                                html.Img(src=meta['img_logo'], style={'width': '200px', 'marginBottom': '20px'}),
                                html.H2(team, className="text-center bold mb-2"),
                                html.H5(meta['slogan'], className="text-center text-muted italic mb-4")
                            ], className="text-center"),
                            
                            html.Hr(),
                            
                            dbc.Row([
                                dbc.Col(html.Div([
                                    html.Span("Founded", className="text-muted text-uppercase small bold d-block mb-1"),
                                    html.H4(meta['founded'], className="bold")
                                ]), width=6, className="text-center border-end"),
                                dbc.Col(html.Div([
                                    html.Span("Legends", className="text-muted text-uppercase small bold d-block mb-1"),
                                    html.H5(meta['legends'], className="bold", style={'lineHeight': '1.4'})
                                ]), width=6, className="text-center"),
                            ], className="mb-4 align-items-center"),
                            
                            html.Hr(),
                            
                            html.Div([
                                dbc.Row([
                                    dbc.Col(html.Div([
                                        html.Img(src=meta['img_kit_h'], style={'height': '250px', 'objectFit': 'contain'}),
                                        html.H6("Home Kit", className="mt-3 bold text-muted")
                                    ]), className="text-center"),
                                    dbc.Col(html.Div([
                                        html.Img(src=meta['img_kit_a'], style={'height': '250px', 'objectFit': 'contain'}),
                                        html.H6("Away Kit", className="mt-3 bold text-muted")
                                    ]), className="text-center")
                                ], className="d-flex align-items-center flex-grow-1")
                            ], className="d-flex flex-column flex-grow-1 justify-content-center")
                            
                        ], className="d-flex flex-column h-100 p-4")
                    ], style=card_style, className="h-100"),
                ], width=4),
                
                # Right Column
                dbc.Col([
                    dbc.Row([
                        dbc.Col(dbc.Card([
                            dbc.CardHeader("Stadium", style=header_style), 
                            dbc.CardImg(src=meta['img_stadium'], top=True, style={'height': '180px', 'objectFit': 'contain'}), 
                            dbc.CardBody([html.H5(meta['stadium'], className="bold m-0")])
                        ], style=card_style, className="h-100"), width=6),
                        
                        # [Manager Card with Tooltip]
                        dbc.Col(dbc.Card([
                            dbc.CardHeader("Manager (Tactical Style)", style=header_style),
                            dbc.CardImg(
                                src=meta['img_manager'], 
                                top=True, 
                                style={'height': '180px', 'objectFit': 'contain', 'objectPosition': 'top', 'cursor': 'help'}, 
                                id=f"manager-img-{team}"
                            ),
                            dbc.CardBody([html.H5(meta['manager'], className="bold m-0")]),
                            dbc.Tooltip(
                                get_tactical_tooltip(team),
                                target=f"manager-img-{team}",
                                placement="top",
                                style={'fontSize': '0.9rem', 'maxWidth': '300px', 'textAlign': 'left', 'backgroundColor': 'rgba(0,0,0,0.9)'}
                            )
                        ], style=card_style, className="h-100"), width=6),
                    ], className="mb-4 g-3"),
                    
                    dbc.Card([
                        dbc.CardHeader("Captains", style=header_style), 
                        dbc.CardBody(dbc.Row(captain_content, className="align-items-center h-100"), className="p-2")
                    ], style=card_style, className="mb-4"),
                    
                    dbc.Row([
                        dbc.Col(records_card, width=6),
                        dbc.Col(honors_card, width=6)
                    ], className="g-3 mb-0"),

                ], width=8)
            ], className="mb-4"),

            dbc.Row([
                dbc.Col(generate_injury_card(team, colors), width=6),
                dbc.Col(best11_card, width=6)
            ])
        ], fluid=True)

    if mode == 'specific':
        if not match_id: return html.Div()
        df = raw_df[raw_df['game_id'] == match_id]
        teams = df['team_name_ko'].unique()
        if len(teams) < 2: return html.Div("Data incomplete")
        team_a, team_b = team, [t for t in teams if t != team][0]
        colors_b = get_team_colors(team_b)
        
        meta_a = get_team_metadata(team_a)
        meta_b = get_team_metadata(team_b)

        header_style_b = {
            'background': f'linear-gradient(90deg, {colors_b[0]}, {colors_b[1]})',
            'color': get_contrasting_text_color(colors_b[0]),
            'fontWeight': 'bold',
            'borderBottom': 'none',
            'padding': '8px 15px',
            'fontSize': '0.95rem'
        }
        
        lineup_a = fig_match_lineup(match_id, team_a, colors)
        lineup_b = fig_match_lineup(match_id, team_b, colors_b)

        injury_info_a = generate_match_injury_card(team_a, selected_date, colors, title="Absences")
        injury_info_b = generate_match_injury_card(team_b, selected_date, colors_b, title="Absences")

        # [Tab 1: Summary]
        if tab == 'tab-1':
            lineup_height = "550px"
            return dbc.Container([
                dbc.Row([
                    dbc.Col(generate_match_header_card(df, team_a, team_b, colors, colors_b, meta_a, meta_b, selected_date), width=12)
                ], className="mb-4"),

                dbc.Row([
                    dbc.Col(injury_info_a, width=2, style={'height': lineup_height}),
                    dbc.Col(dbc.Card([
                         dbc.CardHeader(f"{team_a}", style=header_style, className="text-center"), 
                         dbc.CardBody(dcc.Graph(figure=lineup_a, config={'displayModeBar': False, 'responsive': True}, style={'height': '100%', 'width': '100%'}), className="p-0 h-100 d-flex justify-content-center align-items-center")
                     ], style=card_style, className="h-100"), width=4, style={'height': lineup_height}),
                     
                     dbc.Col(dbc.Card([
                         dbc.CardHeader(f"{team_b}", style=header_style_b, className="text-center"), 
                         dbc.CardBody(dcc.Graph(figure=lineup_b, config={'displayModeBar': False, 'responsive': True}, style={'height': '100%', 'width': '100%'}), className="p-0 h-100 d-flex justify-content-center align-items-center")
                     ], style=card_style, className="h-100"), width=4, style={'height': lineup_height}),
                    dbc.Col(injury_info_b, width=2, style={'height': lineup_height})
                ], className="g-3 mb-4 align-items-stretch"),

                dbc.Row([
                    dbc.Col(dbc.Card([dbc.CardHeader("Match Stats", style=header_style), dbc.CardBody(generate_stats_table(df, team_a, team_b, colors, colors_b), className="p-0")], style=card_style, className="h-100"), width=6),
                    dbc.Col(dbc.Card([
                        dbc.CardHeader("Style Comparison", style=header_style), 
                        dbc.CardBody(dcc.Graph(figure=fig_team_radar(df, team_a, team_b, colors, colors_b), config={'displayModeBar': False, 'responsive': True}, style={'height': '300px'}), className="p-0 d-flex justify-content-center align-items-center")
                    ], style=card_style, className="h-100"), width=6),
                ], className="g-3 mb-4"),

                dbc.Row([
                    dbc.Col(dbc.Card([dbc.CardHeader("Game Momentum", style=header_style), dbc.CardBody(dcc.Graph(figure=fig_momentum(df, team_a, team_b, colors, colors_b), config={'displayModeBar': False, 'responsive': True}, style={'height': '250px'}), className="p-0")], style=card_style, className="h-100"), width=12),
                ], className="g-3 mb-4")
            ], fluid=True, className="px-3 py-2")
            
        # [Tab 2: Attack] - Completely Separated Cards
        elif tab == 'tab-2':
            return dbc.Container([
                dbc.Row([
                    dbc.Col(dbc.Card([dbc.CardHeader("xG Timeline", style=header_style), dbc.CardBody(dcc.Graph(figure=fig_xg_timeline(df, team_a, team_b, colors, colors_b), config={'responsive': False}, style={'height': '300px', 'width': '100%'}))], style=card_style), width=12),
                ], className="mb-4"),
                
                 # 1. Shot Maps (Row)
                 dbc.Row([
                    dbc.Col(dbc.Card([dbc.CardHeader(f"{team_a} Shot Map", style=header_style), dbc.CardBody(dcc.Graph(figure=fig_shot_map(df, team_a, colors), config={'responsive': False}, style={'height': '600px', 'width': '100%'}))], style=card_style), width=6),
                    dbc.Col(dbc.Card([dbc.CardHeader(f"{team_b} Shot Map", style=header_style_b), dbc.CardBody(dcc.Graph(figure=fig_shot_map(df, team_b, colors_b), config={'responsive': False}, style={'height': '600px', 'width': '100%'}))], style=card_style), width=6),
                ], className="g-4 mb-5"),
                 
                 # 2. Heatmaps (Row)
                 dbc.Row([
                    dbc.Col(dbc.Card([dbc.CardHeader(f"{team_a} Heatmap", style=header_style), dbc.CardBody(dcc.Graph(figure=create_vertical_pitch_figure(f"{team_a} Heatmap", colors).add_trace(go.Histogram2dContour(x=df[df['team_name_ko']==team_a]['norm_y'], y=df[df['team_name_ko']==team_a]['norm_x'], colorscale=[[0, 'rgba(255,255,255,0)'], [1, colors[0]]], showscale=False)), config={'responsive': False}, style={'height': '500px'}), className="p-0 d-flex justify-content-center align-items-center")], style=card_style), width=6),
                    dbc.Col(dbc.Card([dbc.CardHeader(f"{team_b} Heatmap", style=header_style_b), dbc.CardBody(dcc.Graph(figure=create_vertical_pitch_figure(f"{team_b} Heatmap", colors_b).add_trace(go.Histogram2dContour(x=df[df['team_name_ko']==team_b]['norm_y'], y=df[df['team_name_ko']==team_b]['norm_x'], colorscale=[[0, 'rgba(255,255,255,0)'], [1, colors_b[0]]], showscale=False)), config={'responsive': False}, style={'height': '500px'}), className="p-0 d-flex justify-content-center align-items-center")], style=card_style), width=6),
                ], className="g-4 mb-5"),

                # 3. Attack Direction (Row)
                dbc.Row([
                    dbc.Col(dbc.Card([dbc.CardHeader(f"{team_a} Attack Direction", style=header_style), dbc.CardBody(dcc.Graph(figure=fig_attack_direction(df, team_a, colors), config={'responsive': False}, style={'height': '300px', 'width': '100%'}), className="p-0 d-flex justify-content-center align-items-center")], style=card_style), width=6),
                    dbc.Col(dbc.Card([dbc.CardHeader(f"{team_b} Attack Direction", style=header_style_b), dbc.CardBody(dcc.Graph(figure=fig_attack_direction(df, team_b, colors_b), config={'responsive': False}, style={'height': '300px', 'width': '100%'}), className="p-0 d-flex justify-content-center align-items-center")], style=card_style), width=6),
                ], className="g-4 mb-5")
            ], fluid=True)

        # [Tab 3: Pass]
        elif tab == 'tab-3':
            return dbc.Container([
                dbc.Row([
                    dbc.Col(dbc.Card([dbc.CardHeader(f"{team_a} Pass Network", style=header_style), dbc.CardBody(dcc.Graph(figure=fig_pass_network(df, team_a, colors), config={'responsive': False}, style={'height': '450px', 'width': '100%'}), className="p-0 d-flex justify-content-center align-items-center")], style=card_style), width=6),
                    dbc.Col(dbc.Card([dbc.CardHeader(f"{team_b} Pass Network", style=header_style_b), dbc.CardBody(dcc.Graph(figure=fig_pass_network(df, team_b, colors_b), config={'responsive': False}, style={'height': '450px', 'width': '100%'}), className="p-0 d-flex justify-content-center align-items-center")], style=card_style), width=6),
                ], className="g-4 mb-5"),
                dbc.Row([
                    dbc.Col(dbc.Card([dbc.CardHeader(f"{team_a} Pass Flow", style=header_style), dbc.CardBody(dcc.Graph(figure=fig_pass_flow(df, team_a, colors), config={'responsive': False}, style={'height': '450px', 'width': '100%'}), className="p-0 d-flex justify-content-center align-items-center")], style=card_style), width=6),
                    dbc.Col(dbc.Card([dbc.CardHeader(f"{team_b} Pass Flow", style=header_style_b), dbc.CardBody(dcc.Graph(figure=fig_pass_flow(df, team_b, colors_b), config={'responsive': False}, style={'height': '450px', 'width': '100%'}), className="p-0 d-flex justify-content-center align-items-center")], style=card_style), width=6),
                ], className="g-4 mb-5"),
                 dbc.Row([
                    dbc.Col(dbc.Card([dbc.CardHeader(f"{team_a} Pass Sonar", style=header_style), dbc.CardBody(dcc.Graph(figure=fig_pass_sonar(df, team_a, colors), config={'responsive': False}, style={'height': '450px', 'width': '100%'}), className="p-0 d-flex justify-content-center align-items-center")], style=card_style), width=6),
                    dbc.Col(dbc.Card([dbc.CardHeader(f"{team_b} Pass Sonar", style=header_style_b), dbc.CardBody(dcc.Graph(figure=fig_pass_sonar(df, team_b, colors_b), config={'responsive': False}, style={'height': '450px', 'width': '100%'}), className="p-0 d-flex justify-content-center align-items-center")], style=card_style), width=6),
                ], className="g-4 mb-5")
            ], fluid=True)

        # [Tab 4: Defense]
        elif tab == 'tab-4':
            return dbc.Container([
                dbc.Row([
                    dbc.Col(dbc.Card([dbc.CardHeader(f"{team_a} Defensive Lines", style=header_style), dbc.CardBody(dcc.Graph(figure=fig_defensive(df, team_a, colors), config={'responsive': False}, style={'height': '450px', 'width': '100%'}), className="p-0 d-flex justify-content-center align-items-center")], style=card_style), width=6),
                    dbc.Col(dbc.Card([dbc.CardHeader(f"{team_b} Defensive Lines", style=header_style_b), dbc.CardBody(dcc.Graph(figure=fig_defensive(df, team_b, colors_b), config={'responsive': False}, style={'height': '450px', 'width': '100%'}), className="p-0 d-flex justify-content-center align-items-center")], style=card_style), width=6),
                ], className="g-4 mb-5"),
                dbc.Row([
                    dbc.Col(dbc.Card([dbc.CardHeader(f"{team_a} Defensive Actions", style=header_style), dbc.CardBody(dcc.Graph(figure=fig_defensive_actions(df, team_a, colors), config={'responsive': False}, style={'height': '450px', 'width': '100%'}), className="p-0 d-flex justify-content-center align-items-center")], style=card_style), width=6),
                    dbc.Col(dbc.Card([dbc.CardHeader(f"{team_b} Defensive Actions", style=header_style_b), dbc.CardBody(dcc.Graph(figure=fig_defensive_actions(df, team_b, colors_b), config={'responsive': False}, style={'height': '450px', 'width': '100%'}), className="p-0 d-flex justify-content-center align-items-center")], style=card_style), width=6),
                ], className="g-4 mb-5")
            ], fluid=True)

    elif mode == 'recent':
        team_games = raw_df[raw_df['team_name_ko'] == team]['game_id'].unique()
        sorted_games = sorted(team_games, key=lambda x: int(x), reverse=True)[:count]
        recent_df = raw_df[raw_df['game_id'].isin(sorted_games) & (raw_df['team_name_ko'] == team)]
        
        if tab == 'tab-1':
            return dbc.Container([
                dbc.Row([
                    dbc.Col(dbc.Card([dbc.CardHeader("Recent Form (Goals vs xG)", style=header_style), dbc.CardBody(dcc.Graph(figure=fig_goals_xg_trend(recent_df, team, colors), config={'displayModeBar': False}, style={'height': '300px', 'width': '100%'}))], style=card_style), width=8),
                    dbc.Col(dbc.Card([dbc.CardHeader("Stats Overview (Avg)", style=header_style), dbc.CardBody(generate_recent_stats_table(raw_df, team, count, colors), className="p-0")], style=card_style), width=4),
                ], className="g-3 mb-4")
            ], fluid=True, className="p-3")
            
        elif tab == 'tab-2':
            return dbc.Container([
                dbc.Row([
                    dbc.Col(dbc.Card([dbc.CardHeader("Action Zones", style=header_style), dbc.CardBody(dcc.Graph(figure=fig_action_zones(recent_df, team, colors), config={'displayModeBar': False}, style={'height': '300px'}), className="p-0")], style=card_style), width=6),
                    dbc.Col(dbc.Card([dbc.CardHeader("Attack Direction", style=header_style), dbc.CardBody(dcc.Graph(figure=fig_attack_direction(recent_df, team, colors), config={'displayModeBar': False}, style={'height': '300px'}), className="p-0")], style=card_style), width=6),
                ], className="g-3 mb-4"),
                dbc.Row([
                    dbc.Col(dbc.Card([dbc.CardHeader("Shot Map", style=header_style), dbc.CardBody(dcc.Graph(figure=fig_shot_map(recent_df, team, colors), config={'displayModeBar': False}, style={'height': '500px'}), className="p-0 d-flex justify-content-center align-items-center")], style=card_style), width=12)
                ], className="g-3 mb-4")
            ], fluid=True, className="p-3")

        elif tab == 'tab-3':
            return dbc.Container([
                dbc.Row([
                    dbc.Col(dbc.Card([dbc.CardHeader("Pass Flow", style=header_style), dbc.CardBody(dcc.Graph(figure=fig_pass_flow(recent_df, team, colors), config={'displayModeBar': False}, style={'height': '500px'}), className="p-0 d-flex justify-content-center align-items-center")], style=card_style), width=12),
                ], className="g-3 mb-4")
            ], fluid=True, className="p-3")

        elif tab == 'tab-4':
            return dbc.Container([
                dbc.Row([
                    dbc.Col(dbc.Card([dbc.CardHeader("Defensive Lines", style=header_style), dbc.CardBody(dcc.Graph(figure=fig_defensive(recent_df, team, colors), config={'displayModeBar': False}, style={'height': '450px'}), className="p-0 d-flex justify-content-center align-items-center")], style=card_style), width=6),
                    dbc.Col(dbc.Card([dbc.CardHeader("Defensive Actions", style=header_style), dbc.CardBody(dcc.Graph(figure=fig_defensive_actions(recent_df, team, colors), config={'displayModeBar': False}, style={'height': '450px'}), className="p-0 d-flex justify-content-center align-items-center")], style=card_style), width=6),
                ], className="g-3 mb-4")
            ], fluid=True, className="p-3")

def build_game_context(match_id, team_name, mode):
    # 1. 데이터가 없거나 특정 경기가 아닌 경우
    if not match_id or mode != 'specific':
        return "현재 특정 경기에 대한 데이터가 없습니다. 팀의 전반적인 철학에 대해 이야기하세요."

    try:
        # 2. 해당 경기 데이터 필터링
        game_data = raw_df[raw_df['game_id'] == str(match_id)]
        if game_data.empty: return "데이터를 찾을 수 없습니다."

        # 3. 상대팀 찾기
        teams = game_data['team_name_ko'].unique()
        opp_team = [t for t in teams if t != team_name][0] if len(teams) > 1 else "상대팀"

        # 4. 주요 스탯 계산 (점수, xG, 슈팅)
        my_team_data = game_data[game_data['team_name_ko'] == team_name]
        opp_team_data = game_data[game_data['team_name_ko'] == opp_team]

        my_score = len(my_team_data[my_team_data['result_name'] == 'Goal'])
        opp_score = len(opp_team_data[opp_team_data['result_name'] == 'Goal'])
        
        my_xg = my_team_data['xG'].sum()
        my_shoot = len(my_team_data[my_team_data['type_name'].isin(['Shot', 'Goal'])])
        
        # 5. 경기 결과 판정
        result = "무승부"
        if my_score > opp_score: result = "승리"
        elif my_score < opp_score: result = "패배"

        # 6. AI에게 넘겨줄 요약 텍스트 생성
        context = f"""
        [경기 정보]
        - 상대팀: {opp_team}
        - 결과: {my_score} : {opp_score} ({result})
        - 우리팀 기록: 득점 {my_score}, 기대득점(xG) {my_xg:.2f}, 슈팅수 {my_shoot}개
        
        [상황 설명]
        이 데이터를 바탕으로 경기를 복기하거나 분석하는 투로 말하세요.
        이겼다면 선수들을 칭찬하거나 겸손해하고, 졌다면 원인을 분석하거나 다음을 기약하세요.
        xG(기대득점)가 높았는데 졌다면 "운이 없었다"거나 "결정력이 부족했다"고 말하세요.
        """
        return context
        
    except Exception as e:
        return f"데이터 로드 중 오류 발생: {str(e)}"
    
@app.callback(
    [Output("chat-history-display", "children"),
     Output("chat-store", "data"),
     Output("chat-input", "value")],
    [Input("chat-send-btn", "n_clicks"),
     Input("chat-input", "n_submit")],
    [State("chat-input", "value"),
     State("chat-store", "data"),
     State("team-select", "value"),
     State("analysis-tabs", "value"),
     State("match-select", "value"),
     State("mode-select", "value")]
)


    
def update_chat(n_clicks, n_submit, user_input, chat_history, team, current_tab, match_id, mode):
    # 1. 화면 초기화 및 날짜/감독 계산
    target_date = None
    if mode == 'specific' and match_id:
        try:
            game_row = raw_df[raw_df['game_id'] == str(match_id)]
            if not game_row.empty:
                target_date = str(game_row['game_date'].iloc[0])
        except: pass

    manager_name = get_manager_for_date(team, target_date)
    meta = get_team_metadata(team, target_date)
    manager_img = meta.get('img_manager')

    # 초기 로딩 시
    if (n_clicks == 0 and n_submit is None) or not user_input:
        return display_chat(chat_history, manager_img), chat_history, ""
        
    # 2. 사용자 메시지 저장
    chat_history.append({"role": "user", "text": user_input})
    
    # 3. 프로필 가져오기
    profile = MANAGER_SPEECH_PROFILES.get(manager_name, MANAGER_SPEECH_PROFILES['Default'])
    
    # [핵심] 4. 현재 선택된 경기의 실제 데이터 가져오기 (RAG)
    game_context_data = build_game_context(match_id, team, mode)

    if not GOOGLE_API_KEY:
        bot_reply = "⚠️ API Key가 설정되지 않았습니다."
    else:
        try:
            # System Prompt에 [MATCH DATA] 섹션 추가
            system_prompt = f"""
            [ROLE]
            당신은 {target_date if target_date else '현재'} 시점의 K리그1 {team} 감독, '{manager_name}'입니다.
            
            [MATCH DATA & CONTEXT]
            사용자는 현재 대시보드에서 다음 경기의 데이터를 보고 있습니다. 
            이 데이터를 근거로 답변하세요. 거짓말을 하거나 없는 데이터를 지어내지 마세요.
            {game_context_data}

            [SPEECH RULES]
            - 말투 스타일: {profile['sentence_style']}
            - 관점: {profile['perspective']}
            - 자주 쓰는 표현: {", ".join(profile['frequent_phrases'])}
            - 금지 표현: {", ".join(profile['avoid'])}

            [STRICT]
            - AI임을 밝히지 마십시오.
            - 실제 감독이 인터뷰하듯, 위 경기 데이터(점수, xG 등)를 인용하여 전문적으로 답변하세요.
            - 한국어로 답변하세요.
            """
            
            history_text = "\n".join([f"{msg['role']}: {msg['text']}" for msg in chat_history[-5:]])
            full_prompt = f"{system_prompt}\n\n[대화 기록]\n{history_text}\n\n[현재 질문]\n사용자: {user_input}\n감독:"
            
            response = model.generate_content(full_prompt)
            bot_reply = response.text
            
        except Exception as e:
            bot_reply = f"전술 지시 중 오류가 발생했습니다: {str(e)}"

    chat_history.append({"role": "bot", "text": bot_reply})
    
    return display_chat(chat_history, manager_img), chat_history, ""

def display_chat(history, manager_img):
    if not history:
        return html.Div("감독님과 전술 회의를 시작하세요!", className="text-muted text-center mt-5")
    
    messages = []
    for msg in history:
        if msg['role'] == 'user':
            messages.append(html.Div([
                html.Span(msg['text'], style={'backgroundColor': '#dcf8c6', 'padding': '10px 15px', 'borderRadius': '15px', 'display': 'inline-block', 'maxWidth': '70%'})
            ], style={'textAlign': 'right', 'marginBottom': '10px'}))
        else:
            messages.append(html.Div([
                html.Div([
                    html.Img(src=manager_img, style={'width': '40px', 'height': '40px', 'borderRadius': '50%', 'objectFit': 'cover', 'marginRight': '10px'}),
                    html.Span(msg['text'], style={'backgroundColor': 'white', 'padding': '10px 15px', 'borderRadius': '15px', 'border': '1px solid #e0e0e0', 'display': 'inline-block', 'maxWidth': '70%'})
                ], className="d-flex align-items-start")
            ], style={'textAlign': 'left', 'marginBottom': '10px'}))
            
    return messages

if __name__ == "__main__":
    app.run(debug=True, port=8050)