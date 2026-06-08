import os
import re
import numpy as np
import joblib
import pandas as pd
import MeCab

# app/ 폴더의 상위 기준 경로 제어
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 
MODELS_DIR = os.path.join(BASE_DIR, "ml", "models")

# PhishTank_2026.csv 로컬 파일 경로 정의
PHISHTANK_CSV_PATH = os.path.join(BASE_DIR, "..", "data", "csv", "PhishTank_2026.csv")

mecab = MeCab.Tagger()

class SmishingOverseerPredictor:
    def __init__(self, threshold=0.10):
        self.threshold = threshold
        self.phishtank_db = set()
        
        # 1. AI 모델 아티팩트 메모리 상주
        self.vectorizer = joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"))
        self.models = {
            "RF": joblib.load(os.path.join(MODELS_DIR, "randomforest_sms_without500_phishing_model.pkl")),
            "SVM": joblib.load(os.path.join(MODELS_DIR, "svm_sms_without500_phishing_model.pkl")),
            "NB": joblib.load(os.path.join(MODELS_DIR, "naivebayes_sms_without500_phishing_model.pkl")),
            "LR": joblib.load(os.path.join(MODELS_DIR, "logisticregression_sms_without500_phishing_model.pkl"))
        }
        self.overseer = joblib.load(os.path.join(MODELS_DIR, "xgboost_overseer_sms.joblib"))
        
        # 2. 로컬 CSV 기반 블랙리스트 고속 메모리 적재로 변경
        self._load_local_phishtank_db()

    def _load_local_phishtank_db(self):
        """서버 부팅 시 딱 한 번 디스크의 CSV 파일을 읽어 RAM에 set 구조로 박아넣음"""
        try:
            if os.path.exists(PHISHTANK_CSV_PATH):
                # 대용량일 경우를 대비해 필요한 'URL' 컬럼만 지정해서 고속 로드
                df = pd.read_csv(PHISHTANK_CSV_PATH, usecols=['URL'])
                
                # 공백 제거 및 문자열 변환 처리 후 탐색 속도가 O(1)인 set 구조로 변환
                self.phishtank_db = set(df['URL'].astype(str).str.strip().tolist())
                print(f"로컬 PhishTank DB 로드 완료 (총 {len(self.phishtank_db):,}개 URL 메모리 상주)")
            else:
                print(f"PhishTank 파일 없음 ({PHISHTANK_CSV_PATH}). 오직 ML 앙상블로만 판정합니다.")
        except Exception as e:
            print(f"PhishTank 로드 중 오류 발생: {e}")

    def _analyze_features(self, text_str):
        text_str = str(text_str)
        total_len = len(text_str)
        eng_cnt = len(re.sub(r'[^a-zA-Z]', '', text_str))
        num_cnt = len(re.sub(r'[^0-9]', '', text_str))
        special_cnt = len(re.sub(r'[a-zA-Z0-9가-힣\s]', '', text_str))
        
        noun_cnt, josa_cnt, verb_cnt, adv_cnt = 0, 0, 0, 0
        node = mecab.parseToNode(text_str)
        while node:
            surface = node.surface
            tag = node.feature.split(',')[0]
            if surface:
                if tag.startswith('N'):   noun_cnt += 1
                elif tag.startswith('J'): josa_cnt += 1
                elif tag.startswith('V'): verb_cnt += 1
                elif tag.startswith('M'): adv_cnt += 1
            node = node.next
        return [total_len, eng_cnt, num_cnt, special_cnt, noun_cnt, josa_cnt, verb_cnt, adv_cnt]

    def predict(self, full_text, extracted_urls):
        # 1차 가드: 추출된 URL이 로컬에서 불려온 피쉬탱크 해시 셋에 걸리는지 대조
        for url in extracted_urls:
            if url in self.phishtank_db:
                return {"is_phishing": True, "phishing_prob": 1.0, "source": "phishtank"}
        
        # 2차 가드: 이상 없을 시 92%짜리 AI 가동
        vec_text = self.vectorizer.transform([full_text])
        rf_prob = self.models["RF"].predict_proba(vec_text)[0, 1]
        svm_prob = self.models["SVM"].predict_proba(vec_text)[0, 1]
        nb_prob = self.models["NB"].predict_proba(vec_text)[0, 1]
        lr_prob = self.models["LR"].predict_proba(vec_text)[0, 1]
        
        vote_vector = np.array([rf_prob, svm_prob, nb_prob, lr_prob])
        features = np.array([self._analyze_features(full_text)])
        pred_weights = self.overseer.predict(features)[0]
        
        final_prob = float(np.sum(vote_vector * pred_weights))
        is_phishing = bool(final_prob >= self.threshold)
        
        return {"is_phishing": is_phishing, "phishing_prob": round(final_prob, 4), "source": "overseer_ai"}