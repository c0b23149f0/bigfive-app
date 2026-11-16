import os
import csv
import json
import urllib.request
from datetime import datetime
from typing import Dict, Any, Optional

CSV_PATH = os.path.join(os.path.dirname(__file__), '研究データ.csv')
GAS_URL = 'https://script.google.com/macros/s/AKfycbwcPhtouNbfxBZn3eKElQe-d8U6MYQaDwrFCVrN4PFFuptDoYqSOCdnp3Lug_IXN7mwaQ/exec'

HEADER = [
    '回答者ID', '回答日時', '同意',
    'TIPI_Q1', 'TIPI_Q2', 'TIPI_Q3', 'TIPI_Q4', 'TIPI_Q5',
    'TIPI_Q6', 'TIPI_Q7', 'TIPI_Q8', 'TIPI_Q9', 'TIPI_Q10',
    'TIPI_外向性', 'TIPI_協調性', 'TIPI_勤勉性', 'TIPI_神経症傾向', 'TIPI_開放性',
    '推定_外向性', '推定_協調性', '推定_誠実性', '推定_神経症傾向', '推定_開放性',
    '事前_スコア実感合致度(Q1)', '事前_スコア単体納得感(Q2)',
    '事後_根拠込み全体納得感(Q3)',
    '外向性_納得度(1-5)', '外向性_理由',
    '協調性_納得度(1-5)', '協調性_理由',
    '誠実性_納得度(1-5)', '誠実性_理由',
    '神経症傾向_納得度(1-5)', '神経症傾向_理由',
    '開放性_納得度(1-5)', '開放性_理由'
]

TRAIT_ORDER = [
    ("Extraversion", "外向性"),
    ("Agreeableness", "協調性"),
    ("Conscientiousness", "誠実性"),
    ("Neuroticism", "神経症傾向"),
    ("Openness", "開放性")
]

def send_to_spreadsheet(row_data: list):
    """Googleスプレッドシート(GAS)に安全にデータを送信します。"""
    try:
        payload = json.dumps({
            'headers': HEADER,
            'row': row_data
        }).encode('utf-8')
        
        req = urllib.request.Request(
            GAS_URL,
            data=payload,
            headers={'Content-Type': 'application/json'}
        )
        with urllib.request.urlopen(req, timeout=8) as response:
            pass
    except Exception as e:
        print(f'[Warning] スプレッドシート同期に失敗しました: {e}')

def init_db():
    """CSVファイルが存在しない場合は、見出し（ヘッダー）を作成します。"""
    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, mode='w', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(HEADER)

def save_tipi_results(tipi_raw_data: dict, scores: dict) -> str:
    """
    心理テスト（TIPI-J）の回答とスコアを記録し、新規の回答者IDを返します。
    """
    init_db()
    
    subject_id = datetime.now().strftime("%Y%m%d%H%M%S")
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    q_keys = list(tipi_raw_data.keys())
    q_answers = [tipi_raw_data[k] for k in q_keys]
    
    row = [
        subject_id,
        current_time,
        "同意済"
    ] + q_answers + [
        scores.get('Extraversion', ''),
        scores.get('Agreeableness', ''),
        scores.get('Conscientiousness', ''),
        scores.get('Neuroticism', ''),
        scores.get('Openness', '')
    ]
    
    with open(CSV_PATH, mode='a', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row)
        
    return subject_id

def save_experiment_results(
    subject_id: str,
    estimated_scores: Dict[str, Any],
    survey_pre: Dict[str, Any],
    survey_post: Dict[str, Any],
    trait_evaluations: Optional[Dict[str, Dict[str, Any]]] = None
) -> bool:
    """
    AIの推定スコア、事前アンケート、事後全体納得感(Q3)、各特性の個別評価（1〜5と理由）をCSVに保存・更新します。
    """
    init_db()
    
    est_e = estimated_scores.get('Extraversion', '')
    est_a = estimated_scores.get('Agreeableness', '')
    est_c = estimated_scores.get('Conscientiousness', '')
    est_n = estimated_scores.get('Neuroticism', '')
    est_o = estimated_scores.get('Openness', '')
    
    pre_q1 = survey_pre.get('q1', '')
    pre_q2 = survey_pre.get('q2', '')
    post_q3 = survey_post.get('q3', '')
    
    # 5特性それぞれの納得度(1-5)と理由
    eval_values = []
    if trait_evaluations is None:
        trait_evaluations = {}
        
    for trait_key, _ in TRAIT_ORDER:
        t_data = trait_evaluations.get(trait_key, {})
        rating = t_data.get('rating', '')
        reason = t_data.get('reason', '')
        eval_values.extend([rating, reason])
    
    append_values = [est_e, est_a, est_c, est_n, est_o, pre_q1, pre_q2, post_q3] + eval_values
    
    # 既存のCSVを全行読み込み
    rows = []
    updated = False
    
    if os.path.exists(CSV_PATH):
        with open(CSV_PATH, mode='r', encoding='utf-8-sig', newline='') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for r in reader:
                if not r:
                    continue
                # IDが一致する行を更新
                if str(r[0]).strip() == str(subject_id).strip():
                    base_row = r[:18]  # ID〜TIPIスコアまで
                    while len(base_row) < 18:
                        base_row.append('')
                    new_row = base_row + append_values
                    rows.append(new_row)
                    updated = True
                else:
                    rows.append(r)
    
    # 一致するIDがなかった場合は新規行として追加
    if not updated:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        dummy_tipi = [''] * 15  # 同意 + Q1-10 + TIPIスコア5
        new_row = [subject_id, current_time] + dummy_tipi + append_values
        rows.append(new_row)
        
    # 上書き保存
    with open(CSV_PATH, mode='w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)
        writer.writerows(rows)
        
    # クラウド/ローカル両対応: Googleスプレッドシートへ自動送信
    target_row = next((r for r in rows if str(r[0]).strip() == str(subject_id).strip()), None)
    if target_row:
        send_to_spreadsheet(target_row)
        
    return True
