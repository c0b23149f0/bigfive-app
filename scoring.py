"""
scoring.py - Traitスコア計算エンジン (Big Five性格特性推定)

本モジュールは、研究（対話に基づく説明可能なBig Five性格推定）における
数理モデルに基づいて、Evidence・State（Facetクラスタ）の推論結果から
Big Five特性のスコアを確定的に計算します。

数理モデル:
1. 確信度重み w_{k,f}:
   - High (高): 1.0
   - Medium (中): 0.6
   - Low (低): 0.3
2. 各Facet f の加重平均:
   \bar{s}_f = \frac{\sum_k w_{k,f} \cdot s_{k,f}}{\sum_k w_{k,f}}
3. 根拠件数による飽和補正係数 C_f:
   C_f = 1 - e^{-\lambda N_f}  (N_f は独立なFacetクラスタ数)
4. 観測Facet数による飽和補正係数 C_T と特性スコア S_T^* (-2 ~ +2):
   C_T = 1 - e^{-\gamma |O_T|}  (|O_T| は観測されたFacetの種類数)
   S_T^* = C_T \cdot \frac{1}{|O_T|} \sum_{f \in O_T} (C_f \cdot \bar{s}_f)
5. TIPI-J互換スケール S_T (1.0 ~ 7.0, 0.5刻み):
   S_T = \frac{\text{round}(2 \cdot (4 + 1.5 \cdot S_T^*))}{2}
"""

import math
from typing import Dict, List, Any, Optional, Tuple, Union

# 確信度の重み定義
CONFIDENCE_WEIGHTS = {
    "high": 1.0,
    "高": 1.0,
    "medium": 0.6,
    "med": 0.6,
    "中": 0.6,
    "low": 0.3,
    "低": 0.3
}

# Big Five 5特性の定義（日本語名 -> 英語キー）
BIG_FIVE_TRAITS = {
    "開放性": "Openness",
    "誠実性": "Conscientiousness",
    "勤勉性": "Conscientiousness",
    "外向性": "Extraversion",
    "協調性": "Agreeableness",
    "神経症傾向": "Neuroticism",
    "情緒不安定性": "Neuroticism"
}

# IPIP-NEO 30 Facet の Big Five 所属マッピング（正規化名 -> (TraitKey, TraitJa)）
FACET_TO_TRAIT_MAP = {
    # --- 開放性 (Openness) ---
    "想像力": ("Openness", "開放性"),
    "審美性": ("Openness", "開放性"),
    "芸術的関心": ("Openness", "開放性"),
    "感情性": ("Openness", "開放性"),
    "情動性": ("Openness", "開放性"),
    "行動性": ("Openness", "開放性"),
    "冒険性": ("Openness", "開放性"),
    "冒険心": ("Openness", "開放性"),
    "アイデア": ("Openness", "開放性"),
    "知性": ("Openness", "開放性"),
    "価値観": ("Openness", "開放性"),
    "自由主義": ("Openness", "開放性"),
    
    # --- 誠実性 / 勤勉性 (Conscientiousness) ---
    "能力感": ("Conscientiousness", "誠実性"),
    "自己効力感": ("Conscientiousness", "誠実性"),
    "有能感": ("Conscientiousness", "誠実性"),
    "秩序": ("Conscientiousness", "誠実性"),
    "秩序性": ("Conscientiousness", "誠実性"),
    "責任感": ("Conscientiousness", "誠実性"),
    "義務感": ("Conscientiousness", "誠実性"),
    "達成欲": ("Conscientiousness", "誠実性"),
    "達成努力": ("Conscientiousness", "誠実性"),
    "自己規律": ("Conscientiousness", "誠実性"),
    "自己鍛錬": ("Conscientiousness", "誠実性"),
    "自制心": ("Conscientiousness", "誠実性"),
    "慎重さ": ("Conscientiousness", "誠実性"),
    "熟慮": ("Conscientiousness", "誠実性"),
    
    # --- 外向性 (Extraversion) ---
    "温かさ": ("Extraversion", "外向性"),
    "親しみやすさ": ("Extraversion", "外向性"),
    "友好性": ("Extraversion", "外向性"),
    "社交性": ("Extraversion", "外向性"),
    "群居性": ("Extraversion", "外向性"),
    "断定性": ("Extraversion", "外向性"),
    "自己主張": ("Extraversion", "外向性"),
    "活動性": ("Extraversion", "外向性"),
    "活動水準": ("Extraversion", "外向性"),
    "刺激追求": ("Extraversion", "外向性"),
    "刺激希求性": ("Extraversion", "外向性"),
    "快感情": ("Extraversion", "外向性"),
    "快活さ": ("Extraversion", "外向性"),
    "ポジティブな感情": ("Extraversion", "外向性"),
    
    # --- 協調性 (Agreeableness) ---
    "信頼": ("Agreeableness", "協調性"),
    "素直さ": ("Agreeableness", "協調性"),
    "道徳性": ("Agreeableness", "協調性"),
    "率直さ": ("Agreeableness", "協調性"),
    "利他性": ("Agreeableness", "協調性"),
    "利他主義": ("Agreeableness", "協調性"),
    "順応性": ("Agreeableness", "協調性"),
    "協力性": ("Agreeableness", "協調性"),
    "従順さ": ("Agreeableness", "協調性"),
    "慈愛": ("Agreeableness", "協調性"),
    "謙虚さ": ("Agreeableness", "協調性"),
    "情緒的共感": ("Agreeableness", "協調性"),
    "共感性": ("Agreeableness", "協調性"),
    "同情": ("Agreeableness", "協調性"),
    "優しさ": ("Agreeableness", "協調性"),
    
    # --- 神経症傾向 (Neuroticism) ---
    "不安": ("Neuroticism", "神経症傾向"),
    "怒り": ("Neuroticism", "神経症傾向"),
    "敵意": ("Neuroticism", "神経症傾向"),
    "抑うつ": ("Neuroticism", "神経症傾向"),
    "自意識": ("Neuroticism", "神経症傾向"),
    "自意識過剰": ("Neuroticism", "神経症傾向"),
    "衝動性": ("Neuroticism", "神経症傾向"),
    "無節制": ("Neuroticism", "神経症傾向"),
    "脆弱性": ("Neuroticism", "神経症傾向"),
    "傷つきやすさ": ("Neuroticism", "神経症傾向")
}

# 英語Facet名にも対応
ENGLISH_FACETS = {
    "Fantasy": "想像力", "Aesthetics": "審美性", "Feelings": "感情性",
    "Actions": "行動性", "Ideas": "アイデア", "Values": "価値観",
    "Competence": "能力感", "Order": "秩序性", "Orderliness": "秩序性",
    "Dutifulness": "責任感", "Achievement Striving": "達成努力",
    "Self-Discipline": "自己規律", "Deliberation": "慎重さ", "Cautiousness": "慎重さ",
    "Warmth": "温かさ", "Friendliness": "温かさ", "Gregariousness": "社交性",
    "Assertiveness": "自己主張", "Activity Level": "活動性", "Activity": "活動性",
    "Excitement-Seeking": "刺激追求", "Positive Emotions": "快活さ", "Cheerfulness": "快活さ",
    "Trust": "信頼", "Straightforwardness": "素直さ", "Morality": "道徳性",
    "Altruism": "利他性", "Compliance": "順応性", "Cooperation": "協力性",
    "Modesty": "謙虚さ", "Tender-Mindedness": "情緒的共感", "Sympathy": "共感性",
    "Anxiety": "不安", "Angry Hostility": "怒り", "Anger": "怒り",
    "Depression": "抑うつ", "Self-Consciousness": "自意識",
    "Impulsiveness": "衝動性", "Immoderation": "衝動性",
    "Vulnerability": "傷つきやすさ"
}

def normalize_facet_name(raw_name: str) -> str:
    """Facet名を標準名称に正規化します。"""
    name = raw_name.strip()
    if name in ENGLISH_FACETS:
        return ENGLISH_FACETS[name]
    return name

def get_trait_for_facet(facet_name: str) -> Optional[Tuple[str, str]]:
    """Facet名から対応する (TraitKey, TraitJa) を取得します。"""
    norm = normalize_facet_name(facet_name)
    if norm in FACET_TO_TRAIT_MAP:
        return FACET_TO_TRAIT_MAP[norm]
    # 大文字小文字を無視して英語名検索
    for k, v in ENGLISH_FACETS.items():
        if k.lower() == norm.lower():
            return FACET_TO_TRAIT_MAP.get(v)
    return None

def parse_confidence_weight(confidence: Any) -> float:
    """確信度文字列または数値を重み(0.3~1.0)に変換します。"""
    if isinstance(confidence, (int, float)):
        return float(confidence)
    conf_str = str(confidence).strip().lower()
    return CONFIDENCE_WEIGHTS.get(conf_str, 0.6) # デフォルトは中(0.6)

def parse_strength(strength: Any) -> float:
    """強度を数値(-2.0 ~ +2.0)に正規化します。"""
    try:
        val = float(strength)
        return max(-2.0, min(2.0, val))
    except (ValueError, TypeError):
        return 0.0

def calculate_facet_level_score(
    clusters: List[Dict[str, Any]], 
    lambda_param: float = 0.8
) -> Dict[str, Any]:
    """
    1つのFacetに対するFacet Clusterリストから、Facetスコアと補正値を計算します。
    
    Args:
        clusters: そのFacetに属するFacet Cluster (FC) のリスト
                  各要素は {'strength': float, 'confidence': str/float, ...}
        lambda_param: 根拠件数飽和補正のパラメータ \lambda (デフォルト 0.8)
    
    Returns:
        {
            'raw_count': int (N_f),
            'weighted_mean': float (\bar{s}_f),
            'c_f': float (C_f),
            'corrected_score': float (C_f * \bar{s}_f),
            'clusters': list
        }
    """
    if not clusters:
        return {
            'raw_count': 0,
            'weighted_mean': 0.0,
            'c_f': 0.0,
            'corrected_score': 0.0,
            'clusters': []
        }
    
    n_f = len(clusters)
    sum_weighted_strength = 0.0
    sum_weights = 0.0
    
    for c in clusters:
        s = parse_strength(c.get('strength', 0))
        w = parse_confidence_weight(c.get('confidence', 'medium'))
        sum_weighted_strength += w * s
        sum_weights += w
    
    weighted_mean = (sum_weighted_strength / sum_weights) if sum_weights > 0 else 0.0
    c_f = 1.0 - math.exp(-lambda_param * n_f)
    corrected_score = c_f * weighted_mean
    
    return {
        'raw_count': n_f,
        'weighted_mean': weighted_mean,
        'c_f': c_f,
        'corrected_score': corrected_score,
        'clusters': clusters
    }

def convert_to_tipi_scale(score_star: float) -> float:
    """
    -2.0 ~ +2.0 のスコア S_T^* を、TIPI-J互換の 1.0 ~ 7.0 (0.5刻み) に変換します。
    式: S_T = round(2 * (4 + 1.5 * S_T^*)) / 2
    """
    raw_tipi = 4.0 + 1.5 * score_star
    rounded_tipi = round(2.0 * raw_tipi) / 2.0
    return max(1.0, min(7.0, rounded_tipi))

def calculate_trait_scores(
    facet_evidence_clusters: List[Dict[str, Any]],
    lambda_param: float = 0.8,
    gamma_param: float = 0.5
) -> Dict[str, Any]:
    """
    第2段階統合後のFacet Evidence Clusterリストから、Big Five全5特性のスコアを計算します。
    
    Args:
        facet_evidence_clusters: 第2段階統合の出力クラスタリスト
            形式例: [
                {
                    "facet_cluster_id": "FC001",
                    "trait": "Conscientiousness",
                    "facet": "Orderliness", # または "秩序性"
                    "strength": 1,
                    "confidence": "high",
                    "canonical_evidence": "旅行前に当日の予定を確認する",
                    "reason": "..."
                },
                ...
            ]
        lambda_param: Facet根拠件数の飽和補正パラメータ (デフォルト 0.8)
        gamma_param: 観測Facet数の飽和補正パラメータ (デフォルト 0.5)
        
    Returns:
        {
            'trait_scores': {
                'Extraversion': float (1.0 ~ 7.0),
                'Agreeableness': float,
                'Conscientiousness': float,
                'Neuroticism': float,
                'Openness': float
            },
            'trait_scores_raw': { # S_T^* (-2.0 ~ +2.0)
                'Extraversion': float, ...
            },
            'details': { # 各特性ごとの詳細情報 (説明生成や分析用)
                'Conscientiousness': {
                    'trait_ja': '誠実性',
                    'score_star': float,
                    'tipi_score': float,
                    'c_t': float,
                    'observed_facets_count': int,
                    'facets': {
                        '慎重さ': { ... },
                        ...
                    }
                },
                ...
            }
        }
    """
    # 5特性の初期構造を準備
    trait_keys = ["Extraversion", "Agreeableness", "Conscientiousness", "Neuroticism", "Openness"]
    trait_ja_map = {
        "Extraversion": "外向性",
        "Agreeableness": "協調性",
        "Conscientiousness": "誠実性",
        "Neuroticism": "神経症傾向",
        "Openness": "開放性"
    }
    
    # クラスタを (TraitKey, FacetName) ごとにグループ化
    grouped_by_facet: Dict[str, Dict[str, List[Dict[str, Any]]]] = {t: {} for t in trait_keys}
    
    for item in facet_evidence_clusters:
        raw_facet = item.get('facet', '')
        trait_info = get_trait_for_facet(raw_facet)
        
        if trait_info:
            trait_key, _ = trait_info
            facet_norm = normalize_facet_name(raw_facet)
        else:
            # item内の 'trait' フィールドもフォールバックとしてチェック
            trait_field = item.get('trait', '')
            trait_key = BIG_FIVE_TRAITS.get(trait_field, trait_field)
            facet_norm = normalize_facet_name(raw_facet) if raw_facet else "不明"
            
        if trait_key not in grouped_by_facet:
            continue
            
        if facet_norm not in grouped_by_facet[trait_key]:
            grouped_by_facet[trait_key][facet_norm] = []
        grouped_by_facet[trait_key][facet_norm].append(item)
    
    # 各特性のスコアを計算
    trait_scores = {}
    trait_scores_raw = {}
    details = {}
    
    for t_key in trait_keys:
        t_ja = trait_ja_map[t_key]
        facets_dict = grouped_by_facet[t_key]
        
        # 観測されたFacet集合 O_T
        observed_facets = {}
        sum_facet_corrected = 0.0
        
        for f_name, cluster_list in facets_dict.items():
            f_res = calculate_facet_level_score(cluster_list, lambda_param=lambda_param)
            if f_res['raw_count'] > 0:
                observed_facets[f_name] = f_res
                sum_facet_corrected += f_res['corrected_score']
                
        o_t_count = len(observed_facets)
        
        if o_t_count > 0:
            c_t = 1.0 - math.exp(-gamma_param * o_t_count)
            avg_facet_score = sum_facet_corrected / o_t_count
            s_t_star = c_t * avg_facet_score
        else:
            c_t = 0.0
            avg_facet_score = 0.0
            s_t_star = 0.0
            
        tipi_score = convert_to_tipi_scale(s_t_star)
        
        trait_scores[t_key] = tipi_score
        trait_scores_raw[t_key] = s_t_star
        details[t_key] = {
            'trait_ja': t_ja,
            'score_star': s_t_star,
            'tipi_score': tipi_score,
            'c_t': c_t,
            'observed_facets_count': o_t_count,
            'facets': observed_facets
        }
        
    return {
        'trait_scores': trait_scores,
        'trait_scores_raw': trait_scores_raw,
        'details': details
    }