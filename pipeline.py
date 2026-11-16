"""
pipeline.py - Chain-of-Personality-Evidence (CoPE) 推論パイプライン

対話ログから、以下の段階的Prompt ChainingによりBig Five性格特性の推定と
説明文の生成を一気通貫で実行します。

全7ステップ構成:
- ステップ1: 情報収集対話 (対話ログ: session_id, utterances)
- ステップ2: Evidence抽出 (発話から客観的事実を抽出)
- ステップ3: 第1段階統合 (同一場面・同一行動の言い換えをECへ統合)
- ステップ4: State推定 (IPIP-NEO 30 Facetへの対応付け、強度・確信度・理由)
- ステップ5: 第2段階統合 (同一Facet内での同一エピソード統合 -> FC)
- ステップ6: Traitスコア計算 (scoring.py による確定数理計算)
- ステップ7: 推定結果と根拠の表示 (説明文生成)
"""

import os
import json
import re
from typing import Dict, List, Any, Optional, Callable

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    _env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(_env_path):
        with open(_env_path, "r", encoding="utf-8") as _f:
            for _line in _f:
                _line = _line.strip()
                if _line and not _line.startswith("#") and "=" in _line:
                    _k, _v = _line.split("=", 1)
                    os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))

import scoring

# ==========================================
# 1. プロンプト定義
# ==========================================

# IPIP-NEO 30 Facet 定義テキスト
IPIP_NEO_FACETS_TEXT = """
【IPIP-NEO 30 Facet (下位因子) 定義】
■ 外向性 (Extraversion)
- 温かさ (Warmth / 親しみやすさ): 親しみやすく優しい対人関係を築く傾向。
- 社交性 (Gregariousness): 人との関わりを好み、集団活動を好む傾向。
- 自己主張 (Assertiveness / 断定性): 意見をはっきり言い、主導権を握る傾向。
- 活動性 (Activity Level): 活発でエネルギッシュ、テンポよく行動する傾向。
- 刺激追求 (Excitement-Seeking / 刺激希求性): 新しい刺激やスリル、冒険を求める傾向。
- 快活さ (Positive Emotions / 快感情): 喜びや楽しさを感じやすく陽気に振る舞う傾向。

■ 協調性 (Agreeableness)
- 信頼 (Trust): 他人は基本的に善意を持つと信じ、疑いにくい傾向。
- 率直さ (Straightforwardness / 道徳性 / 素直さ): 嘘や誤魔化しを嫌い、実直に接する傾向。
- 利他性 (Altruism / 利他主義): 自分の利益より他人のために尽くす傾向。
- 順応性 (Compliance / 協力性): 争いを避け、意見が対立した時に譲歩・調和する傾向。
- 謙虚さ (Modesty / 慈愛): 自分を過信・自慢せず控えめな態度をとる傾向。
- 共感性 (Tender-Mindedness / 情緒的共感): 他人の感情や苦しみに深く共感し優しい傾向。

■ 誠実性 (Conscientiousness)
- 自己効力感 (Competence / 能力感 / 有能感): 物事を計画的・効果的にやり遂げる自信を持つ傾向。
- 秩序性 (Order / 秩序 / 整理整頓): 順序立てて進めたり、整理整頓・時間管理を重視する傾向。
- 責任感 (Dutifulness / 義務感): 約束やルール、倫理的原則を忠実に守る傾向。
- 達成努力 (Achievement Striving / 達成欲): 高い目標を設定し、粘り強く努力する傾向。
- 自己規律 (Self-Discipline / 自制心): 誘惑に負けず、怠け心を抑えてやり抜く傾向。
- 慎重さ (Deliberation / 熟慮 / 慎重): 行動前に熟考し計画を練り、リスクや軽率な行動を避ける傾向。

■ 神経症傾向 (Neuroticism)
- 不安 (Anxiety): 心配性で緊張しやすく、物事の悪い結果を予想しやすい傾向。
- 怒り (Angry Hostility / 敵意): フラストレーションを感じやすく、怒りや苛立ちを経験しやすい傾向。
- 抑うつ (Depression): 落ち込みやすく、悲観的・自責的になりやすい傾向。
- 自意識 (Self-Consciousness / 自意識過剰): 他人からの評価を過度に気にし、恥ずかしがりな傾向。
- 衝動性 (Impulsiveness / 無節制): 目の前の誘惑や一時的な欲求に流されやすい傾向。
- 傷つきやすさ (Vulnerability / 脆弱性): ストレスや困難に弱く、冷静さを失いやすい傾向。

■ 開放性 (Openness to Experience)
- 想像力 (Fantasy): 空想力が豊かで、現実離れした内面世界を楽しむ傾向。
- 審美性 (Aesthetics / 芸術的関心): 芸術、音楽、自然の美しさに深く感動する傾向。
- 感情性 (Feelings / 情動性): 自分の心の動きや感情の起伏に敏感で深く味わう傾向。
- 冒険性 (Actions / 行動性 / 冒険心): 新しい場所に行ったり、未知の活動に挑戦することを好む傾向。
- 知性 (Ideas / アイデア): 抽象的な概念、哲学的な議論、知的な探求を楽しむ傾向。
- 自由主義 (Values / 価値観): 伝統や常識にとらわれず、多様な価値観を柔軟に受け入れる傾向。
"""

# プロンプト2: Evidence抽出
PROMPT_STEP2_EXTRACTION = """あなたは性格分析の専門家です。
入力された対話ログの中から、ユーザの性格特性や行動傾向の判断根拠となる【客観的な事実に基づいた発話部分】を検出し、個別の根拠（Evidence）として抽出してください。

【抽出ルール】
1. ユーザ（user）の発話のみを対象とします（システムの発話からは抽出しない）。
2. 単なる相槌や挨拶ではなく、具体的な行動、経験、習慣、こだわり、思考プロセスが語られている箇所を抽出してください。
3. 元の発話テキスト（source_text）と、客観的な行動事実として簡潔に要約したテキスト（evidence_text）を対にしてください。
4. 該当する根拠がない場合は空のリストを返してください。

【出力JSONフォーマット】
必ず以下のJSON形式のみを出力してください。
{
  "evidence": [
    {
      "evidence_id": "E01",
      "source_text": "発話の該当部分の引用",
      "evidence_text": "行動事実の要約"
    }
  ]
}
"""

# プロンプト3: 第1段階統合
PROMPT_STEP3_DEDUPLICATION = """あなたは性格分析の専門家です。
抽出されたEvidenceリストから、同一の出来事や同じ文脈に関する言い換え・重複説明を検出し、1つのクラスタ（Evidence Cluster: EC）に統合してください。

【統合判定ルール】
■ 重複（統合する）:
- 同じ出来事を別の言葉で説明している
- 同じ行動を繰り返し言い直している
- 一方が他方の補足説明や言い換えにすぎない
- 同じ発話を複数の短いEvidenceに分割した結果、内容が重なっている

■ 独立（統合しない・個別クラスタとする）:
- 異なる時期や場面で生じた行動である
- 行動の目的や対象が異なる
- 同じ性格傾向を示していても、別の経験や行動に基づいている

【出力JSONフォーマット】
必ず以下のJSON形式のみを出力してください。
{
  "evidence_clusters": [
    {
      "evidence_cluster_id": "EC001",
      "canonical_evidence": "統合・代表化した行動事実の要約文",
      "member_evidence_ids": ["E01", "E02"],
      "source_texts": ["元発話1の引用", "元発話2の引用"]
    }
  ]
}
"""

# プロンプト4: State推定
PROMPT_STEP4_STATE_MAPPING = """あなたは性格心理学の専門家です。
入力された各Evidenceクラスタ（EC）が、提供されたIPIP-NEOの30個のFacet（下位因子）のどれに最も強く結びついているかを分析・推定してください。

""" + IPIP_NEO_FACETS_TEXT + """

【判定基準】
1. **Facet**: 最も関連の深いFacet名を上記リストから1つ選択（例: "慎重さ", "秩序性", "不安" 等）。
2. **Trait**: そのFacetが属するBig Five特性名（"Extraversion", "Agreeableness", "Conscientiousness", "Neuroticism", "Openness" のいずれか）。
3. **強度 (strength)**: その行動によってFacetの傾向がどの方向にどれくらい強く表れているかを -2 〜 +2 の整数で評価。
   - +2: 非常に強い正の傾向（例: 極めて慎重、非常に計画的）
   - +1: やや正の傾向（例: 少し慎重、やや几帳面）
   -  0: 中立的
   - -1: やや負の傾向（例: やや無計画、少し大雑把）
   - -2: 非常に強い負の傾向（例: 著しく無計画・衝動的）
4. **確信度 (confidence)**: その行動をそのFacetに割り振る解釈の一意性・妥当性を評価。
   - "high": 解釈が明確で一意にそのFacetに決まる
   - "medium": 複数の解釈が考えられるが妥当性が高い
   - "low": 間接的・断片的であり解釈にブレが生じうる
5. **理由 (reason)**: なぜそのFacet・強度・確信度と判断したのかの論理的根拠を簡潔に記述。

【出力JSONフォーマット】
必ず以下のJSON形式のみを出力してください。
{
  "facet_mappings": [
    {
      "mapping_id": "M001",
      "evidence_cluster_id": "EC001",
      "trait": "Conscientiousness",
      "facet": "慎重さ",
      "strength": 2,
      "confidence": "high",
      "reason": "問題の発生を予測し、事前に代替案を準備しているため"
    }
  ]
}
"""

# プロンプト5: 第2段階統合
PROMPT_STEP5_FACET_DEDUPLICATION = """あなたは性格心理学の専門家です。
同じFacet（下位因子）に対応付けられた複数のStateマッピングについて比較し、同一出来事・同一エピソードに基づくものを1つのクラスタ（Facet Cluster: FC）に統合してください。

【統合判定ルール】
次の条件を【すべて満たす場合のみ】同一のFacet Clusterへ統合（relation: "duplicate"）してください:
1. 同じFacetを支持している
2. 同一の出来事または同一文脈に基づいている
3. 同一の性格傾向を示している
4. 一方が他方の言い換え・補足・具体例として説明できる

上記を満たさない（異なる経験・別場面・独立した行動）場合は、独立したFacet Cluster（relation: "independent"）として扱ってください。

【出力JSONフォーマット】
必ず以下のJSON形式のみを出力してください。
{
  "facet_evidence_clusters": [
    {
      "facet_cluster_id": "FC001",
      "trait": "Conscientiousness",
      "facet": "慎重さ",
      "member_mapping_ids": ["M001", "M002"],
      "representative_mapping_id": "M001",
      "canonical_evidence": "旅行前に当日の予定や安全対策を確認する",
      "relation": "duplicate",
      "strength": 2,
      "confidence": "high",
      "reason": "いずれも同一の旅行計画における事前準備の具体例であるため統合"
    }
  ]
}
"""

# プロンプト7: 推定結果と根拠の表示 (説明文生成)
PROMPT_STEP7_EXPLANATION = """あなたは性格診断結果を説明するカウンセラーです。
以下はプログラムによって厳密に計算されたBig Five性格推定スコアと、その根拠データです。
与えられたデータに基づいて、ユーザに向けた丁寧で納得感のある説明文を作成してください。

【厳格な遵守事項】
1. **スコアの改変禁止**: 提示されたスコア（1〜7）を勝手に変更したり、独自に再計算しないでください。
2. **ハルシネーションの禁止**: 元発話や根拠データに存在しないエピソードを捏造して付け加えないでください。
3. **新規推測の禁止**: スコアから飛躍して、データにない新たな性格特徴を勝手に推測しないでください。
4. **内部変数名・プログラム用語の出力厳禁（最重要）**:
   `observed_facets_count` や `trait_scores_tipi` などのJSONキー名や英数字の内部変数名を文章中にそのまま出力することは絶対に禁止です。
   観測データが存在しない特性については、必ず「今回の対話からは、この特性に関連する具体的な行動や発言は観察されませんでした（基準値の4.0点となっています）」のように、自然で丁寧な日本語で説明してください。
5. **説明に含めるべき内容**:
   - 各Big Five特性の 1〜7 のスコア
   - スコアに寄与した主要なFacet（下位因子）
   - その根拠となった実際のユーザ発話の引用と、判断理由
   - 観測が不足しているFacetや特性についての自然な言及
   - 「この結果は今回の対話中に観測された具体的な発話に基づく推定であり、あなたの人格全体を確定するものではありません」という注記

【出力形式】
Markdown形式で読みやすく整理して出力してください。
"""

# ==========================================
# 2. JSON抽出・LLM呼び出しヘルパー
# ==========================================

def extract_json_from_text(text: str) -> Any:
    """LLMの出力テキストからJSONを安全に抽出・パースします。"""
    cleaned = text.strip()
    
    # ```json ... ``` コードブロックを抽出
    code_block_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned, re.IGNORECASE)
    if code_block_match:
        cleaned = code_block_match.group(1).strip()
        
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # 最初と最後の { } または [ ] を探す
        first_brace = cleaned.find('{')
        last_brace = cleaned.rfind('}')
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            sub = cleaned[first_brace:last_brace+1]
            try:
                return json.loads(sub)
            except json.JSONDecodeError:
                pass
                
        first_bracket = cleaned.find('[')
        last_bracket = cleaned.rfind(']')
        if first_bracket != -1 and last_bracket != -1 and last_bracket > first_bracket:
            sub = cleaned[first_bracket:last_bracket+1]
            try:
                return json.loads(sub)
            except json.JSONDecodeError:
                pass
                
        raise ValueError(f"JSONのパースに失敗しました。応答内容:\n{text}")

def call_gemini_api(
    prompt: str,
    system_instruction: Optional[str] = None,
    client: Any = None,
    model_name: str = "gemini-3.5-flash-lite"
) -> str:
    """Gemini APIを呼び出してテキスト応答を取得します。"""
    if client is not None:
        if hasattr(client, "generate_content"):
            res = client.generate_content(prompt)
            return res.text
        elif callable(client):
            return client(prompt, system_instruction)
            
    import google.generativeai as genai
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY が環境変数に設定されていません。")
        
    genai.configure(api_key=api_key)
    
    generation_config = {
        "temperature": 0.2, # 推論の再現性と安定性を重視。ランダム性のやつ
    }
    
    model = genai.GenerativeModel(
        model_name=model_name,
        system_instruction=system_instruction,
        generation_config=generation_config
    )
    
    response = model.generate_content(prompt)
    
    # 安全にテキストを取り出す
    try:
        if hasattr(response, "text") and response.text:
            return response.text
    except Exception:
        pass
        
    if hasattr(response, "candidates") and response.candidates:
        cand = response.candidates[0]
        if hasattr(cand, "content") and hasattr(cand.content, "parts"):
            parts_text = [p.text for p in cand.content.parts if hasattr(p, "text") and p.text]
            if parts_text:
                return "".join(parts_text)
                
    raise ValueError(f"Gemini APIからのテキスト取得に失敗しました。Response: {response}")

# ==========================================
# 3. パイプライン各ステップの実装
# ==========================================

def step2_extract_evidence(
    utterances: List[Dict[str, Any]],#対話ログのリストのやつ
    client: Any = None,
    model_name: str = "gemini-3.6-flash"
) -> List[Dict[str, Any]]:
    """ステップ2: 対話ログからEvidenceを抽出します。"""
    dialogue_json = json.dumps({"utterances": utterances}, ensure_ascii=False, indent=2)
    prompt = f"以下の対話ログから、性格判断の根拠となる事実（Evidence）を抽出してください。\n\n{dialogue_json}"
    
    resp_text = call_gemini_api(prompt, system_instruction=PROMPT_STEP2_EXTRACTION, client=client, model_name=model_name)
    data = extract_json_from_text(resp_text)
    
    if isinstance(data, dict) and "evidence" in data:
        return data["evidence"]
    elif isinstance(data, list):
        return data
    return []

def step3_first_stage_dedup(
    evidence_list: List[Dict[str, Any]],
    client: Any = None,
    model_name: str = "gemini-3.6-flash"
) -> List[Dict[str, Any]]:


    """ステップ3: Evidenceリストから同一場面の重複を統合（EC生成）。"""
    if not evidence_list:
        return []
        
    evidence_json = json.dumps({"evidence": evidence_list}, ensure_ascii=False, indent=2)
    prompt = f"以下のEvidenceリストから重複を統合し、Evidence Cluster (EC) リストを作成してください。\n\n{evidence_json}"
    
    resp_text = call_gemini_api(prompt, system_instruction=PROMPT_STEP3_DEDUPLICATION, client=client, model_name=model_name)
    data = extract_json_from_text(resp_text)
    
    if isinstance(data, dict) and "evidence_clusters" in data:
        return data["evidence_clusters"]
    elif isinstance(data, list):
        return data
    return []

def step4_state_estimation(
    evidence_clusters: List[Dict[str, Any]],
    client: Any = None,
    model_name: str = "gemini-3.6-flash"
) -> List[Dict[str, Any]]:
    """ステップ4: Evidence Cluster (EC) を IPIP-NEO 30 Facet に対応付け (State推定)。"""
    if not evidence_clusters:
        return []
        
    ec_json = json.dumps({"evidence_clusters": evidence_clusters}, ensure_ascii=False, indent=2)
    prompt = f"以下のEvidenceクラスタリストを30個のFacetに対応付け、強度・確信度・理由を判定してください。\n\n{ec_json}"
    
    resp_text = call_gemini_api(prompt, system_instruction=PROMPT_STEP4_STATE_MAPPING, client=client, model_name=model_name)
    data = extract_json_from_text(resp_text)
    
    if isinstance(data, dict) and "facet_mappings" in data:
        return data["facet_mappings"]
    elif isinstance(data, list):
        return data
    return []

def step5_second_stage_dedup(
    facet_mappings: List[Dict[str, Any]],
    client: Any = None,
    model_name: str = "gemini-3.6-flash"
) -> List[Dict[str, Any]]:
    """ステップ5: 同一Facet内でのエピソード重複統合 (FC生成)。"""
    if not facet_mappings:
        return []
        
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for m in facet_mappings:
        facet = m.get("facet", "不明")
        grouped.setdefault(facet, []).append(m)
        
    all_facet_clusters = []
    fc_counter = 1
    
    for facet, mappings in grouped.items():
        if len(mappings) == 1:
            m = mappings[0]
            fc_id = f"FC{fc_counter:03d}"
            fc_counter += 1
            all_facet_clusters.append({
                "facet_cluster_id": fc_id,
                "trait": m.get("trait", ""),
                "facet": facet,
                "member_mapping_ids": [m.get("mapping_id", "")],
                "representative_mapping_id": m.get("mapping_id", ""),
                "canonical_evidence": m.get("reason", ""),
                "relation": "independent",
                "strength": m.get("strength", 0),
                "confidence": m.get("confidence", "medium"),
                "reason": m.get("reason", "")
            })
        else:
            req_data = {
                "task": "facet_level_deduplication",
                "facet": facet,
                "mappings": mappings
            }
            prompt = f"以下のFacet '{facet}' に関するマッピングリストを比較し、同一出来事の重複を統合してください。\n\n{json.dumps(req_data, ensure_ascii=False, indent=2)}"
            resp_text = call_gemini_api(prompt, system_instruction=PROMPT_STEP5_FACET_DEDUPLICATION, client=client, model_name=model_name)
            data = extract_json_from_text(resp_text)
            
            clusters = []
            if isinstance(data, dict) and "facet_evidence_clusters" in data:
                clusters = data["facet_evidence_clusters"]
            elif isinstance(data, list):
                clusters = data
                
            for c in clusters:
                if "facet_cluster_id" not in c:
                    c["facet_cluster_id"] = f"FC{fc_counter:03d}"
                    fc_counter += 1
                all_facet_clusters.append(c)
                
    return all_facet_clusters

def step7_generate_explanation(
    trait_scores_result: Dict[str, Any],
    facet_clusters: List[Dict[str, Any]],
    client: Any = None,
    model_name: str = "gemini-3.5-flash-lite"
) -> str:
    """ステップ7: 計算されたスコアと根拠データから説明文を生成します。"""
    payload = {
        "trait_scores_tipi": trait_scores_result.get("trait_scores", {}),
        "details": trait_scores_result.get("details", {}),
        "facet_clusters": facet_clusters
    }
    
    prompt = (
        "以下の計算結果および根拠データに基づき、ユーザへの性格推定結果と説明文を生成してください。\n"
        "※プログラムの英字キー名（例: observed_facets_count など）は絶対に出力せず、自然で分かりやすい日本語で表現してください。\n\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )
    return call_gemini_api(prompt, system_instruction=PROMPT_STEP7_EXPLANATION, client=client, model_name=model_name)

# ==========================================
# 4. パイプライン統合実行エントリーポイント
# ==========================================

ALL_BIG_FIVE_TRAITS = [
    "Extraversion",
    "Agreeableness",
    "Conscientiousness",
    "Neuroticism",
    "Openness"
]

def run_intermediate_analysis(
    utterances: List[Dict[str, Any]],
    client: Any = None,
    model_name: str = "gemini-3.5-flash-lite"
) -> Dict[str, Any]:
    """
    ステップ2〜5を実行し、現在対話から観測できている特性・未観測の特性を特定します。
    対話の話題転換や終了判定を行うための軽量な中間解析です。
    """
    # ユーザ発話がない場合は未観測のまま即座に返す
    user_utterances = [u for u in utterances if u.get("speaker") == "user" and u.get("text", "").strip()]
    if not user_utterances:
        return {
            "observed_traits": [],
            "unobserved_traits": list(ALL_BIG_FIVE_TRAITS),
            "observed_facets": [],
            "fc_count": 0,
            "is_ready_to_finish": False,
            "facet_clusters": []
        }

    try:
        # ステップ2: Evidence抽出
        evidence_list = step2_extract_evidence(utterances, client=client, model_name=model_name)
        if not evidence_list:
            return {
                "observed_traits": [],
                "unobserved_traits": list(ALL_BIG_FIVE_TRAITS),
                "observed_facets": [],
                "fc_count": 0,
                "is_ready_to_finish": False,
                "facet_clusters": []
            }
        
        # ステップ3: 第1段階統合
        evidence_clusters = step3_first_stage_dedup(evidence_list, client=client, model_name=model_name)
        
        # ステップ4: State推定
        facet_mappings = step4_state_estimation(evidence_clusters, client=client, model_name=model_name)
        
        # ステップ5: 第2段階統合
        facet_clusters = step5_second_stage_dedup(facet_mappings, client=client, model_name=model_name)
        
        # 観測された特性・Facetの集計
        observed_traits_set = set()
        observed_facets = []
        for fc in facet_clusters:
            trait = fc.get("trait")
            facet = fc.get("facet")
            if trait and trait in ALL_BIG_FIVE_TRAITS:
                observed_traits_set.add(trait)
            if facet and facet not in observed_facets:
                observed_facets.append(facet)
                
        observed_traits = [t for t in ALL_BIG_FIVE_TRAITS if t in observed_traits_set]
        unobserved_traits = [t for t in ALL_BIG_FIVE_TRAITS if t not in observed_traits_set]
        
        # 終了判定基準: 4つ以上の特性が観測された、または独立クラスタ（FC）が5個以上得られたら終了準備完了
        is_ready = (len(observed_traits) >= 4) or (len(facet_clusters) >= 5)
        
        return {
            "observed_traits": observed_traits,
            "unobserved_traits": unobserved_traits,
            "observed_facets": observed_facets,
            "fc_count": len(facet_clusters),
            "is_ready_to_finish": is_ready,
            "facet_clusters": facet_clusters
        }
    except Exception as e:
        # 中間解析でエラーが起きても対話全体を止めないようフォールバック
        return {
            "error": str(e),
            "observed_traits": [],
            "unobserved_traits": list(ALL_BIG_FIVE_TRAITS),
            "observed_facets": [],
            "fc_count": 0,
            "is_ready_to_finish": False,
            "facet_clusters": []
        }

def run_cope_pipeline(
    utterances: List[Dict[str, Any]],
    session_id: str = "default_session",
    client: Any = None,
    model_name: str = "gemini-3.5-flash-lite",
    lambda_param: float = 0.8,
    gamma_param: float = 0.5
) -> Dict[str, Any]:
    """
    CoPE 全7ステップの推論パイプラインを実行し、全中間JSONおよび結果を返します。
    """
    # ステップ2: Evidence抽出
    evidence_list = step2_extract_evidence(utterances, client=client, model_name=model_name)
    
    # ステップ3: 第1段階統合
    evidence_clusters = step3_first_stage_dedup(evidence_list, client=client, model_name=model_name)
    
    # ステップ4: State推定
    facet_mappings = step4_state_estimation(evidence_clusters, client=client, model_name=model_name)
    
    # ステップ5: 第2段階統合
    facet_clusters = step5_second_stage_dedup(facet_mappings, client=client, model_name=model_name)
    
    # ステップ6: Traitスコア計算 (確定プログラム)
    scores_result = scoring.calculate_trait_scores(
        facet_clusters,
        lambda_param=lambda_param,
        gamma_param=gamma_param
    )
    
    # ステップ7: 推定結果と根拠の表示 (説明文生成)
    explanation = step7_generate_explanation(
        scores_result,
        facet_clusters,
        client=client,
        model_name=model_name
    )
    
    return {
        "session_id": session_id,
        "step2_evidence": evidence_list,
        "step3_evidence_clusters": evidence_clusters,
        "step4_state_mappings": facet_mappings,
        "step5_facet_clusters": facet_clusters,
        "step6_trait_scores": scores_result.get("trait_scores", {}),
        "step6_trait_scores_raw": scores_result.get("trait_scores_raw", {}),
        "step6_details": scores_result.get("details", {}),
        "step7_explanation": explanation
    }

# ==========================================
# 5. 比較用ベースライン手法 (スライド38, 62)
# ==========================================

PROMPT_DIRECT_ESTIMATION = """あなたは性格診断の専門家です。
入力された対話ログ全体から、ユーザのBig Five性格特性（1〜7の0.5刻み）を直接推定してください。

【評価対象】
- Extraversion (外向性): 1.0〜7.0
- Agreeableness (協調性): 1.0〜7.0
- Conscientiousness (勤勉性/誠実性): 1.0〜7.0
- Neuroticism (神経症傾向): 1.0〜7.0
- Openness (開放性): 1.0〜7.0

【出力JSONフォーマット】
{
  "trait_scores": {
    "Extraversion": 5.0,
    "Agreeableness": 4.5,
    "Conscientiousness": 6.0,
    "Neuroticism": 3.0,
    "Openness": 5.5
  },
  "explanation": "推定理由の説明"
}
"""

def run_direct_estimation_baseline(
    utterances: List[Dict[str, Any]],
    session_id: str = "default_session",
    client: Any = None,
    model_name: str = "gemini-3.6-flash"
) -> Dict[str, Any]:
    """比較手法1: 対話ログから直接LLMでBig Fiveを推定するベースライン。"""
    dialogue_json = json.dumps({"utterances": utterances}, ensure_ascii=False, indent=2)
    prompt = f"以下の対話ログからBig Five性格特性を推定してください。\n\n{dialogue_json}"
    
    resp_text = call_gemini_api(prompt, system_instruction=PROMPT_DIRECT_ESTIMATION, client=client, model_name=model_name)
    data = extract_json_from_text(resp_text)
    
    return {
        "session_id": session_id,
        "method": "direct_estimation",
        "trait_scores": data.get("trait_scores", {}),
        "explanation": data.get("explanation", "")
    }