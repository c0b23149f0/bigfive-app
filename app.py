import streamlit as st
import database
import pipeline
import sqlite3
import os
import google.generativeai as genai
from dotenv import load_dotenv

# 環境変数（APIキー等）の読み込み
load_dotenv()
gemini_api_key = os.environ.get("GEMINI_API_KEY")
if gemini_api_key:
    genai.configure(api_key=gemini_api_key)

# --- ページ設定と制作者情報（コード内のみ） ---
# Author: 吉松遼馬 (Ryoma Yoshimatsu)
st.set_page_config(page_title="性格特性に関する研究アンケート", layout="centered")

# --- 定数・辞書の定義 ---
OPTIONS = {
    "強くそう思う": 7,
    "まぁまぁそう思う": 6,
    "少しそう思う": 5,
    "どちらでもない": 4,
    "少し違うと思う": 3,
    "おおよそ\n違うと思う": 2,
    "全く違うと思う": 1
}

QUESTIONS = [
    "1. 活発、外向的だと思う。",
    "2. 他人に不満をもち、もめごとを起こしやすいと思う。",
    "3. しっかりしていて、自分に厳しいと思う。",
    "4. 心配性で、うろたえやすいと思う。",
    "5. 新しいことが好きで、変わった考えをもつと思う。",
    "6. ひかえめで、おとなしいと思う。",
    "7. 人に気をつかう、やさしい人間だと思う。",
    "8. だらしなく、うっかりしていると思う。",
    "9. 冷静で、気分が安定していると思う。",
    "10. 発想力に欠けた、平凡な人間だと思う。"
]

TRAIT_LABELS = {
    "Extraversion": "外向性",
    "Agreeableness": "協調性",
    "Conscientiousness": "誠実性",
    "Neuroticism": "神経症傾向",
    "Openness": "開放性",
}

TRAIT_TOPIC_HINTS = {
    "Neuroticism": {
        "focus": "予期せぬトラブルや予定変更が起きた時の感情の動き、または緊張や不安の感じ方",
        "question_phrase": "楽しみにしていた予定が急にキャンセルになったりうまくいかないことが重なった時って、気持ちの切り替えはどうやってしてますか？（もしすでにトラブルや切り替えの話を聞いている場合は、「初めての場所や大事な予定がある前って、結構緊張や心配を感じやすい方ですか？」など別の切り口で聞いてください）"
    },
    "Openness": {
        "focus": "新しいジャンルや未知の体験への踏み出し方・好奇心の向け方",
        "question_phrase": "普段、まったく知らないジャンルのものや、やったことない体験に誘われたりした時って、どんな感じで受け止めることが多いですか？（もしすでに挑戦や興味の話を聞いている場合は、「旅行やご飯などで、いつも決まった定番を選ぶことが多いですか？それとも新しいお店を開拓するのが好きですか？」など別の切り口で聞いてください）"
    },
    "Extraversion": {
        "focus": "休日や空き時間の過ごし方・誰かと過ごすことへの感覚",
        "question_phrase": "お休みの日はどんなふうに過ごすことが多いですか？（もしすでに休日の話を聞いている場合は、「一人で静かに集中している時間と、誰かと賑やかに過ごす時間、どちらがリフレッシュしやすいですか？」など別の切り口で聞いてください）"
    },
    "Agreeableness": {
        "focus": "友人や知人と一緒に動く時の意思決定・場のまとめ方",
        "question_phrase": "友達とご飯に行ったりお出かけする時って、お店や行き先ってどうやって決めることが多いですか？（もしすでに決める際の話を聞いている場合は、「グループで意見が分かれた時って、自分の好みを主張する方ですか？それとも周りの希望を優先することが多いですか？」など別の切り口で聞いてください）"
    },
    "Conscientiousness": {
        "focus": "物事への取り組み方・計画の立て方（旅行や日頃の作業など）",
        "question_phrase": "旅行とか遠出する時って、前もってプランをしっかり立てる方ですか？それとも現地についてから決めることの方が多いですか？（もしすでに計画性の話を聞いている場合は、「普段のやらなければいけない課題や予定は、前もって早めに片付けるタイプですか？それとも締め切りギリギリに進めるタイプですか？」など別の切り口で聞いてください）"
    }
}

def calculate_tipi_scores(answers_points):
    """
    10項目の得点（1〜7）からBig Fiveの5つの次元のスコア（平均値）を計算します。
    外向性： 項目１＋（８－項目６）
    協調性： （８－項目２）＋項目７
    勤勉性： 項目３＋（８－項目８）
    神経症傾向：項目４＋（８－項目９）
    開放性： 項目５＋（８－項目 10）
    """
    # ユーザーの回答はリストのインデックス0〜9（項目1〜10）
    p = answers_points
    
    extraversion = (p[0] + (8 - p[5])) / 2
    agreeableness = ((8 - p[1]) + p[6]) / 2
    conscientiousness = (p[2] + (8 - p[7])) / 2
    neuroticism = (p[3] + (8 - p[8])) / 2
    openness = (p[4] + (8 - p[9])) / 2

    return {
        "Extraversion": extraversion,
        "Agreeableness": agreeableness,
        "Conscientiousness": conscientiousness,
        "Neuroticism": neuroticism,
        "Openness": openness
    }

def extract_gemini_response_text(response) -> str:
    """Geminiのレスポンスから安全にテキストを抽出する関数（SDKのPart未取得バグ対策）"""
    try:
        if hasattr(response, "text") and response.text:
            return response.text
    except Exception:
        pass
    
    # response.textが失敗した場合、直接 candidates -> content -> parts から文字列を救出
    if hasattr(response, "candidates") and response.candidates:
        cand = response.candidates[0]
        if hasattr(cand, "content") and hasattr(cand.content, "parts"):
            parts_text = [p.text for p in cand.content.parts if hasattr(p, "text") and p.text]
            if parts_text:
                return "".join(parts_text)
                
    return "すみません、少し考え込んでしまいました。もう一度送っていただけますか？"

# --- セッション状態の初期化 ---
if 'step' not in st.session_state:
    st.session_state.step = 'consent' # 本番フロー（説明・同意から開始）
if 'subject_id' not in st.session_state:
    st.session_state.subject_id = None
if 'pipeline_result' not in st.session_state:
    st.session_state.pipeline_result = None

# Geminiチャット用のセッション状態
if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = []
if 'topic_turn_count' not in st.session_state:
    st.session_state.topic_turn_count = 0
if 'analysis_status' not in st.session_state:
    st.session_state.analysis_status = {
        "observed_traits": [],
        "unobserved_traits": ["Extraversion", "Agreeableness", "Conscientiousness", "Neuroticism", "Openness"],
        "observed_facets": [],
        "fc_count": 0,
        "is_ready_to_finish": False
    }
if 'is_dialogue_finished' not in st.session_state:
    st.session_state.is_dialogue_finished = False
if 'current_topic_trait' not in st.session_state:
    st.session_state.current_topic_trait = None  # 現在掘り下げ中の特性を追跡
if 'asked_topics' not in st.session_state:
    st.session_state.asked_topics = []  # 一度転換先に指定した特性を記録（同じ質問の繰り返し防止）

if 'gemini_chat_session' not in st.session_state:
    # インタビュアーとしてのシステムプロンプトを定義
    system_instruction = """あなたは、相手と楽しく自然に雑談をする対話パートナーです。
親しみやすく丁寧な会話を通じて、相手の日常の行動パターンや物事への接し方を自然に引き出してください。

# 会話の絶対ルール

1. **相手の発言内容をしっかり踏まえて「一歩先」へ展開する（最重要）:**
   相手が答えてくれた内容（行動や体験）を無視して同じ質問を繰り返す（堂々巡り）ことは絶対に禁止です。
   「すでに答えてもらったこと」を受け止め、その時の気持ちや、普段の対人スタンス・物事の進め方へと自然に話を展開してください。

2. **名詞や事実だけの無意味な掘り下げを禁止:**
   店名、料理のメニュー名、場所、スペック、設定確認など、性格に関係のない単なる事実を聞き出すことは禁止です。
   「その時どう感じたか」「普段人と接する時に何を大切にしているか」「物事に取り組むときのこだわり」など、相手の価値観や行動のスタンスを掘り下げてください。

3. **質問は1回の発言に必ず1つだけ（二重質問や勝手な自己完結の厳禁）:**
   「Aでしたか？ところで私は〜」のように、1つの発言の中で質問を投げながら相手の返事を待たずに別の話へ進めたり、質問を2つ同時に投げたりしてはいけません。

4. **口調は「親しみやすい丁寧語（です・ます調）」に統一:**
   タメ口や砕けすぎた口調は禁止です。
   「〜ですよね！」「〜でしたか？」「確かにそれは共感できます」のように、礼儀正しさと親しみやすさを両立した【です・ます調】で短くスッキリ（2〜3文程度）話してください。

5. **性格診断・感情分析のような質問の禁止:**
   以下のような聞き方は絶対に禁止です。
   ❌ 「大切にしていることは何ですか？」「こだわりはありますか？」（直接的な性格・価値観の問い）
   ❌ 「どんなふうにワクワク感を味わいますか？」「その感情をどう処理しますか？」（感情をメタ的に分析させる問い）
   ❌ 「そのとき何を感じましたか？」「気持ちはどうでしたか？」（感情の言語化を強要する問い）
   
   ✅ 代わりに、**具体的な日常の出来事・行動・次の体験**を入口にしてください。
   例：「旅行前って前もって調べる方ですか？」「友達と集まる時どうやって行き先決めてますか？」「そういえば最近それ以外で気になってることってありますか？」

6. **自分から会話を終わらせることの絶対禁止:**
   「ありがとうございました」「それでは終わりにしましょう」「今日はたくさんお話しできました」など、会話を締めくくる・終了を示唆するいかなる発言も絶対にしてはいけません。
   会話の終了はシステム側が決定します。あなたは常に次の質問を1つだけ返し続けてください。【システム指示: 対話終了】が届いた場合のみ、締めくくり発言が許可されます。

7. **過去に尋ねた質問やトピックの重複質問を絶対に禁止:**
   過去の会話履歴を必ず確認し、一度聞いたテーマ（例: 予定のキャンセルや気持ちの切り替え、休日の過ごし方、旅行の計画など）と同じ、または極めて類似した質問をもう一度投げかけることは絶対に禁止です。
   同じ特性に関する話題であっても、必ず別の切り口（別の場面や体験）から質問してください。

# 開始
システムから「開始」とメッセージが送られたら、以下のように挨拶して会話を始めてください：

「こんにちは！これからよろしくお願いします。もし途中で答えにくい質問があったら遠慮なく言ってくださいね、別の話に変えますので！
では早速ですが、最近何かハマっていることや、休日に楽しんでいることって何かありますか？」"""
    
    if gemini_api_key:
        model = genai.GenerativeModel('gemini-3.5-flash-lite', system_instruction=system_instruction)
        st.session_state.gemini_chat_session = model.start_chat(history=[])
    else:
        st.session_state.gemini_chat_session = None

# --- UIコンポーネント ---

if st.session_state.step == 'consent':
    st.title("性格特性に関する研究へのご協力のお願い")
    
    st.markdown("""
    本実験は、対話を通じた性格特性（Big Five）の推定手法や、その分析過程に関する研究の一環として実施しております。
    
    ---
    
    ### 📋 本実験の流れ（所要時間：約10〜15分）
    本実験は、以下の**3つのステップ**で進みます。

    1. **心理テストへの回答（約2分）**  
       簡単な10問の性格診断アンケート（TIPI-J）にお答えいただきます。
    2. **AIとの気軽な雑談・対話（約5〜8分）**  
       AIインタビュアーからの質問に、普段のご自身の様子や休日の過ごし方などを思い浮かべながらリラックスしてお答えください。
    3. **分析結果の確認とアンケート（約3分）**  
       AIによって推定されたあなたの性格特性スコアや、そのように判断した詳しい理由をご確認いただき、簡単な評価アンケートにお答えいただいて終了となります。

    ---

    **【データの取り扱いについて】**
    * 収集したデータは本研究の学術目的のみに使用し、個人を特定する情報（氏名・住所等）は一切収集いたしません。
    * 結果は統計的に処理され、学術発表などで公表される場合がありますが、個人が特定される形での公表は一切ありません。
    * 参加は任意であり、途中で中断することも可能です。その場合でも不利益を被ることはありません。
    
    上記の説明を読み、研究の趣旨を理解した上でご協力いただける場合は、以下のチェックボックスにチェックを入れ、「実験を開始する」ボタンを押してください。
    """)
    
    agree = st.checkbox("上記の説明を理解し、研究への参加に同意します。")
    if st.button("実験を開始する（心理テストへ） ➔", type="primary"):
        if agree:
            st.session_state.step = 'tipi_j'
            st.rerun()
        else:
            st.warning("参加に同意いただける場合はチェックボックスにチェックを入れてください。")

elif st.session_state.step == 'tipi_j':
    st.title("📝 性格診断テスト（TIPI-J）")
    st.markdown("以下の10項目の記述について、あなたご自身の普段の性格や様子にどれくらい当てはまるか、直感でお答えください。")
    
    with st.form("tipi_form"):
        answers = []
        for q in QUESTIONS:
            # 記入不可な選択式（ラジオボタン）を使用し、デフォルトを未選択(index=None)にする
            ans = st.radio(q, list(OPTIONS.keys()), index=None, horizontal=True, key=q)
            answers.append(ans)
            
        submitted = st.form_submit_button("回答を送信して、AIとの対話に進む ➔", type="primary")
        
        if submitted:
            if None in answers:
                st.error("未回答の項目があります。すべての質問に回答してから送信してください。")
            else:
                # 得点に変換
                answers_points = [OPTIONS[ans] for ans in answers]
                
                # 生データ辞書を作成
                raw_data = {QUESTIONS[i]: answers[i] for i in range(10)}
                
                # スコア計算
                scores = calculate_tipi_scores(answers_points)
                st.session_state.last_scores = scores
                
                # データベースに保存
                subject_id = database.save_tipi_results(raw_data, scores)
                st.session_state.subject_id = subject_id
                
                # 中間確認画面をスキップし、直接AIとの対話画面へ遷移！
                st.session_state.step = 'chat'
                st.rerun()

elif st.session_state.step == 'chat':
    st.title("AIとの対話（インタビュー）")
    st.markdown("インタビュアーからの質問に、普段のご自身の様子を思い浮かべながら自由にお答えください。")
    
    status = st.session_state.analysis_status
    observed = status.get("observed_traits", [])
    unobserved = status.get("unobserved_traits", [])
    fc_count = status.get("fc_count", 0)

    # --- テスト用のサイドバー操作パネル ---
    with st.sidebar:
        st.markdown("### 📊 リアルタイム観測状況")
        st.caption("4往復ごと（または情報が揃った場合は早めに）話題を転換しながら解析が更新されます。")
        for t_key, t_label in TRAIT_LABELS.items():
            if t_key in observed:
                st.markdown(f"✅ **{t_label}**: 観測済み")
            else:
                st.markdown(f"⏳ **{t_label}**: 未観測")
        st.write(f"**独立根拠（FC）数**: {fc_count} 件")
        st.write(f"**現在の話題の進捗**: {st.session_state.topic_turn_count} / 4 往復")

        st.divider()
        st.markdown("### 操作メニュー")
        if st.button("対話のみリセット（チャットを最初から）"):
            st.session_state.chat_messages = []
            st.session_state.topic_turn_count = 0
            st.session_state.current_topic_trait = None
            st.session_state.asked_topics = []
            st.session_state.is_dialogue_finished = False
            st.session_state.analysis_status = {
                "observed_traits": [],
                "unobserved_traits": list(TRAIT_LABELS.keys()),
                "observed_facets": [],
                "fc_count": 0,
                "is_ready_to_finish": False
            }
            if 'gemini_chat_session' in st.session_state:
                del st.session_state.gemini_chat_session
            st.rerun()

        if st.button("最初（説明・心理テスト）からやり直す"):
            st.session_state.clear()
            st.session_state.step = 'consent'
            st.rerun()

        st.divider()
        st.markdown("### 手動での対話終了")
        if st.button("✅ 対話を終了して分析する", type="primary",
                     disabled=(len(st.session_state.chat_messages) == 0)):
            # まずAIに締めくくり発話をさせて、チャット画面に表示する
            finish_instruction = (
                "【システム指示: 対話終了】\n"
                "相手のこれまでの話に短く1文で共感し、"
                "「たくさんお話を聞かせていただきありがとうございました！"
                "それではここでインタビューを終了しますね。"
                "画面下の「性格分析結果を確認する」ボタンから分析結果をご確認ください。」"
                "という主旨で、温かく対話を締めくくってください。"
            )
            try:
                fin_resp = st.session_state.gemini_chat_session.send_message(finish_instruction)
                fin_text = extract_gemini_response_text(fin_resp)
                st.session_state.chat_messages.append({"role": "assistant", "content": fin_text})
            except Exception:
                st.session_state.chat_messages.append({
                    "role": "assistant",
                    "content": "たくさんお話を聞かせていただきありがとうございました！それでは分析結果をご確認ください。"
                })
            st.session_state.is_dialogue_finished = True
            st.rerun()
    # ----------------------------------
    
    if not gemini_api_key:
        st.error("エラー: Gemini APIキーが設定されていません。.envファイルを確認してください。")
    else:
        # 最初の挨拶をGeminiに言わせる（まだチャット履歴が空の場合）
        if len(st.session_state.chat_messages) == 0:
            try:
                initial_response = st.session_state.gemini_chat_session.send_message("開始")
                initial_text = extract_gemini_response_text(initial_response)
                st.session_state.chat_messages.append({"role": "assistant", "content": initial_text})
            except Exception as e:
                st.error("現在AIへのアクセスが制限されています（数十秒ほど待ってからページを更新してください）。詳細: " + str(e))
            
        # チャット履歴の表示
        for msg in st.session_state.chat_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                
        # 対話が終了している場合、結果確認ボタンを中央に大きく表示
        if st.session_state.is_dialogue_finished:
            st.success("🎉 十分な情報が集まり、対話が完了しました！")
            if st.button("📊 性格分析結果を確認する ➔", type="primary", use_container_width=True):
                utterances = []
                for i, msg in enumerate(st.session_state.chat_messages):
                    speaker = "system" if msg["role"] == "assistant" else "user"
                    utterances.append({
                        "utterance_id": f"U{i+1:03d}",
                        "speaker": speaker,
                        "text": msg["content"]
                    })
                with st.spinner("最終スコアを計算し、分析結果を作成中です..."):
                    try:
                        result = pipeline.run_cope_pipeline(
                            utterances=utterances,
                            session_id=st.session_state.get("subject_id", "unknown")
                        )
                        st.session_state.pipeline_result = result
                        st.session_state.step = 'result_score_only'
                        st.rerun()
                    except Exception as e:
                        st.error(f"分析中にエラーが発生しました: {e}")
        else:
            # ユーザーの入力欄
            user_input = st.chat_input("ここに回答を入力してください...")
            if user_input:
                # ユーザーの発言を履歴に追加・画面描画
                st.session_state.chat_messages.append({"role": "user", "content": user_input})
                with st.chat_message("user"):
                    st.markdown(user_input)
                
                # --- 話題転換ロジック（2段階制）---
                # ・強制転換（4往復 or 現トピック特性が既に観測済みで2往復以上）
                # ・ソフト誘導（2往復目、まだ未観測なら「タイミングがよければ移行して」と軽くヒントのみ）
                prompt_to_send = user_input
                turn_count = st.session_state.topic_turn_count
                current_trait = st.session_state.current_topic_trait
                observed_now = st.session_state.analysis_status.get("observed_traits", [])
                unobserved_now = st.session_state.analysis_status.get("unobserved_traits", [])
                priority_order = ["Neuroticism", "Openness", "Extraversion", "Agreeableness", "Conscientiousness"]
                asked_topics = st.session_state.asked_topics  # 一度転換済みの話題リスト

                # 転換先の特性を決める（未観測 & まだ質問していない中で優先度が高いもの）
                # まず「未観測 かつ 未質問」のものを探す。なければ「未質問」だけで探す
                next_target = next(
                    (t for t in priority_order if t in unobserved_now and t not in asked_topics), None
                )
                if not next_target:
                    # 全部聞いた or 全部観測済み → 未観測から再挑戦
                    next_target = next((t for t in priority_order if t in unobserved_now), None)
                if not next_target:
                    next_target = unobserved_now[0] if unobserved_now else priority_order[0]

                # 強制転換すべき条件
                force_switch = (
                    turn_count >= 4  # 4往復超えたら強制
                    or (turn_count >= 2 and current_trait is not None and current_trait in observed_now)
                    # ↑現トピックの特性データが取れていたら2往復で早期転換OK
                )
                # ソフト誘導すべき条件（強制ではないが次の話題を意識させる）
                soft_nudge = (turn_count == 2 and not force_switch and next_target is not None)

                did_force_switch = False  # このターンで強制転換したかのフラグ（終了判定スキップ用）

                if force_switch:
                    hint_data = TRAIT_TOPIC_HINTS.get(next_target, TRAIT_TOPIC_HINTS["Neuroticism"])
                    focus_theme = hint_data["focus"]
                    q_phrase = hint_data["question_phrase"]
                    prompt_to_send = (
                        f"【相手の発言】: {user_input}\n\n"
                        f"【システム指示: 話題の転換】\n"
                        f"相手の発言に短く1文で共感・相槌を打ってください（直前の話題への質問は絶対に禁止です）。\n"
                        f"その上で、ぎこちない自分語りのエピソードは挟まず、「話題は少し変わるのですが〜」「ところで〜」と自然に切り出して、以下の新しいテーマについて相手に質問してください。\n"
                        f"・聞きたいテーマ: 「{focus_theme}」\n"
                        f"・質問のイメージ: 「{q_phrase}」\n"
                        f"★最重要注意: 過去の対話履歴で既に似たような質問（例: 予定のキャンセルや気持ちの切り替えなど）をしている場合は、同じ質問を絶対に繰り返さず、必ず括弧内に記載された別の切り口や別の場面で質問してください。\n"
                        f"※質問は必ずこの新しい話題の1つだけにしてください。全体で2〜3文のシンプルで自然な会話にしてください。"
                    )
                    st.session_state.topic_turn_count = 0
                    st.session_state.current_topic_trait = next_target
                    if next_target not in st.session_state.asked_topics:
                        st.session_state.asked_topics.append(next_target)
                    did_force_switch = True  # 強制転換フラグON
                elif soft_nudge:
                    hint_data = TRAIT_TOPIC_HINTS.get(next_target, TRAIT_TOPIC_HINTS["Neuroticism"])
                    focus_theme = hint_data["focus"]
                    prompt_to_send = (
                        f"{user_input}\n\n"
                        f"【システム参考情報（強制ではありません）】\n"
                        f"現在の会話から得られる情報が十分であれば、「{focus_theme}」に関する話題へ"
                        f"自然なタイミングで移行していただくと良いです。\n"
                        f"ただし、相手がまだ話している途中や、今の話題をもう少し掘り下げられそうな場合は"
                        f"引き続き今の話題を続けてください。移行する場合も唐突にならないよう自然に繋げてください。"
                    )
                    st.session_state.topic_turn_count += 1
                else:
                    st.session_state.topic_turn_count += 1
                
                # Geminiから返答を取得（1回の入力に対して必ず1回だけ返答！）
                with st.chat_message("assistant"):
                    with st.spinner("入力中..."):
                        try:
                            response = st.session_state.gemini_chat_session.send_message(prompt_to_send)
                            response_text = extract_gemini_response_text(response)
                            st.markdown(response_text)
                            st.session_state.chat_messages.append({"role": "assistant", "content": response_text})
                            
                            with open("chat_log_temp.txt", "w", encoding="utf-8") as f:
                                for m in st.session_state.chat_messages:
                                    role_name = "AI" if m["role"] == "assistant" else "あなた"
                                    f.write(f"[{role_name}]\n{m['content']}\n\n")
                                    
                        except Exception as e:
                            st.error("エラーが発生しました。数十秒待ってから再度送信してください。詳細: " + str(e))
                            st.stop()
                
                # 裏で中間解析を実行（次の話題転換や終了判定のための情報更新）
                # ★ force_switch直後は新しい質問への回答がまだないため終了判定をスキップする
                utterances = []
                for i, msg in enumerate(st.session_state.chat_messages):
                    speaker = "system" if msg["role"] == "assistant" else "user"
                    utterances.append({
                        "utterance_id": f"U{i+1:03d}",
                        "speaker": speaker,
                        "text": msg["content"]
                    })
                
                user_turns = len([u for u in utterances if u["speaker"] == "user"])
                if user_turns >= 4 and not did_force_switch:
                    # force_switchしたターンは解析・終了判定しない（まだ回答を受け取っていないため）
                    analysis_res = pipeline.run_intermediate_analysis(utterances)
                    st.session_state.analysis_status = analysis_res
                    
                    # 終了条件: 全5特性が満遍なく観測され、かつ十分な根拠（FC >= 7）が集まるまで終了させない
                    # （ただし長引いた場合のセーフティとしてユーザー発話12往復で終了）
                    is_ready = analysis_res.get("is_ready_to_finish", False) or (user_turns >= 12)
                    if is_ready:
                        # 終了条件達成！
                        finish_instruction = (
                            "【システム指示: 対話終了】\n"
                            "十分な性格判断情報が集まりました。相手の発言に優しく共感した上で、"
                            "「たくさんお話を聞かせていただきありがとうございました！あなたの普段の様子や考え方がよく分かりました。"
                            "それではここでインタビューを終了し、分析結果を作成しますね。画面下のボタンから分析結果をご確認ください。」"
                            "という主旨で、温かく対話を締めくくってください。"
                        )
                        fin_resp = st.session_state.gemini_chat_session.send_message(finish_instruction)
                        fin_text = extract_gemini_response_text(fin_resp)
                        st.session_state.chat_messages.append({"role": "assistant", "content": fin_text})
                        st.session_state.is_dialogue_finished = True
                        st.rerun()

elif st.session_state.step == 'result_score_only':
    st.title("📊 性格分析結果（スコアの提示）")

    res = st.session_state.pipeline_result
    if res is None:
        st.error("結果データがありません。対話画面から再度お試しください。")
        st.stop()

    TRAIT_LABELS = {
        "Extraversion": "外向性",
        "Agreeableness": "協調性",
        "Conscientiousness": "誠実性",
        "Neuroticism": "神経症傾向",
        "Openness": "開放性",
    }

    scores = res.get("step6_trait_scores", {})

    st.markdown("AIとの対話から、あなたのBig Five性格特性スコアが以下のように推定されました。")
    st.info("※詳しい分析理由や根拠を見る前に、まずは**このスコアの数値のみを見た印象**について、下部のアンケートにお答えください。")

    # --- スコア表示（横5列） ---
    st.subheader("【推定スコア（1〜7点）】")
    cols = st.columns(5)
    for i, (trait_key, label) in enumerate(TRAIT_LABELS.items()):
        score = scores.get(trait_key, "-")
        cols[i].metric(label, f"{score} / 7点" if isinstance(score, (int, float)) else score)

    st.divider()

    # --- 事前アンケート（スコアのみ） ---
    st.subheader("📝 評価アンケート（第1段階）")
    st.write("上記のスコア結果（数値のみ）を見て、あなた自身の感覚にどれくらい近いか教えてください。")
    
    q1 = st.radio(
        "Q1. 推定された各スコア（点数）は、あなた自身の普段の実感と合っていましたか？",
        [1, 2, 3, 4, 5],
        index=2,
        horizontal=True,
        key="pre_q1"
    )
    st.caption("1: 全く合っていない ↔ 5: 非常に合っている")
    
    q2 = st.radio(
        "Q2. スコア（点数の数値）だけを見て、どれくらい納得感がありますか？",
        [1, 2, 3, 4, 5],
        index=2,
        horizontal=True,
        key="pre_q2"
    )
    st.caption("1: 全く納得できない ↔ 5: 非常に納得できる")

    if st.button("回答を送信して、詳しい根拠を見る ➔", type="primary"):
        st.session_state.survey_pre = {"q1": q1, "q2": q2}
        st.session_state.step = 'result_with_explanation'
        st.rerun()

elif st.session_state.step == 'result_with_explanation':
    st.title("📊 性格分析結果（詳細な根拠・説明）")

    res = st.session_state.pipeline_result
    if res is None:
        st.error("結果データがありません。対話画面から再度お試しください。")
        st.stop()

    TRAIT_LABELS = {
        "Extraversion": "外向性",
        "Agreeableness": "協調性",
        "Conscientiousness": "誠実性",
        "Neuroticism": "神経症傾向",
        "Openness": "開放性",
    }

    scores = res.get("step6_trait_scores", {})
    details = res.get("step6_details", {})

    # --- スコア再掲 ---
    st.subheader("【推定スコア（1〜7点）】")
    cols = st.columns(5)
    for i, (trait_key, label) in enumerate(TRAIT_LABELS.items()):
        score = scores.get(trait_key, "-")
        cols[i].metric(label, f"{score} / 7点" if isinstance(score, (int, float)) else score)

    st.divider()

    # --- 詳細根拠（アコーディオン） ---
    st.subheader("🔍 なぜこの結果になったのか（分析プロセスの説明）")
    st.markdown("対話の中で観察されたあなたの具体的な発言や行動、および判断の理由です。")

    # 説明文
    explanation = res.get("step7_explanation", "")
    if explanation:
        st.markdown("### 📝 全体のまとめ")
        st.markdown(
            f'<div style="background-color:#f4f6f9;color:#1a1a1a;padding:18px;'
            f'border-radius:5px;border-left:5px solid #0066cc;font-size:1.1em;'
            f'line-height:1.6;">{explanation}</div>',
            unsafe_allow_html=True
        )
        st.write("")
        st.divider()

    # 各特性のタブ
    st.markdown("### 📋 各特性の詳細な根拠")
    tab_keys = list(TRAIT_LABELS.keys())
    tabs = st.tabs([TRAIT_LABELS[k] for k in tab_keys])

    for tab, trait_key in zip(tabs, tab_keys):
        with tab:
            detail = details.get(trait_key, {})
            facets = detail.get("facets", {})
            if not facets:
                st.write("対話ログから、この特性に関連する具体的な行動や発言は観察されませんでした。")
            else:
                for facet_name, facet_data in facets.items():
                    st.markdown(f"**Facet（下位因子）: {facet_name}**")
                    for cluster in facet_data.get("clusters", []):
                        ev = cluster.get("canonical_evidence", "")
                        reason = cluster.get("reason", "")
                        strength = cluster.get("strength", "")
                        confidence = cluster.get("confidence", "")
                        if ev:
                            st.write(f"- 根拠となった発言・行動: 「{ev}」")
                        if reason:
                            st.write(f"  → 判断の理由: {reason}")
                        if strength or confidence:
                            st.write(f"  （強度: {strength} / 確信度: {confidence}）")

    st.divider()

    # --- 各5特性の個別評価（1〜5段階 ＋ 理由選択） ---
    st.subheader("📝 各性格特性の個別評価と理由（第2段階）")
    st.write("上記の詳しい根拠や説明を踏まえ、5つの特性それぞれについて、推定結果への納得感とそう感じた理由を教えてください。")

    REASON_OPTIONS = [
        "普段の自分の性格や行動パターンとよく合っている",
        "対話で話したエピソードや理由が、的確にスコアに反映されている",
        "どちらかといえば思い当たる節がある",
        "普段の自分よりも高め（または低め）に判定されている気がする",
        "特定の場面や話題だけの話だったので、普段の自分とは少しズレがある",
        "今回の対話では、この性格に関する話をあまり話せなかった"
    ]

    trait_evaluations = {}

    for trait_key, trait_label in TRAIT_LABELS.items():
        t_score = scores.get(trait_key, "-")
        st.markdown(f"#### ■ {trait_label}（推定スコア: {t_score} / 7点）")
        col_eval1, col_eval2 = st.columns([1, 2])
        with col_eval1:
            rating = st.radio(
                f"{trait_label}の納得感",
                [1, 2, 3, 4, 5],
                index=2,
                horizontal=True,
                key=f"rating_{trait_key}",
                help="1: 全く納得できない 〜 5: 非常に納得できる"
            )
            st.caption("1: 納得できない ↔ 5: 納得できる")
        with col_eval2:
            reason = st.selectbox(
                f"そう感じた理由を選択してください",
                REASON_OPTIONS,
                key=f"reason_{trait_key}"
            )
        trait_evaluations[trait_key] = {"rating": rating, "reason": reason}
        st.write("")

    st.divider()

    # --- システム全体への評価（1問のみ・5段階評価） ---
    st.subheader("📝 推定結果全体への納得感評価")
    st.write("詳しい根拠や説明を読んだ上で、推定結果全体についてお答えください。")
    q3 = st.radio(
        "Q. 詳しい根拠や判断理由の説明を読んだ上で、この推定結果全体にどれくらい納得感がありますか？",
        [1, 2, 3, 4, 5],
        index=2,
        horizontal=True,
        key="post_q3",
        help="1: 全く納得できない 〜 5: 非常に納得できる"
    )
    st.caption("1: 全く納得できない ↔ 5: 非常に納得できる")

    if st.button("最終評価を送信して実験を終了する", type="primary"):
        # データベース（CSV）にすべての結果を一括保存！
        subject_id = st.session_state.get("subject_id", "unknown")
        survey_pre = st.session_state.get("survey_pre", {"q1": 3, "q2": 3})
        survey_post = {"q3": q3}
        
        try:
            database.save_experiment_results(
                subject_id=subject_id,
                estimated_scores=scores,
                survey_pre=survey_pre,
                survey_post=survey_post,
                trait_evaluations=trait_evaluations
            )
            st.session_state.step = 'all_completed'
            st.rerun()
        except Exception as e:
            st.error(f"データの保存中にエラーが発生しました: {e}")
            st.session_state.step = 'all_completed'
            st.rerun()
        except Exception as e:
            st.error(f"データの保存中にエラーが発生しました: {e}")

elif st.session_state.step == 'all_completed':
    st.title("🎉 実験終了")
    st.success("すべてのアンケートへのご回答ありがとうございました！")
    st.markdown(f"**参加者ID**: `{st.session_state.get('subject_id', 'unknown')}`")
    st.markdown("""
    あなたの回答データおよび分析結果は正常に記録・保存されました。
    ご協力誠にありがとうございました！
    この画面を閉じて終了してください。
    """)
