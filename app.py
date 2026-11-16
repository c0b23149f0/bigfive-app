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
    "Extraversion": {
        "anecdote": "休日に家でずっと一人で過ごしていると、ふと誰かと無性に話したくなって友達に連絡しちゃったこと",
        "question_phrase": "普段のお休みは、家で一人でゆっくり過ごす時間が多いですか？誰かと会って話す時間が多いですか？"
    },
    "Agreeableness": {
        "anecdote": "友達とご飯に行く時に、お互い遠慮し合って『何でもいいよ』って言ってたら全然お店が決まらなかったこと",
        "question_phrase": "普段友達と出かけたり遊んだりする時って、行き先やお店はどうやって決めることが多いですか？"
    },
    "Conscientiousness": {
        "anecdote": "旅行やお出かけの準備を前日までサボってたら、当日忘れ物だらけで大慌てしたこと",
        "question_phrase": "お出かけや旅行の準備とか計画って、普段どんな感じで進めることが多いですか？"
    },
    "Neuroticism": {
        "anecdote": "楽しみにしていた予定が当日の雨で急に中止になっちゃって、ショックで一日テンションが落ちちゃったこと",
        "question_phrase": "楽しみにしていた予定が急にダメになっちゃったり思い通りにいかない時って、どんな感じになりますか？"
    },
    "Openness": {
        "anecdote": "カフェやご飯屋さんに行くと、冒険できなくて結局いつも同じお気に入りのメニューばっかり頼んじゃうこと",
        "question_phrase": "見たことない限定メニューや珍しい新商品があったら、気になって試してみたくなったりしますか？"
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

if 'gemini_chat_session' not in st.session_state:
    # インタビュアーとしてのシステムプロンプトを定義
    system_instruction = """あなたは、相手と楽しく自然に雑談をする対話パートナーです。
親しみやすく丁寧な会話を通じて、相手の日常の具体的なエピソードや体験談を引き出してください。

# 会話の絶対ルール

1. **相手の話したテーマをしっかり受け止めて深掘りする（最重要）:**
   相手が出してくれた話題（ゲーム、趣味、仕事、旅行など）に対して、すぐに「〜以外だと何してますか？」のように話題を切り捨てることは絶対に禁止です。
   相手の言葉に純粋な興味を持ち、まずは「具体的にどんなことをしているのか」「最近で特に印象に残っている具体的な出来事や場面はどんな時だったか」を丁寧に掘り下げてください。
   ※自分から勝手に別の話題に切り替えてはいけません。1つのテーマについて具体的な体験談が出るまでじっくり会話を続けてください。

2. **口調は「親しみやすい丁寧語（です・ます調）」に統一:**
   タメ口やくだけすぎた口調（「〜だよね」「〜したんじゃない？」「（笑）」など）は禁止です。
   「〜ですよね！」「〜でしたか？」「確かにそれは熱くなりますね」のように、礼儀正しさと親しみやすさを両立した【です・ます調】で話してください。

3. **2択質問や「〜なタイプですか？」の絶対禁止:**
   「A派ですか？B派ですか？」「〜なタイプですか？」と白黒の選択を迫る質問は心理テストのようになるため禁止です。相手自身の言葉で自由に答えられるオープンな聞き方をしてください。

4. **「最近の具体的な出来事・エピソード」を聞く:**
   「最近やってて、一番盛り上がった瞬間や焦った場面ってどんな時でしたか？」「その時は具体的にどんな対応をしたんですか？」のように、実際の体験談や行動を聞いてください。

5. **返答は短くスッキリさせる（長文の語りや先回りの禁止）:**
   軽い相槌や共感をサラッと1〜2行で伝え、すぐに相手に話を振るようにして、会話のキャッチボールのテンポを良くしてください。

6. **オウム返しの厳禁:**
   相手の言葉を「〇〇なんですね！」と機械的に復唱するのを禁止します。

7. **無味乾燥なスペック・環境確認の禁止:**
   「機種は何ですか？」「何時間やりますか？」といった設定確認は禁止です。

8. **不自然なカタカナ・分析用語の禁止:**
   「クールダウン」「ルーティン」「モチベーション」といった横文字や、面接・心理分析を想起させる言葉は使わないでください。

9. **質問は1回に1つだけ:**
   相手が気軽に答えられるよう、1回の発言に質問は必ず1つだけにしてください。

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
       AIインタビュアーからの質問に、普段のご自身の様子や休日の過ごし方などを思い浮かべながらリラックスしてお答えください（堅苦しい面接ではなく、自然な日常会話です）。
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
        st.caption("3往復ごとに裏でCoPE正規パイプライン（ステップ2〜5）を実行して更新されます。")
        for t_key, t_label in TRAIT_LABELS.items():
            if t_key in observed:
                st.markdown(f"✅ **{t_label}**: 観測済み")
            else:
                st.markdown(f"⏳ **{t_label}**: 未観測")
        st.write(f"**独立根拠（FC）数**: {fc_count} 件")
        st.write(f"**現在の話題の進捗**: {st.session_state.topic_turn_count} / 3 往復")

        st.divider()
        st.markdown("### 操作メニュー")
        if st.button("対話のみリセット（チャットを最初から）"):
            st.session_state.chat_messages = []
            st.session_state.topic_turn_count = 0
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
            utterances = []
            for i, msg in enumerate(st.session_state.chat_messages):
                speaker = "system" if msg["role"] == "assistant" else "user"
                utterances.append({
                    "utterance_id": f"U{i+1:03d}",
                    "speaker": speaker,
                    "text": msg["content"]
                })

            with st.spinner("分析中です。しばらくお待ちください...（数分かかる場合があります）"):
                try:
                    result = pipeline.run_cope_pipeline(
                        utterances=utterances,
                        session_id=st.session_state.get("subject_id", "unknown")
                    )
                    st.session_state.pipeline_result = result
                    st.session_state.step = 'result_score_only'
                    st.rerun()
                except Exception as e:
                    st.error(f"分析中にエラーが発生しました。詳細: {e}")
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
                st.session_state.chat_messages.append({"role": "user", "content": user_input})
                st.session_state.topic_turn_count += 1
                with st.chat_message("user"):
                    st.markdown(user_input)
                    
                # Geminiから通常の返答を取得
                with st.chat_message("assistant"):
                    with st.spinner("入力中..."):
                        try:
                            response = st.session_state.gemini_chat_session.send_message(user_input)
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
                
                # 3往復（またはそれ以上）に達した場合、裏で中間解析を実行（案B）
                if st.session_state.topic_turn_count >= 3:
                    with st.spinner("対話ログの分析状況を確認しています..."):
                        utterances = []
                        for i, msg in enumerate(st.session_state.chat_messages):
                            speaker = "system" if msg["role"] == "assistant" else "user"
                            utterances.append({
                                "utterance_id": f"U{i+1:03d}",
                                "speaker": speaker,
                                "text": msg["content"]
                            })
                        
                        # ステップ2〜5の正規パイプラインを高速実行
                        analysis_res = pipeline.run_intermediate_analysis(utterances)
                        st.session_state.analysis_status = analysis_res
                        
                        user_turns = len([u for u in utterances if u["speaker"] == "user"])
                        is_ready = analysis_res.get("is_ready_to_finish", False) or (user_turns >= 8)
                        
                        if is_ready:
                            # 終了条件達成！締めくくりの発話を生成
                            finish_instruction = (
                                "【システム指示: 対話終了】\n"
                                "十分な性格判断情報が集まりました。相手の先ほどの発言に優しく共感・相槌を打った上で、"
                                "「たくさんお話を聞かせていただきありがとうございました！あなたの普段の様子や考え方がよく分かりました。"
                                "それではここでインタビューを終了し、分析結果を作成しますね。画面下のボタンから分析結果をご確認ください。」"
                                "という主旨で、温かく自然に対話を締めくくってください。"
                            )
                            with st.chat_message("assistant"):
                                with st.spinner("対話をまとめています..."):
                                    try:
                                        fin_resp = st.session_state.gemini_chat_session.send_message(finish_instruction)
                                        fin_text = extract_gemini_response_text(fin_resp)
                                        st.markdown(fin_text)
                                        st.session_state.chat_messages.append({"role": "assistant", "content": fin_text})
                                        st.session_state.is_dialogue_finished = True
                                        st.rerun()
                                    except Exception:
                                        st.session_state.is_dialogue_finished = True
                                        st.rerun()
                        else:
                            # まだ未観測の特性がある場合、自然に話題を転換！
                            unobserved = analysis_res.get("unobserved_traits", [])
                            target_trait = unobserved[0] if unobserved else "Openness"
                            target_label = TRAIT_LABELS.get(target_trait, "新しい体験")
                            
                            hint_data = TRAIT_TOPIC_HINTS.get(target_trait, TRAIT_TOPIC_HINTS["Openness"])
                            anecdote = hint_data["anecdote"]
                            q_phrase = hint_data["question_phrase"]
                            
                            switch_instruction = (
                                f"【システム指示: 人間味のある自然な話題転換】\n"
                                f"現在のテーマの深掘りは十分です。相手の先ほどの発言に短く共感・リアクションをした上で、\n"
                                f"あなた自身のちょっとした日常エピソード（前フリ）を語り、相手にオープンな質問を振って自然に別の話題へ切り替えてください。\n\n"
                                f"【前フリの日常エピソード案】: 「{anecdote}」\n"
                                f"【相手への尋ね方のイメージ】: 「{q_phrase}」\n\n"
                                f"【絶対の禁止事項】\n"
                                f"・「Aですか？それともBですか？」のような2択の選択肢を押し付ける質問は禁止です（尋問・テストに聞こえるため）。\n"
                                f"・「トラブルの対処法は？」といった面接のような重い質問も禁止です。\n"
                                f"・友達とカフェでおしゃべりしているような、親しみやすく丁寧なトーンで自然に話をつなげてください。"
                            )
                            with st.chat_message("assistant"):
                                with st.spinner("次の話題へ展開中..."):
                                    try:
                                        sw_resp = st.session_state.gemini_chat_session.send_message(switch_instruction)
                                        sw_text = extract_gemini_response_text(sw_resp)
                                        st.markdown(sw_text)
                                        st.session_state.chat_messages.append({"role": "assistant", "content": sw_text})
                                        st.session_state.topic_turn_count = 0
                                        st.rerun()
                                    except Exception as e:
                                        st.session_state.topic_turn_count = 0
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
