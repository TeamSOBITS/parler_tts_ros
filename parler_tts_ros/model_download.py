import torch  # デバイス確認のために使用
from parler_tts import ParlerTTSForConditionalGeneration  # ParlerTTSモデルクラス
from transformers import AutoTokenizer  # トークナイザークラス
import time # 時間計測に使用（オプション）

# ダウンロード対象のアセット設定
# 言語ごとに、ダウンロードが必要なモデルとトークナイザーの情報を定義します。
# name: Hugging Face Hub上のリポジトリ名
# subfolder: トークナイザーなどがモデルと同じリポジトリ内のサブフォルダにある場合に指定
DOWNLOAD_CONFIG = {
    "en": {
        "description": "英語モデルとトークナイザー",
        "model_name": "parler-tts/parler-tts-mini-v1",
        "tokenizers": [
            # 英語はモデルと同じリポジトリの共通トークナイザーを使用
            {"name": "parler-tts/parler-tts-mini-v1", "subfolder": None}
        ]
    },
    "ja": {
        "description": "日本語モデルとトークナイザー",
        "model_name": "2121-8/japanese-parler-tts-mini",
        "tokenizers": [
            # 日本語モデルはプロンプト用とdescription用でトークナイザーが分かれています
            {"name": "2121-8/japanese-parler-tts-mini", "subfolder": "prompt_tokenizer"},
            {"name": "2121-8/japanese-parler-tts-mini", "subfolder": "description_tokenizer"}
        ]
    }
    # 必要に応じて他の言語やモデルを追加できます
    # "another_lang": {
    #     "description": "別の言語モデル",
    #     "model_name": "another/model-name",
    #     "tokenizers": [
    #         {"name": "another/tokenizer-name", "subfolder": None}
    #     ]
    # }
}

# メインのダウンロード処理関数
def main():
    print("--- ParlerTTS アセット事前ダウンロードスクリプト ---")
    print("初回実行時はインターネット接続が必要です。")
    print("指定されたモデルとトークナイザーをHugging Face Hubからダウンロードし、ローカルキャッシュに保存します。")
    print("ローカルに既に存在する場合はスキップされ、その際はHugging Faceライブラリから 'Using cached model...' や 'Using cached tokenizer...' のようなメッセージが出力されます。")
    print("-" * 40) # 区切り線

    # 使用可能なデバイスを判定します（ダウンロード自体はCPUでも可能です）
    # from_pretrained の呼び出しにdevice引数を含める場合があるので、判定しておきますが、
    # ダウンロードの処理自体はデバイスに大きく依存しません。
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"デバイスの確認: {device} が利用可能です（ダウンロード処理には直接影響しません）")

    # 設定された各言語とモデルについてダウンロードを実行します
    for lang, config in DOWNLOAD_CONFIG.items():
        description = config["description"]
        model_name = config["model_name"]
        tokenizers_config = config["tokenizers"]

        print(f"\n--- ダウンロード開始: {description} ({lang}) ---")

        # モデルのダウンロード
        print(f"モデル '{model_name}' をダウンロード中...")
        start_time = time.time()
        try:
            # ParlerTTSForConditionalGeneration.from_pretrainedを呼び出すことで、
            # モデルファイルと関連コンフィグがローカルキャッシュにダウンロードされます。
            # ローカルに存在する場合はダウンロードはスキップされ、キャッシュが使用されます。
            ParlerTTSForConditionalGeneration.from_pretrained(model_name)
            end_time = time.time()
            print(f"モデル '{model_name}' のダウンロード試行が完了しました。 ({end_time - start_time:.2f}秒)")
        except Exception as e:
            print(f"エラー: モデル '{model_name}' のダウンロード中にエラーが発生しました: {e}")

        # トークナイザーのダウンロード
        for tokenizer_info in tokenizers_config:
            tokenizer_name = tokenizer_info["name"]
            subfolder = tokenizer_info["subfolder"]
            # 表示用の名前を作成 (サブフォルダがあれば表示)
            display_name = f"{tokenizer_name}" + (f" (サブフォルダ: {subfolder})" if subfolder else "")

            print(f"トークナイザー '{display_name}' をダウンロード中...")
            start_time = time.time()
            try:
                # AutoTokenizer.from_pretrainedを呼び出すことで、
                # トークナイザーファイルがローカルキャッシュにダウンロードされます。
                # ローカルに存在する場合はダウンロードはスキップされ、キャッシュが使用されます。
                AutoTokenizer.from_pretrained(tokenizer_name, subfolder=subfolder)
                end_time = time.time()
                print(f"トークナイザー '{display_name}' のダウンロード試行が完了しました。 ({end_time - start_time:.2f}秒)")
            except Exception as e:
                 print(f"エラー: トークナイザー '{display_name}' のダウンロード中にエラーが発生しました: {e}")

    print("\n--- 全てのアセットのダウンロード試行が完了しました ---")

if __name__ == '__main__':
    main()