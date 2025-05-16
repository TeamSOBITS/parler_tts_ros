from parler_tts import ParlerTTSForConditionalGeneration

# メインのダウンロード処理関数
def main():
    model_list = ["parler-tts/parler-tts-mini-v1"]

    #英語モデル
        #ミニ           ：parler-tts/parler-tts-mini-v1
        #ミニジェニー   ：parler-tts/parler-mini-v1-jenny
        #感情指定可能   ：parler-tts/parler-tts-mini-expresso
        #CPU向けジェニー：parler-tts/parler-tiny-v1-jenny
    #日本語モデル
        #ミニ           ：2121-8/japanese-parler-tts-mini

    # 設定された各言語とモデルについてダウンロードを実行
    for molel in model_list:
        print(f"\n--- {molel}ダウンロード開始:  ---")
        try:
            ParlerTTSForConditionalGeneration.from_pretrained(molel)
            print(f"モデル '{molel}' のダウンロード完了")
        except Exception as e:
            print(f"エラー: モデル '{molel}' のダウンロード中にエラーが発生: {e}")

    print("\n--- 全モデルのダウンロード完了 ---")

if __name__ == '__main__':
    main()




























































#隠し要素：2121-8/japanese-parler-tts-mini-bate