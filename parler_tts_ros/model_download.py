from parler_tts import ParlerTTSForConditionalGeneration

# メインのダウンロード処理関数
def main():
    model_list = ["parler-tts/parler-mini-v1"]

    #日本語モデル：2121-8/japanese-parler-tts-mini

    #英語モデル
        #ミニ        ：parler-tts/parler-mini-v1
        #ミニジェニー：parler-tts/parler-mini-v1-jenny

        #感情　　　　：parler-tts/parler-tts-mini-expresso
    
        #CPU向け：parler-tts/parler-tts-tiny-v1
        #CPU向けジェニー：parler-tts/parler-tiny-v1-jenny

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