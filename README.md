<a name="readme-top"></a>

> [!WARNING]
> 本リポジトリはサポートされて間もないため，今後も頻繁に大きく改良される可能性があります．

# Parler_tts_ros

<!-- レポジトリの概要 -->
## 概要
Parler_TTSは，特定のスピーカーのスタイル(性別，ピッチ，話し方など)で高品質で自然な音声を生成できる軽量のテキスト読み上げ(TTS)モデルです．

Stability AIとエジンバラ大学のDan LythとSimon Kingによる論文[Natural language guidance of high-fidelity text-to-speech with synthetic annotations](https://www.text-description-to-speech.com) からの複製されました．

本リポジトリはParler_TTSをROS2のAction通信で使用できるようにしたもので，英語と日本語での発話に対応しています．

<!-- セットアップ -->
## セットアップ

ここで，本レポジトリのセットアップ方法について説明します．
### 環境条件

まず，以下の環境を整えてから，次のインストール段階に進んでください．
| System  | Version |
| --- | --- |
| Ubuntu | 22.04 (Jammy Jellyfish) |
| ROS    | Humble Hawksbill |
| Python | 3.10 |

### インストール方法
1. ROS2の`src`フォルダに移動します．
    ```sh
    cd ~/colcon_ws/src/
    ```

2. 本レポジトリをcloneします．
    ```sh
    git clone https://github.com/TeamSOBITS/parler_tts_ros.git
    ```
3. レポジトリの中へ移動します．
    ```sh
    cd parler_tts_ros/
    ```
4. 依存パッケージをインストールします．時間がかかるので注意．
    ```sh
    bash install.sh
    ```
 > [!WARNING]
 > すでにtorchライブラリがインストールされていた場合、このコマンドによってバージョンが変わってしまう可能性があるので注意してください．
5. パッケージをコンパイルします．
    ```sh
    cd ~/colcon_ws/
    colcon build --symlink-install
    source ~/colcon_ws/install/setup.sh
    ```
6. 言語モデルをダウンロードします．時間がかかるので注意してください．
    ```sh
    cd ~/colcon_ws/src/parler_tts_ros/
    ```
    ```sh
    ros2 run parler_tts_ros model_download
    ```

<!-- 実行・操作方法 -->
## 実行・操作方法
1. アクションサーバーを起動します．**Ready to ParlerTTS in**と表示されるまでgoalを送らずに待機してください．(時間がかかるので注意)
   ```sh
   ros2 launch parler_tts_ros parler_tts_server.launch.py 
   ```
2. アクションクライアントを起動し，発話させたい文字を送信します．


<!-- 各種パラメータ -->
## 各種パラメータ

### 言語の切り替えについて
parler_tts_server.launch.pyにある**language**の項目で，
- 英語の場合　：en
- 日本語の場合：ja

として設定します．(デフォルト値は英語)


### 話者や話し方の設定について
parler_tts_server.launch.pyにある**description**の項目で，単純な説明文で話者や話し方を設定できます．

話者について，特定の話者(推奨)か，毎回ランダムな話者にするかを選ぶことができます．

### ランダムな話者
**description**の項目で、話者を**male**か**female**と指定してください．
- 例
    ``` bash
    default_value="A female speaker delivers a slightly expressive and animated speech with a moderate speed and pitch. The recording is of very high quality, with the speaker's voice sounding clear and very close up."
    ```

### 特定の話者
特定の話者を使うことで、毎回同じ話者で発話させることができます．
**description**の項目で、「Alisa」を以下のリストの中から置き換えてください．

- 例
    ``` bash
    default_value='Alisa.fast speed. Expression is rich. The speaking voice is noisy.',
    ```

> [!WARNING]
> 日本語の話者は「JSUT」のみ利用できます．
<details>
<summary>利用可能な話者の全リスト(英語)</summary>

- Laura
- Gary
- Jon
- Lea
- Karen
- Rick
- Brenda
- David
- Eileen
- Jordan
- Mike
- Yann
- Joy
- James
- Eric
- Lauren
- Rose
- Will
- Jason
- Aaron
- Naomie
- Alisa
- Patrick
- Jerry
- Tina
- Jenna
- Bill
- Tom
- Carol
- Barbara
- Rebecca
- Anna
- Bruce
- Emily
</details>

### 話し方について
次のような単純なテキストプロンプトで話し方を変更することができます．
- 例1

    ``` bash
    default_value='Will delivers a slightly expressive and animated speech with a moderate speed and pitch. The recording is of  high quality, with the speaker voice sounding clear.',
    ```
- 例2
    ``` bash
    default_value='Alisa.fast speed. Expression is rich. The speaking voice is noisy.',
    ```

- 最高品質のオーディオを生成するには「very clear audio」という用語を含め、高レベルのバックグラウンドノイズには「very noisy audio」という用語を含めます
- 句読点は、世代の韻律を制御するために使用できます(たとえば、カンマを使用して音声に小さな区切りを追加します)
- 残りの音声機能(性別、発話速度、ピッチ、残響)は、プロンプトから直接制御できます
- EN ver
    - 声の距離は必要ないかも。（そもそも声の大きさが変わらない）
    - A female（ランダム）にしても出力時間は変わらない（若干遅いときもあるけど）


# 発話させる文章について
文末にピリオド(.)あるいは句点(。)をつけてください．
- EN ver
    - 1単語の場合だと超遅い（10秒くらい）
    - 半角数字は微妙。ここは単語の数字のほうがいい
- JA ver
    - 全角スペース、をするとエラー起きる
    - 半角の数字だとバグる（読めるけど）
    - 漢数字は読める
    - ひらがなとカタカナで差は無い気がする（ただ、一単語の場合はひらがなが優勢な気がする）
    - あと誤字るたびにエラーが起きる（多分全角類のエラー）
    - 文章内にローマ字があるとバグる（”あなたはyuhashiですか”だとローマ字の部分がバグる）

# 言語モデルについて
以下のモデルを使用しています．
英語モデル：
https://huggingface.co/parler-tts/parler-tts-mini-v1

日本語モデル：
https://huggingface.co/2121-8/japanese-parler-tts-mini

別のモデルを使用したい場合は下記のリンクから選択し，
model_download.pyとparler_tts_server.pyのモデルに関する項目を変更してください．
https://huggingface.co/models?other=parler_tts&sort=likes
