<a name="readme-top"></a>
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![License][license-shield]][license-url]

# Parler TTS for ROS

<!-- 目次 -->
<details>
  <summary>目次</summary>
  <ol>
    <li>
      <a href="#概要">概要</a>
    </li>
    <li>
      <a href="#セットアップ">セットアップ</a>
      <ul>
        <li><a href="#環境条件">環境条件</a></li>
        <li><a href="#インストール方法">インストール方法</a></li>
      </ul>
    </li>
    <li><a href="#実行操作方法">実行・操作方法</a></li>
    <li><a href="#言語モデルの切り替えについて">言語モデルの切り替えについて</a>
        <ul>
            <li><a href="#英語モデル">英語モデル</a></li>
            <li><a href="#日本語モデル">日本語モデル</a></li>
        </ul>
    </li>
    <li><a href="#話者や話し方の設定について">話者や話し方の設定について</a>
        <ul>
            <li><a href="#ランダムな話者">ランダムな話者</a></li>
            <li><a href="#特定の話者">特定の話者</a></li>
        </ul>
    </li>
    <li><a href="#発話させる文章について">発話させる文章について</a></li>
    <li><a href="#マイルストーン">マイルストーン</a></li>
    <!-- <li><a href="#contributing">Contributing</a></li> -->
    <!-- <li><a href="#license">License</a></li> -->
    <li><a href="#参考文献">参考文献</a></li>
  </ol>
</details>


<!-- レポジトリの概要 -->
## 概要
Parler_TTSは，特定のスピーカーのスタイル(性別，ピッチ，話し方など)で高品質で自然な音声を生成できる軽量のテキスト読み上げ(TTS)モデルです．

Stability AIとエジンバラ大学のDan LythとSimon Kingによる論文[Natural language guidance of high-fidelity text-to-speech with synthetic annotations](https://www.text-description-to-speech.com) からの複製されました．

本リポジトリはParler_TTSをROS2のAction通信で使用できるようにしたもので，英語と日本語での発話に対応しています．

<p align="right">(<a href="#readme-top">上に戻る</a>)</p>

<!-- セットアップ -->
## セットアップ

ここで，本レポジトリのセットアップ方法について説明します．

<p align="right">(<a href="#readme-top">上に戻る</a>)</p>

### 環境条件

まず，以下の環境を整えてから，次のインストール段階に進んでください．
| System  | Version |
| --- | --- |
| Ubuntu | 22.04 (Jammy Jellyfish) |
| ROS    | Humble Hawksbill |
| Python | 3.10 |

<p align="right">(<a href="#readme-top">上に戻る</a>)</p>

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
    ```
    ```sh
    colcon build --symlink-install
    ```
    ```sh
    source ~/colcon_ws/install/setup.sh
    ```
6. 言語モデルをダウンロードします．時間がかかるので注意してください．
    ```sh
    ros2 run parler_tts_ros model_download
    ```
<p align="right">(<a href="#readme-top">上に戻る</a>)</p>

<!-- 実行・操作方法 -->
## 実行・操作方法
1. アクションサーバーを起動します．**Ready to ParlerTTS in**と表示されるまでgoalを送らずに待機してください．(時間がかかるので注意)
   ```sh
   ros2 launch parler_tts_ros parler_tts_server.launch.py 
   ```
2. アクションクライアントを起動し，発話させたい文字を送信します．

<p align="right">(<a href="#readme-top">上に戻る</a>)</p>

## 言語モデルの切り替えについて

1. **model_download.py**の**model_name**の項目を書き換えて実行し，モデルをダウンロードします．
2. **parler_tts_server.launch.py**にある**model_name**の項目を書き換えて実行します．

> [!WARNING]
> CPUのみのPCを使用している場合，発話までに時間がかかるため，**CPU向けジェニー**の使用を推奨します．


### 英語モデル

| 説明 | モデル名 | 話者 |
| --- | --- | --- |
| ミニ(デフォルト) | parler-tts/parler-tts-mini-v1 | *34人から指定可能|
| ミニジェニー | parler-tts/parler-mini-v1-jenny | Jennyのみ |
| 感情指定可能 | parler-tts/parler-tts-mini-expresso | Jerry, Thomas, Talia, Elisabeth |
| CPU向けジェニー | parler-tts/parler-tiny-v1-jenny | Jennyのみ |

<details>
 <summary>* 英語のミニモデルで利用可能な34人の話者の全リスト(英語)</summary>

|  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- |
| Laura | Gary | Jon | Lea | Karen | Rick | Brenda |
| David | Eileen | Jordan | Mike | Yann | Joy | James |
| Eric | Lauren | Rose | Will | Jason | Aaron | Naomie |
| Alisa | Patrick | Jerry | Tina | Jenna | Bill | Tom |
| Carol | Barbara | Rebecca | Anna | Bruce | Emily |  |
</details>


### 日本語モデル
| 説明 | モデル名 | 話者 |
| --- | --- | --- | 
| ミニ |2121-8/japanese-parler-tts-mini | JSUTのみ |

別のモデルを使用したい場合は[こちら](https://huggingface.co/models?other=parler_tts&sort=likes)から選択し，
model_download.pyとparler_tts_server.pyのモデルに関する項目を変更してください．

<p align="right">(<a href="#readme-top">上に戻る</a>)</p>

## 話者や話し方の設定について
parler_tts_server.launch.pyにある**description**の項目で，単純な説明文で話者や話し方を設定できます．
- 感情指定可能モデルについて
    - "happy", "confused", "laughing", "sad", "whisper", "emphasis"などの感情を指定できます！

話者について，特定の話者(推奨)か，毎回ランダムな話者にするかを選ぶことができます．

### ランダムな話者
**description**の項目で、話者を**male**か**female**と指定してください．
- 例
    ``` bash
    default_value="A female speaker delivers a slightly expressive and animated speech with a moderate speed and pitch. The recording is of very high quality, with the speaker's voice sounding clear and very close up."
    ```
### 特定の話者
特定の話者を使うことで、毎回同じ話者で発話させることができます．
**description**の項目で、「利用可能な話者」に置き換えてください．

- 例
    ``` bash
    default_value='Alisa.fast speed. Expression is rich. The speaking voice is noisy.',
    ```

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
- 残りの音声機能(性別、発話速度、ピッチ、残響)は、プロンプトから直接制御できます
- EN ver
    - 声の距離は必要ないかも。（そもそも声の大きさが変わらない）
    - A female（ランダム）にしても出力時間は変わらない（若干遅いときもあるけど）

<p align="right">(<a href="#readme-top">上に戻る</a>)</p>

## 発話させる文章について
- 全モデル共通
    - **文末にピリオド(.)あるいは句点(。)をつけてください．**
    - **数字について，半角(1, 2など)より，one, 二などを推奨**
    - 句読点は、世代の韻律を制御するために使用できます(たとえば、カンマを使用して音声に小さな区切りを追加します)
    - 1単語のみ発話させる場合は生成に時間がかかります

<p align="right">(<a href="#readme-top">上に戻る</a>)</p>

<!-- マイルストーン -->
## マイルストーン

現時点のbugや新規機能の依頼を確認するために[Issueページ][issues-url] をご覧ください．

<p align="right">(<a href="#readme-top">上に戻る</a>)</p>

<!-- 参考文献 -->
## 参考文献
* [parler-tts](https://github.com/huggingface/parler-tts)

<p align="right">(<a href="#readme-top">上に戻る</a>)</p>

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/TeamSOBITS/parler_tts_ros.svg?style=for-the-badge
[contributors-url]: https://github.com/TeamSOBITS/parler_tts_ros/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/TeamSOBITS/parler_tts_ros.svg?style=for-the-badge
[forks-url]: https://github.com/TeamSOBITS/parler_tts_ros/network/members
[stars-shield]: https://img.shields.io/github/stars/TeamSOBITS/parler_tts_ros.svg?style=for-the-badge
[stars-url]: https://github.com/TeamSOBITS/parler_tts_ros/stargazers
[issues-shield]: https://img.shields.io/github/issues/TeamSOBITS/parler_tts_ros.svg?style=for-the-badge
[issues-url]: https://github.com/TeamSOBITS/parler_tts_ros/issues
[license-shield]: https://img.shields.io/github/license/TeamSOBITS/parler_tts_ros.svg?style=for-the-badge
[license-url]: LICENSE
