import rclpy
from rclpy.node import Node
import pygame

from sobits_interfaces.action import TextToSpeech
from ament_index_python.packages import get_package_share_directory
from rclpy.action import ActionServer, GoalResponse, CancelResponse

from parler_tts import ParlerTTSForConditionalGeneration
from rubyinserter import add_ruby
from transformers import AutoTokenizer
import numpy as np
import torch

import subprocess
import codecs
import os
import soundfile as sf
import wave
import time

class ParlerTTSActionServer(Node):
    def __init__(self, device=None):
        super().__init__('parler_tts_action_server')

        start_time = time.time() # 処理開始時刻を記録

        # パラメータの宣言 (launch ファイルから設定可能、デフォルト値も指定)
        self.declare_parameter('language', 'en') # 言語設定のパラメータ宣言、デフォルトは英語
        self.declare_parameter('description', 'Jenna delivers a slightly expressive and animated speech with a moderate speed and pitch. The recording is of very high quality, with the speaker voice sounding clear and very close up.') # 音声合成の説明文のパラメータ宣言、デフォルトは英語話者の説明

        # パラメータの取得
        self.language = self.get_parameter('language').get_parameter_value().string_value # 言語パラメータの値を取得
        self.description = self.get_parameter('description').get_parameter_value().string_value # 音声合成の説明文パラメータの値を取得

        # 音声ファイル出力パスの生成
        self.filename = os.path.join(get_package_share_directory('parler_tts_ros'), 'sounds', 'output')

        # デバイス設定 (GPUが利用可能であればGPUを使用、そうでなければCPUを使用)
        self.device = device if device else "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = None # TTSモデルのインスタンスを初期化
        self.tokenizer = None # トークナイザーのインスタンスを初期化（英語用）
        self.prompt_tokenizer = None # プロンプト用のトークナイザーのインスタンスを初期化（日本語用）
        self.description_tokenizer = None # 説明文用のトークナイザーのインスタンスを初期化（日本語用）
        self.language_config = {
            "en": {
                "model_name": "parler-tts/parler-tts-mini-v1", # 英語モデルのプリトレイン済みモデル名
                "tokenizer_name": "parler-tts/parler-tts-mini-v1", # 英語トークナイザーのプリトレイン済みモデル名
            },
            "ja": {
                "model_name": "2121-8/japanese-parler-tts-mini", # 日本語モデルのプリトレイン済みモデル名
                "prompt_tokenizer_name": "2121-8/japanese-parler-tts-mini", # 日本語プロンプト用トークナイザーのプリトレイン済みモデル名
                "description_tokenizer_name": "2121-8/japanese-parler-tts-mini", # 日本語説明文用トークナイザーのプリトレイン済みモデル名
            },
        }

        model_config = self.language_config.get(self.language) # 設定された言語に対応するモデル構成を取得
        if not model_config:
            raise ValueError(f"Unsupported language: {self.language}") # サポートされていない言語が指定された場合はエラー

        # モデルとトークナイザーのロード
        self.model = ParlerTTSForConditionalGeneration.from_pretrained(model_config["model_name"]).to(self.device) # 指定されたプリトレイン済みモデルをロードし、指定されたデバイスへ移動

        if self.language == "en":
            self.tokenizer = AutoTokenizer.from_pretrained(model_config["tokenizer_name"]) # 英語トークナイザーをロード
            self.tokenizer.pad_token_id = self.model.config.pad_token_id # パディングトークンIDを設定
            self.inputs = self.tokenizer(self.description, return_tensors="pt").to(self.device) # 説明文をトークナイズし、PyTorchテンソルに変換してデバイスへ移動

        elif self.language == "ja":
            self.prompt_tokenizer = AutoTokenizer.from_pretrained(model_config["prompt_tokenizer_name"], subfolder="prompt_tokenizer") # 日本語プロンプト用トークナイザーをロード
            self.description_tokenizer = AutoTokenizer.from_pretrained(model_config["description_tokenizer_name"], subfolder="description_tokenizer") # 日本語説明文用トークナイザーをロード
            self.prompt_tokenizer.pad_token_id = self.model.config.pad_token_id # パディングトークンIDを設定
            self.description_tokenizer.pad_token_id = self.model.config.pad_token_id # パディングトークンIDを設定
            self.inputs = self.description_tokenizer(self.description, return_tensors="pt").to(self.device) # 説明文をトークナイズし、PyTorchテンソルに変換してデバイスへ移動

        end_time = time.time() # セットアップ完了時刻を記録
        self.get_logger().info(f"セットアップ完了: {end_time - start_time:.4f} 秒") # セットアップにかかった時間をログ出力

        # アクションサーバーの作成
        self._action_server = ActionServer(
            self,
            TextToSpeech, # 定義したアクションの型
            'speech_word', # アクション名
            execute_callback=self.execute_callback, # ゴール実行時のコールバック関数
            goal_callback=self.goal_callback, # ゴールリクエスト受信時のコールバック関数
            cancel_callback=self.cancel_callback) # ゴールキャンセルリクエスト受信時のコールバック関数
        self.get_logger().info("Ready to ParlerTTS")

    # ゴールリクエスト受信時のコールバック関数
    def goal_callback(self, goal_request):
        self.get_logger().info('ゴールリクエストを受信')
        return GoalResponse.ACCEPT # ゴールリクエストを受け付ける

    # ゴールキャンセルリクエスト受信時のコールバック関数
    def cancel_callback(self, goal_handle):
        self.get_logger().info('キャンセルリクエストを受信')
        return CancelResponse.ACCEPT # キャンセルリクエストを受け付ける

    # 英語のテキスト読み上げ処理
    def tts_en(self, text):
        # 入力データの準備
        prompt_inputs = self.tokenizer(text, return_tensors="pt").to(self.device) # プロンプト（テキスト）をトークナイズし、PyTorchテンソルに変換してデバイスへ移動

        # 音声合成を実行
        generation = self.model.generate(
            input_ids = self.inputs.input_ids,
            attention_mask = self.inputs.attention_mask,
            prompt_input_ids = prompt_inputs.input_ids,
            prompt_attention_mask = prompt_inputs.attention_mask,
        )
        audio_arr = generation.cpu().numpy().squeeze().astype(np.float32) # 生成された音声をNumPy配列に変換し、CPUへ移動
        sampling_rate = self.model.config.sampling_rate # モデルのサンプリングレートを取得

        # wavファイルの作成
        sf.write(self.filename + ".wav", audio_arr, sampling_rate)

        # 再生時間の取得
        try:
            f = sf.SoundFile(self.filename + ".wav")
            play_time = float(len(f)) / float(f.samplerate)
            self.get_logger().info(f'Time[s]: {str(play_time)}') # 再生時間のログ出力
            return play_time # 再生時間を返す
        except Exception as e:
            self.get_logger().error(f"Error getting play time: {e}")
            return 0.0

    def tts_ja(self, text):
        # ルビの追加
        prompt = add_ruby(text)

        # 入力データの準備
        prompt_inputs = self.prompt_tokenizer(prompt, return_tensors="pt").to(self.device) # プロンプト（ルビ付きテキスト）をトークナイズし、PyTorchテンソルに変換してデバイスへ移動

        # 音声合成を実行
        generation = self.model.generate(
            input_ids = self.inputs.input_ids,
            attention_mask = self.inputs.attention_mask,
            prompt_input_ids = prompt_inputs.input_ids,
            prompt_attention_mask = prompt_inputs.attention_mask,
        )
        audio_arr = generation.cpu().numpy().squeeze().astype(np.float32) # 生成された音声をNumPy配列に変換し、CPUへ移動
        sampling_rate = self.model.config.sampling_rate # モデルのサンプリングレートを取得

        # wavファイルの作成
        sf.write(self.filename + ".wav", audio_arr, sampling_rate)

        # 再生時間の取得
        try:
            f = sf.SoundFile(self.filename + ".wav")
            play_time = float(len(f)) / float(f.samplerate)
            self.get_logger().info(f'Time[s]: {str(play_time)}') # 再生時間のログ出力
            return play_time # 再生時間を返す
        except Exception as e:
            self.get_logger().error(f"Error getting play time: {e}")
            return 0.0

    def execute_callback(self, goal_handle):
        thread_node = Node("execute_callback_ParlerTTS") # コールバック内で一時的なノードを作成
        start_time = time.time() # 音声生成と再生処理開始時刻を記録
        feedback = TextToSpeech.Feedback() # フィードバックメッセージのインスタンスを作成
        response = TextToSpeech.Result() # 結果メッセージのインスタンスを作成
        text = goal_handle.request.text # リクエストされたテキストを取得
        
        # デコード
        text = codecs.decode(str(text).encode('utf-8'))
        # 空文字チェック
        if not text or not text.strip():
            self.get_logger().error("Input text is empty or blank.") # 空文字の場合はエラーログ出力
            return False
        # Unicodeエラーチェック
        try:
            self.get_logger().info("Input text [" + str(text) + " ] ") # 入力テキストのログ出力
        except Exception as e:
            self.get_logger().error(e) # Unicodeエラー発生時のログ出力
            return False
        # 古いファイルの削除
        if os.path.exists(self.filename + ".wav"):
            os.remove(self.filename + ".wav")
        self.get_logger().info(f"Processing ParlerTTS request: {codecs.decode(str(text).encode('utf-8'))}") # 処理開始のログ出力

        response.success = False # 初期値をFalseに設定
        response.total_time = 0.0 # 初期値を0.0に設定

        play_time = 0

        # 日本語の場合はルビを追加
        if self.language == "en":
            play_time = self.tts_en(text) # 英語の読み上げ処理を実行
        elif self.language == "ja":
            play_time = self.tts_ja(text) # 日本語の読み上げ処理を実行

        # pygameの初期化と音声ファイルの再生
        try:
            pygame.mixer.init() # Pygameのミキサーを初期化 (音声再生用)
            pygame.mixer.music.load(self.filename + ".wav")
            end_time = time.time() # 発話開始時刻を記録
            self.get_logger().info(f"発話開始: {end_time - start_time:.4f} 秒")
            pygame.mixer.music.play()

            # 再生時間が正常に取得できた場合
            if (play_time > 0):
                interval = 0.1 # フィードバック送信間隔
                feedback.remaining_time = play_time # 残り時間を初期化
                while rclpy.ok(): # ROSがシャットダウンされるまでループ
                    # キャンセルがリクエストされた場合
                    if goal_handle.is_cancel_requested:
                        self.get_logger().info('Goal canceled') # キャンセルログ出力
                        if pygame.mixer.music.get_busy():
                            pygame.mixer.music.stop() # 再生中の音楽を停止
                        goal_handle.canceled() # ゴールをキャンセル状態にする
                        thread_node.destroy_node() # 一時ノードを破棄
                        del thread_node # 一時ノードのオブジェクトを削除
                        return response # レスポンスを返す
                    rclpy.spin_once(thread_node, timeout_sec=0.1) # イベント処理とタイマーコールバックを処理
                    response.total_time += interval # 経過時間を更新
                    feedback.remaining_time -= interval # 残り時間を更新

                    # 再生が終了した場合
                    if (feedback.remaining_time <= 0.0):
                        break
                    else:
                        goal_handle.publish_feedback(feedback) # フィードバックを送信

                feedback.remaining_time = 0.0 # 残り時間を0に設定
                goal_handle.publish_feedback(feedback) # 最終フィードバックを送信
                self.get_logger().info("ParlerTTS playback completed.") # 処理完了のログ出力

                response.success = True # 成功フラグをTrueに設定
                goal_handle.succeed() # ゴールを成功状態にする
            else:
                self.get_logger().warn("Play time is not valid, skipping playback.")
        except pygame.error as e:
            self.get_logger().error(f"Pygame error during playback: {e}")
        finally:
            if pygame.mixer.get_init():
                pygame.mixer.quit()
            thread_node.destroy_node() # 一時ノードを破棄
            del thread_node # 一時ノードのオブジェクトを削除
            return response # レスポンスを返す

def main(args=None):
    rclpy.init(args=args) # ROS 2のクライアントライブラリを初期化
    action_server = ParlerTTSActionServer() # ParlerTTSActionServerのインスタンスを作成
    rclpy.spin(action_server) # ノードの実行を開始（コールバック関数などを処理）
    rclpy.shutdown() # ROS 2のシャットダウン

if __name__ == "__main__":
    main()