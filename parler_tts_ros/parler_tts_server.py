import rclpy
from rclpy.node import Node
import pygame

from sobits_interfaces.action import TextToSpeech
from rclpy.action import ActionServer, GoalResponse, CancelResponse

from parler_tts import ParlerTTSForConditionalGeneration
# 日本語特有の処理（ルビ挿入など）に使用
from rubyinserter import add_ruby
from transformers import AutoTokenizer
import numpy as np
import soundfile as sf
import torch
import codecs
import time
import io

class ParlerTTSActionServer(Node):
    def __init__(self, device=None):
        super().__init__('parler_tts_action_server')
        # 初期化開始時間の記録
        init_start_time = time.time()

        # ROS 2パラメータとして'language'を宣言し、デフォルト値を設定
        self.declare_parameter('language', 'en')
        # ROS 2パラメータとして'description'を宣言し、デフォルト値を設定（音声の特徴記述）
        self.declare_parameter('description', 'Jenna delivers a slightly expressive and animated speech with a moderate speed and pitch. The recording is of very high quality, with the speaker voice sounding clear and very close up.')

        # 宣言したパラメータの値を取得
        self.language = self.get_parameter('language').get_parameter_value().string_value
        self.description = self.get_parameter('description').get_parameter_value().string_value

        # 使用するデバイス（CPUまたはGPU）を設定
        self.device = device if device else "cuda:0" if torch.cuda.is_available() else "cpu"
        self.get_logger().debug(f"Using device: {self.device}")

        # モデル、トークナイザーのインスタンス変数を初期化
        self.model = None
        self.tokenizer = None
        self.prompt_tokenizer = None
        self.description_tokenizer = None

        # 言語ごとのモデル設定を定義
        self.language_config = {
            "en": {
                "model_name": "parler-tts/parler-tts-mini-v1",
                "tokenizer_name": "parler-tts/parler-tts-mini-v1",
            },
            "ja": {
                "model_name": "2121-8/japanese-parler-tts-mini",
                "prompt_tokenizer_name": "2121-8/japanese-parler-tts-mini",
                "description_tokenizer_name": "2121-8/japanese-parler-tts-mini",
            },
        }

        # 現在の言語設定に対応するモデル設定を取得
        model_config = self.language_config.get(self.language)
        # サポートされていない言語の場合はエラーとして例外を発生させる
        if not model_config:
            self.get_logger().error(f"Unsupported language: {self.language}")
            raise ValueError(f"Unsupported language: {self.language}")
        self.get_logger().debug(f"Loading model: {model_config['model_name']}")

        # モデルをロードし、指定されたデバイスに配置
        self.model = ParlerTTSForConditionalGeneration.from_pretrained(model_config["model_name"]).to(self.device)

        # PyTorchのコンパイル機能が利用可能かチェックし、利用可能ならモデルをコンパイル
        if hasattr(torch, 'compile'):
            compile_start_time = time.time()
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
                compile_end_time = time.time()
                self.get_logger().info(f"Model compiled successfully in {compile_end_time - compile_start_time:.4f} seconds.") # ログ出力
            except Exception as e:
                self.get_logger().warn(f"Failed to compile the model: {e}. Using uncompiled model.") # ログ出力
        else:
            self.get_logger().info("torch.compile() not available. Using uncompiled model.") # ログ出力
            pass # 何もしない

        # 言語に応じたトークナイザーをロードし、descriptionをトークナイズ
        if self.language == "en":
            self.get_logger().debug(f"Loading tokenizer: {model_config['tokenizer_name']}") # ログ出力
            self.tokenizer = AutoTokenizer.from_pretrained(model_config["tokenizer_name"])
            # パディングトークンのIDを設定
            self.tokenizer.pad_token_id = self.model.config.pad_token_id
            # 英語ではpromptとdescriptionで同じトークナイザーを使用
            self.prompt_tokenizer = self.tokenizer
            self.description_tokenizer = self.tokenizer
        elif self.language == "ja":
            self.get_logger().debug(f"Loading prompt tokenizer: {model_config['prompt_tokenizer_name']}") # ログ出力
            # 日本語のプロンプト用トークナイザーをロード
            self.prompt_tokenizer = AutoTokenizer.from_pretrained(model_config["prompt_tokenizer_name"], subfolder="prompt_tokenizer")
            self.get_logger().debug(f"Loading description tokenizer: {model_config['description_tokenizer_name']}") # ログ出力
            # 日本語のdescription用トークナイザーをロード
            self.description_tokenizer = AutoTokenizer.from_pretrained(model_config["description_tokenizer_name"], subfolder="description_tokenizer")
            # パディングトークンのIDを設定
            self.prompt_tokenizer.pad_token_id = self.model.config.pad_token_id
            self.description_tokenizer.pad_token_id = self.model.config.pad_token_id

        # description をトークナイズし、デバイスに配置 (どの言語でも必要)
        self.inputs = self.description_tokenizer(self.description, return_tensors="pt").to(self.device)

        # Pygameミキサーの初期化
        try:
            pygame.mixer.init()
            self.get_logger().info("Pygame mixer initialized.") # ログ出力
        except Exception as e:
            self.get_logger().error(f"Failed to initialize Pygame mixer: {e}") # ログ出力
            pass # エラー処理はログ出力に任せる

        # アクションサーバーの作成と初期化
        self._action_server = ActionServer(
            self, # ノードインスタンス
            TextToSpeech, # 使用するアクションインターフェース
            'speech_word', # アクションサーバー名
            execute_callback=self.execute_callback, # ゴール実行時に呼び出されるコールバック関数
            goal_callback=self.goal_callback, # ゴール受付時に呼び出されるコールバック関数
            cancel_callback=self.cancel_callback) # キャンセル受付時に呼び出されるコールバック関数

        # 初期化完了時間の記録と所要時間の計算
        init_end_time = time.time()
        self.get_logger().info(f"Ready to ParlerTTS in: {init_end_time - init_start_time:.4f} seconds") # ログ出力

    # ノード破棄時の処理
    def destroy_node(self):
        self.get_logger().info('Shutting down ParlerTTS action server...') # ログ出力
        # Pygameミキサーが初期化されていれば終了処理を行う
        if pygame.mixer.get_init():
            try:
                pygame.mixer.quit()
                self.get_logger().info('Pygame mixer quit.') # ログ出力
            except Exception as e:
                 self.get_logger().error(f"Error quitting Pygame mixer: {e}") # ログ出力
                 pass # エラー処理はログ出力に任せる
        # 親クラス（Node）のdestroy_nodeメソッドを呼び出し
        super().destroy_node()

    # ゴール受付時のコールバック関数
    def goal_callback(self, goal_request):
        self.get_logger().debug('Received goal request.') # ログ出力
        # ゴールリクエストを受け入れる
        return GoalResponse.ACCEPT

    # キャンセル受付時のコールバック関数
    def cancel_callback(self, goal_handle):
        self.get_logger().debug('Received cancel request.') # ログ出力
        # キャンセルリクエストを受け入れる
        return CancelResponse.ACCEPT

    # 共通のテキスト音声合成処理を行うメソッド
    def _generate_audio_buffer(self, text):
        # 処理対象のテキストを初期化
        processed_text = text
        # 日本語特有の前処理
        if self.language == "ja":
            # ルビ挿入処理を適用
            processed_text = add_ruby(text)
            self.get_logger().debug(f"Japanese text processed with ruby: {processed_text}") # ログ出力

        # 言語に応じたプロンプトトークナイザーを選択（__init__ で self.prompt_tokenizer が適切に設定されている）
        prompt_tokenizer = self.prompt_tokenizer

        # 処理済みのテキストをトークナイズし、デバイスに配置 (プロンプトとして使用)
        try:
             prompt_inputs = prompt_tokenizer(processed_text, return_tensors="pt").to(self.device)
        except Exception as e:
             self.get_logger().error(f"Error tokenizing prompt text for language {self.language}: {e}") # ログ出力
             # エラー時は再生時間0.0とNoneを返す
             return 0.0, None

        # 推論モードで音声データを生成
        with torch.inference_mode():
            try:
                 generation = self.model.generate(
                     input_ids=self.inputs.input_ids, # DescriptionのトークンID
                     attention_mask=self.inputs.attention_mask, # Descriptionのアテンションマスク
                     prompt_input_ids=prompt_inputs.input_ids, # PromptのトークンID
                     prompt_attention_mask=prompt_inputs.attention_mask, # Promptのアテンションマスク
                 )
            except Exception as e:
                 self.get_logger().error(f"Error during model generation for language {self.language}: {e}") # ログ出力
                 # エラー時は再生時間0.0とNoneを返す
                 return 0.0, None

        # 生成されたテンソルの後処理
        # テンソルをNumPy配列に変換し、squeezeで次元を削減、float32型に変換
        audio_arr = generation.cpu().numpy().squeeze().astype(np.float32)
        # モデルの設定からサンプリングレートを取得
        sampling_rate = self.model.config.sampling_rate

        # 音声の再生時間を計算
        play_time = 0.0
        if sampling_rate > 0 and len(audio_arr) > 0:
            play_time = len(audio_arr) / float(sampling_rate)
            self.get_logger().info(f'Calculated Play Time[s] ({self.language}): {play_time:.4f}') # ログ出力
        else:
            self.get_logger().error(f"Invalid audio data or sampling rate for {self.language} TTS. Audio length: {len(audio_arr)}, Sampling rate: {sampling_rate}") # ログ出力
            # 無効なデータの_場合は再生時間0.0とNoneを返す
            return 0.0, None

        # 音声データをバイナリバッファに書き込み（WAV形式）
        buffer = io.BytesIO()
        try:
            sf.write(buffer, audio_arr, sampling_rate, format='WAV')
            # バッファの読み取り位置を先頭に戻す
            buffer.seek(0)
            # 再生時間と音声データバッファを返す
            return play_time, buffer
        except Exception as e:
            self.get_logger().error(f"Error writing {self.language} WAV to buffer: {e}") # ログ出力
            # エラー時は再生時間0.0とNoneを返す
            return 0.0, None

    # ゴール実行時のコールバック関数
    def execute_callback(self, goal_handle):
        # このコールバック内でROS 2のspin_onceを呼び出すための新しいノードを作成
        thread_node = Node(f"cb_parler_tts_{time.time_ns()}")

        # リクエスト処理開始時間の記録
        request_process_start_time = time.time()
        # フィードバックメッセージと結果メッセージのインスタンスを作成
        feedback = TextToSpeech.Feedback()
        response = TextToSpeech.Result()
        # アクションゴールからテキストを取得
        text = goal_handle.request.text

        # 入力テキストをUTF-8でデコード
        try:
            decoded_text = codecs.decode(str(text).encode('utf-8'))
        except Exception as e:
            self.get_logger().error(f"Error decoding input text: {e}") # ログ出力
            # エラーが発生した場合、結果を失敗とし、ゴールを中断
            response.success = False
            goal_handle.abort()
            # 作成したスレッドノードを破棄
            thread_node.destroy_node()
            del thread_node
            # 結果を返却
            return response

        # デコードされたテキストが空または空白のみかチェック
        if not decoded_text or not decoded_text.strip():
            self.get_logger().error("Input text is empty or blank.") # ログ出力
            # テキストが空または空白の場合、結果を失敗とし、ゴールを中断
            response.success = False
            goal_handle.abort()
            # 作成したスレッドノードを破棄
            thread_node.destroy_node()
            del thread_node
            # 結果を返却
            return response

        self.get_logger().info(f"Input text: [{decoded_text}] (Language: {self.language})") # ログ出力
        self.get_logger().debug(f"Processing ParlerTTS request for: '{decoded_text}'") # ログ出力

        # Pygameミキサーが初期化されているかチェック
        if not pygame.mixer.get_init():
            self.get_logger().error("Pygame mixer is not initialized.") # ログ出力
            # 初期化されていない場合、結果を失敗とし、ゴールを中断
            response.success = False
            goal_handle.abort()
            # 作成したスレッドノードを破棄
            thread_node.destroy_node()
            del thread_node
            # 結果を返却
            return response

        # 結果の初期値を設定
        response.success = False
        response.total_time = 0.0
        play_time = 0.0
        audio_buffer = None

        # 共通化された音声データ生成メソッドを呼び出し
        play_time, audio_buffer = self._generate_audio_buffer(decoded_text)

        # 音声データ生成に失敗したか、再生時間が無効かチェック
        if audio_buffer is None or play_time <= 0:
            self.get_logger().error("Audio buffer generation failed or invalid play time.") # ログ出力
            # 失敗した場合、結果を失敗とし、ゴールを中断
            response.success = False
            goal_handle.abort()
            # 作成したスレッドノードを破棄
            thread_node.destroy_node()
            del thread_node
            # 結果を返却
            return response

        # Pygameミキサーで音声データをロードし、再生
        try:
            pygame.mixer.music.load(audio_buffer)
            pygame.mixer.music.play()
            # 再生開始時間の記録
            play_signal_time = time.time()
            self.get_logger().info(f"Time from request to speech: {play_signal_time - request_process_start_time:.4f} seconds") # ログ出力

            # フィードバックメッセージの残り時間を初期化
            feedback.remaining_time = play_time
            # 再生ループ開始時間の記録
            start_playback_loop_time = time.time()

            # ROS 2が実行中で、かつPygameミキサーが再生中である間ループ
            while rclpy.ok() and pygame.mixer.music.get_busy():
                # キャンセルリクエストがあったかチェック
                if goal_handle.is_cancel_requested:
                    self.get_logger().info('Goal canceled during playback.') # ログ出力
                    # 再生を停止し、ゴールをキャンセル済みに設定
                    pygame.mixer.music.stop()
                    goal_handle.canceled()
                    response.success = False
                    # ループを抜ける
                    break

                # ROS 2のコールバック処理を少しだけ実行
                rclpy.spin_once(thread_node, timeout_sec=0.01)

                # ループ内の現在時間と経過時間を計算
                current_time_in_loop = time.time()
                elapsed_in_loop = current_time_in_loop - start_playback_loop_time
                # 結果メッセージの合計時間を更新
                response.total_time = elapsed_in_loop
                # フィードバックメッセージの残り時間を更新
                feedback.remaining_time = play_time - elapsed_in_loop

                # 残り時間が負にならないように調整
                if feedback.remaining_time < 0:
                    feedback.remaining_time = 0.0

                # フィードバックメッセージをパブリッシュ
                goal_handle.publish_feedback(feedback)

                # 残り時間が0以下になったらループを抜ける
                if feedback.remaining_time <= 0:
                    # 再生終了を待つための短い遅延
                    time.sleep(0.05)
                    # ミキサーがまだビジーでないことを確認
                    if not pygame.mixer.music.get_busy():
                         break

            # キャンセルリクエストがなかった場合
            if not goal_handle.is_cancel_requested:
                # ループ終了時にミキサーがまだビジーだった場合、停止
                if pygame.mixer.music.get_busy():
                     pygame.mixer.music.stop()
                     self.get_logger().warn("Playback loop ended but music was still busy. Stopped.") # ログ出力

                # 最後のフィードバックとして残り時間を0に設定してパブリッシュ
                feedback.remaining_time = 0.0
                goal_handle.publish_feedback(feedback)
                self.get_logger().info("ParlerTTS playback completed.") # ログ出力
                response.success = True
                goal_handle.succeed()

        except pygame.error as e:
            self.get_logger().error(f"Pygame error during playback: {e}") # ログ出力
            # エラー発生時は結果を失敗とし、ゴールを中断
            response.success = False
            goal_handle.abort()
        except Exception as e:
            self.get_logger().error(f"An unexpected error occurred in execute_callback: {e}") # ログ出力
            # エラー発生時は結果を失敗とし、ゴールを中断
            response.success = False
            goal_handle.abort()
        finally:
            # 作成したスレッドノードを破棄
            thread_node.destroy_node()
            del thread_node

        return response

def main(args=None):
    rclpy.init(args=args)
    action_server = ParlerTTSActionServer()
    rclpy.spin(action_server)
    rclpy.shutdown()

if __name__ == "__main__":
    main()