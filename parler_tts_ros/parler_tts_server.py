import rclpy
from rclpy.node import Node
import pygame

from sobits_interfaces.action import TextToSpeech
from rclpy.action import ActionServer, GoalResponse, CancelResponse

from parler_tts import ParlerTTSForConditionalGeneration
from rubyinserter import add_ruby
from transformers import AutoTokenizer
import numpy as np
import torch

import codecs
import soundfile as sf
import time

import io


class ParlerTTSActionServer(Node):
    def __init__(self, device=None):
        super().__init__('parler_tts_action_server')

        init_start_time = time.time()

        self.declare_parameter('language', 'en')
        self.declare_parameter('description', 'Jenna delivers a slightly expressive and animated speech with a moderate speed and pitch. The recording is of very high quality, with the speaker voice sounding clear and very close up.')

        self.language = self.get_parameter('language').get_parameter_value().string_value
        self.description = self.get_parameter('description').get_parameter_value().string_value

        self.device = device if device else "cuda:0" if torch.cuda.is_available() else "cpu"
        self.get_logger().debug(f"Using device: {self.device}")
        self.model = None
        self.tokenizer = None
        self.prompt_tokenizer = None
        self.description_tokenizer = None
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

        model_config = self.language_config.get(self.language)
        if not model_config:
            self.get_logger().error(f"Unsupported language: {self.language}")
            raise ValueError(f"Unsupported language: {self.language}")

        self.get_logger().debug(f"Loading model: {model_config['model_name']}")
        self.model = ParlerTTSForConditionalGeneration.from_pretrained(model_config["model_name"]).to(self.device)

        # torch.compile() は初期化時に一度だけ行う
        if hasattr(torch, 'compile'):
            self.get_logger().info("Attempting to compile the model with torch.compile()...")
            compile_start_time = time.time()
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
                compile_end_time = time.time()
                self.get_logger().info(f"Model compiled successfully in {compile_end_time - compile_start_time:.4f} seconds.")
            except Exception as e:
                self.get_logger().warn(f"Failed to compile the model: {e}. Using uncompiled model.")
        else:
            self.get_logger().info("torch.compile() not available. Using uncompiled model.")

        if self.language == "en":
            self.get_logger().debug(f"Loading tokenizer: {model_config['tokenizer_name']}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_config["tokenizer_name"])
            self.tokenizer.pad_token_id = self.model.config.pad_token_id
            self.inputs = self.tokenizer(self.description, return_tensors="pt").to(self.device)
        elif self.language == "ja":
            self.get_logger().debug(f"Loading prompt tokenizer: {model_config['prompt_tokenizer_name']}")
            self.prompt_tokenizer = AutoTokenizer.from_pretrained(model_config["prompt_tokenizer_name"], subfolder="prompt_tokenizer")
            self.get_logger().debug(f"Loading description tokenizer: {model_config['description_tokenizer_name']}")
            self.description_tokenizer = AutoTokenizer.from_pretrained(model_config["description_tokenizer_name"], subfolder="description_tokenizer")
            self.prompt_tokenizer.pad_token_id = self.model.config.pad_token_id
            self.description_tokenizer.pad_token_id = self.model.config.pad_token_id
            self.inputs = self.description_tokenizer(self.description, return_tensors="pt").to(self.device)

        # Pygame ミキサーの初期化を __init__ で一度だけ行う
        try:
            pygame.mixer.init()
            self.get_logger().info("Pygame mixer initialized.")
        except Exception as e:
            self.get_logger().error(f"Failed to initialize Pygame mixer: {e}")
            # 初期化に失敗した場合、TTS 機能は利用できない可能性があるが、ノード自体は起動させる
            # 後続の実行時にエラーが発生する可能性あり

        self._action_server = ActionServer(
            self,
            TextToSpeech,
            'speech_word',
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback)

        init_end_time = time.time()
        self.get_logger().info(f"Ready to ParlerTTS in: {init_end_time - init_start_time:.4f} seconds")

    # ノード破棄時のクリーンアップ処理
    def destroy_node(self):
        self.get_logger().info('Shutting down ParlerTTS action server...')
        # Pygame ミキサーの解放
        if pygame.mixer.get_init():
            try:
                pygame.mixer.quit()
                self.get_logger().info('Pygame mixer quit.')
            except Exception as e:
                 self.get_logger().error(f"Error quitting Pygame mixer: {e}")
        # 親クラスの destroy_node を呼び出す
        super().destroy_node()


    def goal_callback(self, goal_request):
        self.get_logger().debug('ゴールリクエストを受信')
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        self.get_logger().debug('キャンセルリクエストを受信')
        return CancelResponse.ACCEPT

    def tts_en(self, text):
        prompt_inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        with torch.inference_mode():
            generation = self.model.generate(
                input_ids=self.inputs.input_ids,
                attention_mask=self.inputs.attention_mask,
                prompt_input_ids=prompt_inputs.input_ids,
                prompt_attention_mask=prompt_inputs.attention_mask,
            )

        audio_arr = generation.cpu().numpy().squeeze().astype(np.float32)
        sampling_rate = self.model.config.sampling_rate

        play_time = 0.0
        if sampling_rate > 0 and len(audio_arr) > 0:
            play_time = len(audio_arr) / float(sampling_rate)
            self.get_logger().info(f'Calculated Play Time[s]: {play_time:.4f}')
        else:
            self.get_logger().error(f"Invalid audio data or sampling rate for EN TTS. Audio length: {len(audio_arr)}, Sampling rate: {sampling_rate}")
            return 0.0, None

        buffer = io.BytesIO()
        try:
            sf.write(buffer, audio_arr, sampling_rate, format='WAV')
            buffer.seek(0)
            return play_time, buffer
        except Exception as e:
            self.get_logger().error(f"Error writing EN WAV to buffer: {e}")
            return 0.0, None

    def tts_ja(self, text):
        prompt = add_ruby(text)

        prompt_inputs = self.prompt_tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.inference_mode():
            generation = self.model.generate(
                input_ids=self.inputs.input_ids,
                attention_mask=self.inputs.attention_mask,
                prompt_input_ids=prompt_inputs.input_ids,
                prompt_attention_mask=prompt_inputs.attention_mask,
            )

        audio_arr = generation.cpu().numpy().squeeze().astype(np.float32)
        sampling_rate = self.model.config.sampling_rate

        play_time = 0.0
        if sampling_rate > 0 and len(audio_arr) > 0:
            play_time = len(audio_arr) / float(sampling_rate)
            self.get_logger().info(f'Calculated Play Time[s]: {play_time:.4f}')
        else:
            self.get_logger().error(f"Invalid audio data or sampling rate for JA TTS. Audio length: {len(audio_arr)}, Sampling rate: {sampling_rate}")
            return 0.0, None

        buffer = io.BytesIO()
        try:
            sf.write(buffer, audio_arr, sampling_rate, format='WAV')
            buffer.seek(0)
            return play_time, buffer
        except Exception as e:
            self.get_logger().error(f"Error writing JA WAV to buffer: {e}")
            return 0.0, None


    def execute_callback(self, goal_handle):
        # コールバック処理用の軽量な一時ノード (キャンセルの検出などに使用)
        # ノードの作成・破棄がオーバーヘッドになる可能性はあるが、キャンセルのために一旦維持
        thread_node = Node(f"cb_parler_tts_{time.time_ns()}")


        request_process_start_time = time.time()
        feedback = TextToSpeech.Feedback()
        response = TextToSpeech.Result()
        text = goal_handle.request.text

        try:
            decoded_text = codecs.decode(str(text).encode('utf-8'))
        except Exception as e:
            self.get_logger().error(f"Error decoding input text: {e}")
            response.success = False
            goal_handle.abort()
            # エラー時も一時ノードを破棄
            thread_node.destroy_node()
            del thread_node
            return response

        if not decoded_text or not decoded_text.strip():
            self.get_logger().error("Input text is empty or blank.")
            response.success = False
            goal_handle.abort()
            # エラー時も一時ノードを破棄
            thread_node.destroy_node()
            del thread_node
            return response

        self.get_logger().info(f"Input text: [{decoded_text}]")
        self.get_logger().debug(f"Processing ParlerTTS request for: '{decoded_text}'")

        # Pygame が初期化されていない場合はエラー
        if not pygame.mixer.get_init():
            self.get_logger().error("Pygame mixer is not initialized.")
            response.success = False
            goal_handle.abort()
            thread_node.destroy_node()
            del thread_node
            return response


        response.success = False
        response.total_time = 0.0
        play_time = 0.0
        audio_buffer = None

        # TTS 合成処理
        if self.language == "en":
            play_time, audio_buffer = self.tts_en(decoded_text)
        elif self.language == "ja":
            play_time, audio_buffer = self.tts_ja(decoded_text)

        if audio_buffer is None or play_time <= 0:
            self.get_logger().error("Audio buffer generation failed or invalid play time.")
            response.success = False
            goal_handle.abort()
            thread_node.destroy_node()
            del thread_node
            return response

        try:
            # Pygame の初期化は __init__ で済んでいるため不要
            # pygame.mixer.init()

            # 生成された音声データを Pygame にロード
            # ロード処理はメインの遅延要因の一つになる可能性がある
            pygame.mixer.music.load(audio_buffer)

            # 再生開始
            pygame.mixer.music.play()
            play_signal_time = time.time()
            # リクエスト受付から発話開始までの時間をログ出力
            self.get_logger().info(f"Time from request to speech: {play_signal_time - request_process_start_time:.4f} seconds")

            feedback.remaining_time = play_time
            start_playback_loop_time = time.time()

            # 再生中のループ処理 (フィードバックとキャンセルの検出)
            while rclpy.ok() and pygame.mixer.music.get_busy():
                if goal_handle.is_cancel_requested:
                    self.get_logger().info('Goal canceled during playback.')
                    pygame.mixer.music.stop()
                    goal_handle.canceled()
                    response.success = False
                    break

                # 一時ノードを使って ROS 2 イベントを処理 (キャンセルの検出など)
                rclpy.spin_once(thread_node, timeout_sec=0.01)

                current_time_in_loop = time.time()
                elapsed_in_loop = current_time_in_loop - start_playback_loop_time
                response.total_time = elapsed_in_loop
                feedback.remaining_time = play_time - elapsed_in_loop

                if feedback.remaining_time < 0:
                    feedback.remaining_time = 0.0

                goal_handle.publish_feedback(feedback)

                # 残り時間が0以下になったらループを抜ける準備
                if feedback.remaining_time <= 0:
                     # 少し待って music.get_busy() が更新されるのを待つ
                     time.sleep(0.05)
                     if not pygame.mixer.music.get_busy():
                          break


            # ループ終了後の処理
            if not goal_handle.is_cancel_requested:
                # ループを抜けたがまだ再生中だった場合 (まれなケース)
                if pygame.mixer.music.get_busy():
                     pygame.mixer.music.stop()
                     self.get_logger().warn("Playback loop ended but music was still busy. Stopped.")

                # 最終的なフィードバックを送信
                feedback.remaining_time = 0.0
                goal_handle.publish_feedback(feedback)
                self.get_logger().info("ParlerTTS playback completed.")
                response.success = True
                goal_handle.succeed()

        except pygame.error as e:
            self.get_logger().error(f"Pygame error during playback: {e}")
            response.success = False
            goal_handle.abort()
        except Exception as e:
            self.get_logger().error(f"An unexpected error occurred in execute_callback: {e}")
            response.success = False
            goal_handle.abort()
        finally:
            # Pygame の解放は destroy_node で行うため、ここでは不要
            # if pygame.mixer.get_init():
            #     pygame.mixer.quit()

            # 一時ノードを破棄
            thread_node.destroy_node()
            del thread_node

        return response

def main(args=None):
    rclpy.init(args=args)
    action_server = ParlerTTSActionServer()
    # rclpy.spin はノードが終了するまでブロックされる
    rclpy.spin(action_server)
    # spin が終了したらノード破棄処理が行われる (destroy_node が呼ばれる)
    # action_server.destroy_node() # spin が終わると自動的に呼ばれるので不要
    rclpy.shutdown()

if __name__ == "__main__":
    main()