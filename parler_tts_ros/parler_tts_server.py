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
import soundfile as sf
import tempfile
import time

class ParlerTTSActionServer(Node):
    def __init__(self, device=None):
        super().__init__('parler_tts_action_server')

        start_time = time.time()

        # パラメータの宣言 (launch ファイルから設定可能、デフォルト値も指定)
        self.declare_parameter('language', 'en')
        self.declare_parameter('description', 'Jenna delivers a slightly expressive and animated speech with a moderate speed and pitch. The recording is of very high quality, with the speaker voice sounding clear and very close up.')

        # パラメータの取得
        self.language = self.get_parameter('language').get_parameter_value().string_value
        self.description = self.get_parameter('description').get_parameter_value().string_value

        # デバイス設定
        self.device = device if device else "cuda:0" if torch.cuda.is_available() else "cpu"
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
            raise ValueError(f"Unsupported language: {self.language}")
        self.model = ParlerTTSForConditionalGeneration.from_pretrained(model_config["model_name"]).to(self.device)

        if self.language == "en":
            self.tokenizer = AutoTokenizer.from_pretrained(model_config["tokenizer_name"])
            self.tokenizer.pad_token_id = self.model.config.pad_token_id
        elif self.language == "ja":
            self.prompt_tokenizer = AutoTokenizer.from_pretrained(model_config["prompt_tokenizer_name"], subfolder="prompt_tokenizer")
            self.description_tokenizer = AutoTokenizer.from_pretrained(model_config["description_tokenizer_name"], subfolder="description_tokenizer")
            self.prompt_tokenizer.pad_token_id = self.model.config.pad_token_id
            self.description_tokenizer.pad_token_id = self.model.config.pad_token_id
        end_time = time.time()
        self.get_logger().info(f"セットアップ完了: {end_time - start_time:.4f} 秒")

        pygame.mixer.init() # ここで一度だけ初期化

        self._action_server = ActionServer(
            self,
            TextToSpeech,
            'speech_word',
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback)
        self.get_logger().info("ParlerTTS アクションサーバー起動")
        self.get_logger().info("ParlerTTS は準備完了です。")

    def destroy_node(self):
        pygame.mixer.quit() # ノード終了時に一度だけ終了
        super().destroy_node()

    def goal_callback(self, goal_request):
        self.get_logger().info('ゴールリクエストを受信')
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        self.get_logger().info('キャンセルリクエストを受信')
        return CancelResponse.ACCEPT

    def _prepare_inputs(self, prompt):
        start_time = time.time()
        if self.language == "en":
            inputs = self.tokenizer(self.description, return_tensors="pt").to(self.device)
            prompt_inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        elif self.language == "ja":
            inputs = self.description_tokenizer(self.description, return_tensors="pt").to(self.device)
            prompt_inputs = self.prompt_tokenizer(prompt, return_tensors="pt").to(self.device)
        end_time = time.time()
        self.get_logger().debug(f"トークナイザーの入力準備: {end_time - start_time:.4f} 秒")
        return inputs, prompt_inputs

    def _process_and_play_audio(self, goal_handle, input_ids, attention_mask, prompt_input_ids, prompt_attention_mask):
        start_time = time.time()
        feedback = TextToSpeech.Feedback()
        result = TextToSpeech.Result()

        try:
            generation = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                prompt_input_ids=prompt_input_ids,
                prompt_attention_mask=prompt_attention_mask,
            )
            audio_arr = generation.cpu().numpy().squeeze().astype(np.float32)
            sampling_rate = self.model.config.sampling_rate

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmpfile:
                sf.write(tmpfile.name, audio_arr, sampling_rate)
                pygame.mixer.music.load(tmpfile.name)
                pygame.mixer.music.play()

                play_time = 0.0
                if pygame.mixer.music.get_busy():
                    sound = pygame.mixer.Sound(tmpfile.name)
                    play_time = sound.get_length()

                interval = 0.1
                feedback.remaining_time = play_time
                goal_handle.publish_feedback(feedback)

                while rclpy.ok() and pygame.mixer.music.get_busy():
                    if goal_handle.is_cancel_requested:
                        self.get_logger().info('ゴールがキャンセルされました')
                        pygame.mixer.music.stop()
                        goal_handle.canceled()
                        return result

                    rclpy.spin_once(self, timeout_sec=interval)
                    feedback.remaining_time -= interval
                    goal_handle.publish_feedback(feedback)

                pygame.mixer.music.stop() # 再生終了を明示的に停止
                result.success = True
                result.total_time = play_time
                goal_handle.succeed()
                end_time = time.time()
                self.get_logger().info(f"音声生成と再生処理完了: {end_time - start_time:.4f} 秒")

        except Exception as e:
            self.get_logger().error(f"音声生成中にエラーが発生しました: {e}")
            result.success = False
            goal_handle.abort()

        return result

    def execute_callback(self, goal_handle):
        self.get_logger().info(f"TTSリクエスト処理中: {goal_handle.request.text}")
        text = goal_handle.request.text

        if self.language == "ja":
            prompt = add_ruby(text)
        else:
            prompt = text

        try:
            inputs, prompt_inputs = self._prepare_inputs(prompt)
            result = self._process_and_play_audio(
                goal_handle,
                inputs.input_ids,
                inputs.attention_mask,
                prompt_input_ids=prompt_inputs.input_ids,
                prompt_attention_mask=prompt_inputs.attention_mask,
            )
            return result
        except Exception as e:
            self.get_logger().error(f"TTS処理中にエラーが発生しました: {e}")
            result = TextToSpeech.Result()
            result.success = False
            goal_handle.abort()
            return result

def main(args=None):
    rclpy.init(args=args)
    action_server = ParlerTTSActionServer()
    rclpy.spin(action_server)
    action_server.destroy_node() # ノード終了時に pygame.mixer.quit() を呼び出す
    rclpy.shutdown()

if __name__ == "__main__":
    main()