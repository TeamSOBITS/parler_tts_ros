import rclpy
from rclpy.node import Node
import pygame

from sobits_interfaces.action import TextToSpeech
from ament_index_python.packages import get_package_share_directory
from rclpy.action import ActionServer, GoalResponse, CancelResponse, ActionClient # ActionClientを追加 (もし必要なら)

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

import io


class ParlerTTSActionServer(Node):
    def __init__(self, device=None):
        super().__init__('parler_tts_action_server')

        init_start_time = time.time() # 初期化処理開始時刻

        self.declare_parameter('language', 'en')
        self.declare_parameter('description', 'Jenna delivers a slightly expressive and animated speech with a moderate speed and pitch. The recording is of very high quality, with the speaker voice sounding clear and very close up.')

        self.language = self.get_parameter('language').get_parameter_value().string_value
        self.description = self.get_parameter('description').get_parameter_value().string_value

        self.filename = os.path.join(get_package_share_directory('parler_tts_ros'), 'sounds', 'output') # Note: 現在の実装ではファイル書き出しはしていない

        self.device = device if device else "cuda:0" if torch.cuda.is_available() else "cpu"
        self.get_logger().info(f"Using device: {self.device}")
        self.model = None
        self.tokenizer = None
        self.prompt_tokenizer = None
        self.description_tokenizer = None
        self.language_config = {
            "en": {
                "model_name": "parler-tts/parler-tts-mini-v1",
                "tokenizer_name": "parler-tts/parler-tts-mini-v1",
            },
            #正式版：parler-tts/parler-tts-mini-v1
            "ja": {
                "model_name": "2121-8/japanese-parler-tts-mini",
                "prompt_tokenizer_name": "2121-8/japanese-parler-tts-mini",
                "description_tokenizer_name": "2121-8/japanese-parler-tts-mini",
            },
            #正式版　：2121-8/japanese-parler-tts-mini
            #ベータ版：2121-8/japanese-parler-tts-mini-bate
            #日本語のベータ版をつかうときは、英語のところのモデルに指定する必要あり
        }

        model_config = self.language_config.get(self.language)
        if not model_config:
            self.get_logger().error(f"Unsupported language: {self.language}")
            raise ValueError(f"Unsupported language: {self.language}")

        self.get_logger().info(f"Loading model: {model_config['model_name']}")
        model_load_start_time = time.time()
        self.model = ParlerTTSForConditionalGeneration.from_pretrained(model_config["model_name"]).to(self.device)
        model_load_end_time = time.time()
        self.get_logger().info(f"Model loaded in: {model_load_end_time - model_load_start_time:.4f} seconds")

        # torch.compile() の試行 (PyTorch 2.0+ 環境で効果がある可能性)
        # 大幅な速度向上が期待できる反面、互換性の問題や初回コンパイルに時間がかかる場合があります。
        # 有効にする場合は、以下のコメントを解除してください。
        # if hasattr(torch, 'compile'):
        #     self.get_logger().info("Attempting to compile the model with torch.compile()...")
        #     compile_start_time = time.time()
        #     try:
        #         # modeオプション: "default", "reduce-overhead", "max-autotune"
        #         # "reduce-overhead" はコンパイル時間を短縮しつつ速度向上を目指します
        #         # "max-autotune" はより時間をかけて最適なコンパイルを試みます
        #         self.model = torch.compile(self.model, mode="reduce-overhead")
        #         # self.model = torch.compile(self.model) # default mode
        #         compile_end_time = time.time()
        #         self.get_logger().info(f"Model compiled successfully in {compile_end_time - compile_start_time:.4f} seconds.")
        #     except Exception as e:
        #         self.get_logger().warn(f"Failed to compile the model: {e}. Using uncompiled model.")
        # else:
        #     self.get_logger().info("torch.compile() not available. Using uncompiled model.")


        tokenizer_load_start_time = time.time()
        if self.language == "en":
            self.get_logger().info(f"Loading tokenizer: {model_config['tokenizer_name']}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_config["tokenizer_name"])
            self.tokenizer.pad_token_id = self.model.config.pad_token_id
            self.inputs = self.tokenizer(self.description, return_tensors="pt").to(self.device)
        elif self.language == "ja":
            self.get_logger().info(f"Loading prompt tokenizer: {model_config['prompt_tokenizer_name']}")
            self.prompt_tokenizer = AutoTokenizer.from_pretrained(model_config["prompt_tokenizer_name"], subfolder="prompt_tokenizer")
            self.get_logger().info(f"Loading description tokenizer: {model_config['description_tokenizer_name']}")
            self.description_tokenizer = AutoTokenizer.from_pretrained(model_config["description_tokenizer_name"], subfolder="description_tokenizer")
            self.prompt_tokenizer.pad_token_id = self.model.config.pad_token_id
            self.description_tokenizer.pad_token_id = self.model.config.pad_token_id
            self.inputs = self.description_tokenizer(self.description, return_tensors="pt").to(self.device)
        tokenizer_load_end_time = time.time()
        self.get_logger().info(f"Tokenizers loaded in: {tokenizer_load_end_time - tokenizer_load_start_time:.4f} seconds")


        init_end_time = time.time()
        self.get_logger().info(f"セットアップ完了 (トータル初期化時間): {init_end_time - init_start_time:.4f} 秒")

        self._action_server = ActionServer(
            self,
            TextToSpeech,
            'speech_word',
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback)
        self.get_logger().info("Ready to ParlerTTS")

    def goal_callback(self, goal_request):
        self.get_logger().info('ゴールリクエストを受信')
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        self.get_logger().info('キャンセルリクエストを受信')
        return CancelResponse.ACCEPT

    def tts_en(self, text):
        tokenize_prompt_start_time = time.time()
        prompt_inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        tokenize_prompt_end_time = time.time()
        self.get_logger().debug(f"EN Prompt tokenization time: {tokenize_prompt_end_time - tokenize_prompt_start_time:.4f} sec")

        generation_start_time = time.time()
        with torch.inference_mode(): # 推論モードを適用
            generation = self.model.generate(
                input_ids=self.inputs.input_ids,
                attention_mask=self.inputs.attention_mask,
                prompt_input_ids=prompt_inputs.input_ids,
                prompt_attention_mask=prompt_inputs.attention_mask,
            )
        generation_end_time = time.time()
        self.get_logger().debug(f"EN Model generation time: {generation_end_time - generation_start_time:.4f} sec")

        audio_arr = generation.cpu().numpy().squeeze().astype(np.float32)
        sampling_rate = self.model.config.sampling_rate

        play_time = 0.0
        if sampling_rate > 0 and len(audio_arr) > 0:
            play_time = len(audio_arr) / float(sampling_rate)
            self.get_logger().info(f'Calculated Play Time[s]: {play_time:.4f}')
        else:
            self.get_logger().error(f"Invalid audio data or sampling rate for EN TTS. Audio length: {len(audio_arr)}, Sampling rate: {sampling_rate}")
            return 0.0, None

        buffer_write_start_time = time.time()
        buffer = io.BytesIO()
        try:
            sf.write(buffer, audio_arr, sampling_rate, format='WAV')
            buffer.seek(0)
            buffer_write_end_time = time.time()
            self.get_logger().debug(f"EN WAV buffer write time: {buffer_write_end_time - buffer_write_start_time:.4f} sec")
            return play_time, buffer
        except Exception as e:
            self.get_logger().error(f"Error writing EN WAV to buffer: {e}")
            return 0.0, None

    def tts_ja(self, text):
        ruby_start_time = time.time()
        prompt = add_ruby(text) # ルビ振り処理
        ruby_end_time = time.time()
        self.get_logger().debug(f"JA Ruby processing time: {ruby_end_time - ruby_start_time:.4f} sec")

        tokenize_prompt_start_time = time.time()
        prompt_inputs = self.prompt_tokenizer(prompt, return_tensors="pt").to(self.device)
        tokenize_prompt_end_time = time.time()
        self.get_logger().debug(f"JA Prompt tokenization time: {tokenize_prompt_end_time - tokenize_prompt_start_time:.4f} sec")
        
        generation_start_time = time.time()
        with torch.inference_mode(): # 推論モードを適用
            generation = self.model.generate(
                input_ids=self.inputs.input_ids,
                attention_mask=self.inputs.attention_mask,
                prompt_input_ids=prompt_inputs.input_ids,
                prompt_attention_mask=prompt_inputs.attention_mask,
            )
        generation_end_time = time.time()
        self.get_logger().debug(f"JA Model generation time: {generation_end_time - generation_start_time:.4f} sec")

        audio_arr = generation.cpu().numpy().squeeze().astype(np.float32)
        sampling_rate = self.model.config.sampling_rate

        play_time = 0.0
        if sampling_rate > 0 and len(audio_arr) > 0:
            play_time = len(audio_arr) / float(sampling_rate)
            self.get_logger().info(f'Calculated Play Time[s]: {play_time:.4f}')
        else:
            self.get_logger().error(f"Invalid audio data or sampling rate for JA TTS. Audio length: {len(audio_arr)}, Sampling rate: {sampling_rate}")
            return 0.0, None
        
        buffer_write_start_time = time.time()
        buffer = io.BytesIO()
        try:
            sf.write(buffer, audio_arr, sampling_rate, format='WAV')
            buffer.seek(0)
            buffer_write_end_time = time.time()
            self.get_logger().debug(f"JA WAV buffer write time: {buffer_write_end_time - buffer_write_start_time:.4f} sec")
            return play_time, buffer
        except Exception as e:
            self.get_logger().error(f"Error writing JA WAV to buffer: {e}")
            return 0.0, None

    def execute_callback(self, goal_handle):
        cb_node_create_start_time = time.time()
        # execute_callback内でrclpy.spin_onceを呼び出すために一時的なノードを作成
        # これがパフォーマンスに大きな影響を与える場合は、より軽量なタイマーや
        # アクションサーバー自身のスレッドでフィードバックを管理する方法を検討する必要があるかもしれません。
        # ただし、通常は音声生成時間の方が支配的です。
        thread_node = Node(f"cb_parler_tts_{time.time_ns()}") # ノード名が一意になるようにタイムスタンプ追加
        cb_node_create_end_time = time.time()
        self.get_logger().debug(f"Callback temp node creation time: {cb_node_create_end_time - cb_node_create_start_time:.4f} sec")

        request_process_start_time = time.time() # リクエスト処理開始時刻
        feedback = TextToSpeech.Feedback()
        response = TextToSpeech.Result()
        text = goal_handle.request.text
        
        try:
            decoded_text = codecs.decode(str(text).encode('utf-8'))
        except Exception as e:
            self.get_logger().error(f"Error decoding input text: {e}")
            response.success = False
            goal_handle.abort() # ゴールを失敗状態にする
            thread_node.destroy_node()
            del thread_node
            return response

        if not decoded_text or not decoded_text.strip():
            self.get_logger().error("Input text is empty or blank.")
            response.success = False
            goal_handle.abort()
            thread_node.destroy_node()
            del thread_node
            return response
        
        self.get_logger().info(f"Input text: [{decoded_text}]")

        # 古いファイルの削除 (現在はメモリバッファを使用しているため、この処理は不要かもしれません)
        # もし特定のパスにファイルを保存する要件がなければ、この部分は削除可能です。
        # current_filename = self.filename + ".wav" # output.wav
        # if os.path.exists(current_filename):
        #     self.get_logger().debug(f"Removing old file: {current_filename}")
        #     os.remove(current_filename)

        self.get_logger().info(f"Processing ParlerTTS request for: '{decoded_text}'")

        response.success = False
        response.total_time = 0.0
        play_time = 0.0
        audio_buffer = None

        # 音声合成処理
        synthesis_start_time = time.time()
        if self.language == "en":
            play_time, audio_buffer = self.tts_en(decoded_text)
        elif self.language == "ja":
            play_time, audio_buffer = self.tts_ja(decoded_text)
        synthesis_end_time = time.time()
        self.get_logger().info(f"Total audio synthesis time: {synthesis_end_time - synthesis_start_time:.4f} sec")

        if audio_buffer is None or play_time <= 0:
            self.get_logger().error("Audio buffer generation failed or invalid play time.")
            response.success = False
            goal_handle.abort()
            thread_node.destroy_node()
            del thread_node
            return response

        # Pygame再生
        try:
            pygame_init_start_time = time.time()
            pygame.mixer.init()
            pygame_init_end_time = time.time()
            self.get_logger().info(f"Pygame mixer init time: {pygame_init_end_time - pygame_init_start_time:.4f} sec")

            pygame_load_start_time = time.time()
            pygame.mixer.music.load(audio_buffer)
            pygame_load_end_time = time.time()
            self.get_logger().info(f"Pygame music load time: {pygame_load_end_time - pygame_load_start_time:.4f} sec")

            overall_preparation_time = time.time()
            self.get_logger().info(f"Total time from request process start to pre-play: {overall_preparation_time - request_process_start_time:.4f} sec")
            
            pygame.mixer.music.play()
            play_signal_time = time.time()
            self.get_logger().info(f"発話開始 (play() called). Time from request process start: {play_signal_time - request_process_start_time:.4f} sec")


            interval = 0.1 # フィードバック送信間隔
            feedback.remaining_time = play_time
            start_playback_loop_time = time.time()

            while rclpy.ok() and pygame.mixer.music.get_busy(): # pygame.mixer.music.get_busy()で再生中か確認
                if goal_handle.is_cancel_requested:
                    self.get_logger().info('Goal canceled during playback.')
                    pygame.mixer.music.stop()
                    goal_handle.canceled()
                    response.success = False # キャンセル時は通常 False
                    break # ループを抜ける

                rclpy.spin_once(thread_node, timeout_sec=0.01) # タイムアウトを短くして反応性を上げる
                
                current_time_in_loop = time.time()
                elapsed_in_loop = current_time_in_loop - start_playback_loop_time
                response.total_time = elapsed_in_loop # 経過時間を更新
                feedback.remaining_time = play_time - elapsed_in_loop # 残り時間を更新
                
                if feedback.remaining_time < 0:
                    feedback.remaining_time = 0.0

                goal_handle.publish_feedback(feedback)

                # 実際の再生時間と計算上のplay_timeがずれる可能性を考慮し、
                # get_busy() が False になったらループを抜けるようにする。
                # ただし、念のため remaining_time もチェックする。
                if feedback.remaining_time <= 0:
                    # 少し待ってから最終確認
                    time.sleep(0.1) # 念のためバッファが空になるのを待つ
                    if not pygame.mixer.music.get_busy():
                        break
            
            # ループ終了後の処理
            if not goal_handle.is_cancel_requested:
                if pygame.mixer.music.get_busy(): # まだ再生中なら停止（予期せぬ場合）
                    pygame.mixer.music.stop()
                    self.get_logger().warn("Playback loop ended but music was still busy. Stopped.")

                feedback.remaining_time = 0.0
                goal_handle.publish_feedback(feedback)
                self.get_logger().info("ParlerTTS playback completed.")
                response.success = True
                goal_handle.succeed()
            # キャンセル済みなら response.success は False のまま

        except pygame.error as e:
            self.get_logger().error(f"Pygame error during playback: {e}")
            response.success = False
            goal_handle.abort() # pygameエラー時もabort
        except Exception as e:
            self.get_logger().error(f"An unexpected error occurred in execute_callback: {e}")
            response.success = False
            goal_handle.abort()
        finally:
            if pygame.mixer.get_init():
                pygame.mixer.quit()
            
            cb_node_destroy_start_time = time.time()
            thread_node.destroy_node()
            del thread_node # 明示的な削除
            cb_node_destroy_end_time = time.time()
            self.get_logger().debug(f"Callback temp node destruction time: {cb_node_destroy_end_time - cb_node_destroy_start_time:.4f} sec")
            
            final_request_process_time = time.time()
            self.get_logger().info(f"Total execute_callback processing time: {final_request_process_time - request_process_start_time:.4f} sec")

        return response

def main(args=None):
    rclpy.init(args=args)
    action_server = ParlerTTSActionServer()
    try:
        rclpy.spin(action_server)
    except KeyboardInterrupt:
        action_server.get_logger().info('KeyboardInterrupt, shutting down...')
    finally:
        action_server.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()