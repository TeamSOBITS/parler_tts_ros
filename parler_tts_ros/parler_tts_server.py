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

        # torch.compile() の試行 (コメントはそのまま)
        # if hasattr(torch, 'compile'):
        #     self.get_logger().info("Attempting to compile the model with torch.compile()...")
        #     compile_start_time = time.time()
        #     try:
        #         self.model = torch.compile(self.model, mode="reduce-overhead")
        #         compile_end_time = time.time()
        #         self.get_logger().info(f"Model compiled successfully in {compile_end_time - compile_start_time:.4f} seconds.")
        #     except Exception as e:
        #         self.get_logger().warn(f"Failed to compile the model: {e}. Using uncompiled model.")
        # else:
        #     self.get_logger().info("torch.compile() not available. Using uncompiled model.")

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

        self._action_server = ActionServer(
            self,
            TextToSpeech,
            'speech_word',
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback)

        init_end_time = time.time()
        self.get_logger().info(f"Ready to ParlerTTS in: {init_end_time - init_start_time:.4f} seconds")

    def goal_callback(self, goal_request):
        self.get_logger().debug('ゴールリクエストを受信')
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        self.get_logger().debug('キャンセルリクエストを受信')
        return CancelResponse.ACCEPT

    def tts_en(self, text):
        prompt_inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        # Removed: self.get_logger().debug(f"EN Prompt tokenization time: {time.time() - prompt_inputs.input_ids.νας:.4f} sec")

        with torch.inference_mode():
            generation = self.model.generate(
                input_ids=self.inputs.input_ids,
                attention_mask=self.inputs.attention_mask,
                prompt_input_ids=prompt_inputs.input_ids,
                prompt_attention_mask=prompt_inputs.attention_mask,
            )
        # Removed: self.get_logger().debug(f"EN Model generation time: {time.time() - generation.sequences.νας:.4f} sec")

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
            # Removed: self.get_logger().debug(f"EN WAV buffer write time: {time.time() - buffer.getbuffer().nbytes / (sampling_rate*2) if sampling_rate > 0 else 0 :.4f} sec")
            return play_time, buffer
        except Exception as e:
            self.get_logger().error(f"Error writing EN WAV to buffer: {e}")
            return 0.0, None

    def tts_ja(self, text):
        prompt = add_ruby(text)
        # Removed: self.get_logger().debug(f"JA Ruby processing time: {time.time() - len(prompt)/1000 if len(prompt)>0 else 0 :.4f} sec")

        prompt_inputs = self.prompt_tokenizer(prompt, return_tensors="pt").to(self.device)
        # Removed: self.get_logger().debug(f"JA Prompt tokenization time: {time.time() - prompt_inputs.input_ids.νας:.4f} sec")

        with torch.inference_mode():
            generation = self.model.generate(
                input_ids=self.inputs.input_ids,
                attention_mask=self.inputs.attention_mask,
                prompt_input_ids=prompt_inputs.input_ids,
                prompt_attention_mask=prompt_inputs.attention_mask,
            )
        # Removed: self.get_logger().debug(f"JA Model generation time: {time.time() - generation.sequences.νας:.4f} sec")

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
            # Removed: self.get_logger().debug(f"JA WAV buffer write time: {time.time() - buffer.getbuffer().nbytes / (sampling_rate*2) if sampling_rate > 0 else 0 :.4f} sec")
            return play_time, buffer
        except Exception as e:
            self.get_logger().error(f"Error writing JA WAV to buffer: {e}")
            return 0.0, None


    def execute_callback(self, goal_handle):
        # Removed: self.get_logger().debug(f"Callback temp node creation time: {time.time() - thread_node._handle:.4f} sec")
        # Create the temporary node *after* the previous logging line is removed
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
            # Ensure temporary node is destroyed even on early exit
            thread_node.destroy_node()
            del thread_node
            return response

        if not decoded_text or not decoded_text.strip():
            self.get_logger().error("Input text is empty or blank.")
            response.success = False
            goal_handle.abort()
             # Ensure temporary node is destroyed even on early exit
            thread_node.destroy_node()
            del thread_node
            return response

        self.get_logger().info(f"Input text: [{decoded_text}]")
        self.get_logger().debug(f"Processing ParlerTTS request for: '{decoded_text}'")

        response.success = False
        response.total_time = 0.0
        play_time = 0.0
        audio_buffer = None

        if self.language == "en":
            play_time, audio_buffer = self.tts_en(decoded_text)
        elif self.language == "ja":
            play_time, audio_buffer = self.tts_ja(decoded_text)
        # Removed: self.get_logger().info(f"Total audio synthesis time: {synthesis_end_time - synthesis_start_time:.4f} sec")

        if audio_buffer is None or play_time <= 0:
            self.get_logger().error("Audio buffer generation failed or invalid play time.")
            response.success = False
            goal_handle.abort()
            # Ensure temporary node is destroyed even on early exit
            thread_node.destroy_node()
            del thread_node
            return response

        try:
            pygame.mixer.init()
            pygame.mixer.music.load(audio_buffer)

            pygame.mixer.music.play()
            play_signal_time = time.time()
            self.get_logger().info(f"Time from request to speech: {play_signal_time - request_process_start_time:.4f} seconds")

            feedback.remaining_time = play_time
            start_playback_loop_time = time.time()

            while rclpy.ok() and pygame.mixer.music.get_busy():
                if goal_handle.is_cancel_requested:
                    self.get_logger().info('Goal canceled during playback.')
                    pygame.mixer.music.stop()
                    goal_handle.canceled()
                    response.success = False
                    break

                rclpy.spin_once(thread_node, timeout_sec=0.01)

                current_time_in_loop = time.time()
                elapsed_in_loop = current_time_in_loop - start_playback_loop_time
                response.total_time = elapsed_in_loop
                feedback.remaining_time = play_time - elapsed_in_loop

                if feedback.remaining_time < 0:
                    feedback.remaining_time = 0.0

                goal_handle.publish_feedback(feedback)

                if feedback.remaining_time <= 0:
                    # Add a small sleep to prevent a tight loop immediately after playback finishes
                    # This also gives pygame a moment to update get_busy() state
                    time.sleep(0.05)
                    if not pygame.mixer.music.get_busy():
                         break


            if not goal_handle.is_cancel_requested:
                if pygame.mixer.music.get_busy():
                     pygame.mixer.music.stop()
                     self.get_logger().warn("Playback loop ended but music was still busy. Stopped.")

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
            if pygame.mixer.get_init():
                pygame.mixer.quit()

            # Ensure temporary node is destroyed before the callback returns
            thread_node.destroy_node()
            del thread_node
            # Removed: self.get_logger().debug(f"Callback temp node destruction time: {time.time() - cb_node_create_start_time:.4f} sec")
            # Removed: self.get_logger().info(f"Total execute_callback processing time: {final_request_process_time - request_process_start_time:.4f} sec")


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