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
import os
import soundfile as sf
import tempfile
import time

class TTSInference(Node):
    def __init__(self, device=None):
        super().__init__('tts_inference_server')
        """
        コンストラクタ。

        Args:
            language (str, optional): 使用する言語 ("en" または "ja")。デフォルトは "ja"。
            device (str, optional): 処理を実行するデバイス (例: 'cuda:0', 'cpu')。
                デフォルトは None (利用可能な場合は CUDA、それ以外は CPU)。
        """
        # 並列処理に関する警告を抑制
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.language = "ja"
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
        self._setup()

        if self.language == "ja":
            prompt = "こんにちは、今日はご機嫌いかがかしら？"
            prompt = add_ruby(prompt)
        else:
            prompt = "Hello, can you hear me?"
        description = "Jenna delivers a slightly expressive and animated speech with a moderate speed and pitch. The recording is of very high quality, with the speaker's voice sounding clear and very close up."
        self.run(prompt, description)


    def _setup(self):
        """
        モデルとトークナイザーのロード、言語設定を行う内部メソッド。
        """
        start_time = time.time()
        model_config = self.language_config.get(self.language)
        if not model_config:
            raise ValueError(f"Unsupported language: {self.language}")
        self.model = ParlerTTSForConditionalGeneration.from_pretrained(model_config["model_name"]).to(self.device)

        if self.language == "en":
            self.tokenizer = AutoTokenizer.from_pretrained(model_config["tokenizer_name"])
            self.tokenizer.pad_token_id = self.model.config.pad_token_id
        elif self.language == "ja":
            self.prompt_tokenizer = AutoTokenizer.from_pretrained(
                model_config["prompt_tokenizer_name"], subfolder="prompt_tokenizer"
            )
            self.description_tokenizer = AutoTokenizer.from_pretrained(
                model_config["description_tokenizer_name"], subfolder="description_tokenizer"
            )
            self.prompt_tokenizer.pad_token_id = self.model.config.pad_token_id
            self.description_tokenizer.pad_token_id = self.model.config.pad_token_id
        end_time = time.time()
        print(f"モデルとトークナイザーのロード: {end_time - start_time:.4f} 秒")

    def _prepare_inputs(self, description, prompt):
        """
        入力を準備する内部メソッド。

        Args:
            description (str): 話者記述テキスト。
            prompt (str): 発話テキスト。

        Returns:
            tuple: 入力テンソルとプロンプト入力テンソルのタプル。
        """
        start_time = time.time()
        if self.language == "en":
            inputs = self.tokenizer(description, return_tensors="pt").to(self.device)
            prompt_inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        elif self.language == "ja":
            inputs = self.description_tokenizer(description, return_tensors="pt").to(self.device)
            prompt_inputs = self.prompt_tokenizer(prompt, return_tensors="pt").to(self.device)
        end_time = time.time()
        print(f"トークナイザーの入力準備: {end_time - start_time:.4f} 秒")
        return inputs, prompt_inputs

    def _process_and_play_audio(self, input_ids, attention_mask, prompt_input_ids, prompt_attention_mask):
        """
        音声データを生成し、再生する内部メソッド。

        Args:
            input_ids (torch.Tensor): 入力IDテンソル。
            attention_mask (torch.Tensor): アテンションマスクテンソル。
            prompt_input_ids (torch.Tensor): プロンプト入力IDテンソル。
            prompt_attention_mask (torch.Tensor): プロンプトアテンションマスクテンソル。
        """
        start_time = time.time()
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
            pygame.mixer.init()
            pygame.mixer.music.load(tmpfile.name)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
        end_time = time.time()
        print(f"音声生成と再生処理: {end_time - start_time:.4f} 秒")
        print("音声再生が完了しました。")

    def run(self, prompt, description):
        """
        TTS推論処理を実行するメソッド。

        Args:
            prompt (str): 発話テキスト。
            description (str): 話者記述テキスト。
        """
        start_time = time.time()
        inputs, prompt_inputs = self._prepare_inputs(description, prompt)
        self._process_and_play_audio(
            inputs.input_ids,
            inputs.attention_mask,
            prompt_input_ids=prompt_inputs.input_ids,
            prompt_attention_mask=prompt_inputs.attention_mask,
        )
        end_time = time.time()
        print(f"TTS推論処理全体: {end_time - start_time:.4f} 秒")

def main(args=None):
    rclpy.init(args=args)

    server = TTSInference()
    rclpy.spin(server)
    rclpy.shutdown()

if __name__ == "__main__":
    main()
