import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock modules before import
sys.modules['numpy'] = MagicMock()
sys.modules['transformers'] = MagicMock()
sys.modules['vllm'] = MagicMock()

from src.inference import AudioLLMEngine


class TestAudioLLMEngine(unittest.TestCase):
    def setUp(self):
        sys.modules['numpy'].reset_mock()
        sys.modules['transformers'].reset_mock()
        sys.modules['vllm'].reset_mock()
        
    @patch('src.inference.asdict')
    def test_initialization(self, mock_asdict):
        mock_asdict.return_value = {}
        mock_processor = sys.modules['transformers'].AutoProcessor
        mock_processor_instance = MagicMock()
        mock_processor.from_pretrained.return_value = mock_processor_instance
        
        engine = AudioLLMEngine(model_id="test-model")
        
        sys.modules['vllm'].LLM.assert_called_once()
        mock_processor.from_pretrained.assert_called_with("test-model", trust_remote_code=True)

    @patch('src.inference.asdict')
    def test_generate_text_only(self, mock_asdict):
        mock_asdict.return_value = {}
        mock_llm_class = sys.modules['vllm'].LLM
        mock_llm_instance = MagicMock()
        mock_llm_class.return_value = mock_llm_instance
        
        mock_output = MagicMock()
        mock_output.outputs = [MagicMock(text="Test response")]
        mock_llm_instance.generate.return_value = [mock_output]
        
        mock_processor = sys.modules['transformers'].AutoProcessor
        mock_processor.from_pretrained.return_value = MagicMock()

        engine = AudioLLMEngine(model_id="test-model")
        responses = engine.generate(["Hello"])
        
        self.assertEqual(responses, ["Test response"])
        
    @patch('src.inference.asdict')
    def test_generate_with_audio(self, mock_asdict):
        mock_asdict.return_value = {}
        mock_llm_class = sys.modules['vllm'].LLM
        mock_llm_instance = MagicMock()
        mock_llm_class.return_value = mock_llm_instance
         
        mock_output = MagicMock()
        mock_output.outputs = [MagicMock(text="Audio response")]
        mock_llm_instance.generate.return_value = [mock_output]
        
        mock_processor = sys.modules['transformers'].AutoProcessor
        mock_processor_instance = MagicMock()
        mock_processor.from_pretrained.return_value = mock_processor_instance
        mock_processor_instance.apply_chat_template.return_value = "Formatted Prompt"

        engine = AudioLLMEngine(model_id="test-model")
        
        audio_sample = MagicMock()
        audios = [[audio_sample]]
        
        responses = engine.generate(["Describe"], audios=audios)
        
        self.assertEqual(responses, ["Audio response"])
        
        call_args = mock_llm_instance.generate.call_args
        batch_inputs = call_args[0][0]
        
        self.assertEqual(len(batch_inputs), 1)
        self.assertEqual(batch_inputs[0]["prompt"], "Formatted Prompt")
        self.assertIn("multi_modal_data", batch_inputs[0])
        self.assertIn("audio", batch_inputs[0]["multi_modal_data"])


if __name__ == '__main__':
    unittest.main()
