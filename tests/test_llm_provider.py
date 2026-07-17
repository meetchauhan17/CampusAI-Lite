import unittest
from unittest.mock import patch, MagicMock
from config.settings import Settings
from core.llm_provider import LLMProvider, LLMResponse, CampusAIProviderError

class TestLLMProvider(unittest.TestCase):
    def setUp(self):
        self.settings = Settings(
            BOBSHELL_API_KEY="mock-bob-key",
            BOB_MODEL="premium",
            GROQ_API_KEY="mock-groq-key",
            GEMINI_API_KEY="mock-gemini-key",
            PRIMARY_PROVIDER="bob",
            FALLBACK_ORDER=["bob", "groq", "gemini"]
        )
        self.provider = LLMProvider(self.settings)

    @patch("subprocess.run")
    def test_primary_bob_success(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="---output---\nHello from Bob\n---output---",
            stderr=""
        )
        response = self.provider.generate("Test prompt")

        self.assertEqual(response.text, "Hello from Bob")
        self.assertEqual(response.provider_used, "bob")
        self.assertIsNone(response.tokens_used)
        self.assertFalse(response.fallback_triggered)
        mock_run.assert_called_once()

    @patch("subprocess.run")
    @patch("groq.Groq")
    def test_fallback_to_groq(self, mock_groq_class, mock_run):
        import requests
        # Bob fails with a connection error
        mock_run.side_effect = requests.exceptions.ConnectionError("Bob unavailable")

        # Groq succeeds
        mock_groq_client = MagicMock()
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock(message=MagicMock(content="Hello from Groq"))]
        mock_completion.usage = MagicMock(total_tokens=25)
        mock_groq_client.chat.completions.create.return_value = mock_completion
        mock_groq_class.return_value = mock_groq_client

        with patch("tenacity.nap.time.sleep", return_value=None):
            response = self.provider.generate("Test prompt")

        self.assertEqual(response.text, "Hello from Groq")
        self.assertEqual(response.provider_used, "groq")
        self.assertTrue(response.fallback_triggered)
        self.assertEqual(mock_run.call_count, 2)  # Bob retried twice
        mock_groq_client.chat.completions.create.assert_called_once()

    @patch("subprocess.run")
    @patch("groq.Groq")
    @patch("google.generativeai.GenerativeModel")
    @patch("google.generativeai.configure")
    def test_all_providers_fail(self, mock_gemini_config, mock_gemini_model, mock_groq_class, mock_run):
        mock_run.side_effect = ValueError("Bob error")

        mock_groq_client = MagicMock()
        mock_groq_client.chat.completions.create.side_effect = ValueError("Groq error")
        mock_groq_class.return_value = mock_groq_client

        mock_gemini_instance = MagicMock()
        mock_gemini_instance.generate_content.side_effect = ValueError("Gemini error")
        mock_gemini_model.return_value = mock_gemini_instance

        with patch("tenacity.nap.time.sleep", return_value=None):
            with self.assertRaises(CampusAIProviderError) as context:
                self.provider.generate("Test prompt")

        errors = context.exception.errors
        self.assertIn("bob", errors)
        self.assertIn("groq", errors)
        self.assertIn("gemini", errors)

if __name__ == "__main__":
    unittest.main()
