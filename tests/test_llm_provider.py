import unittest
from unittest.mock import patch, MagicMock
from config.settings import Settings
from core.llm_provider import LLMProvider, LLMResponse, CampusAIProviderError

class TestLLMProvider(unittest.TestCase):
    def setUp(self):
        # Set up settings with mock API keys to avoid external checks
        self.settings = Settings(
            BOBSHELL_API_KEY="mock-bob-key",
            BOB_PROJECT_ID="mock-bob-project-id",
            BOB_URL="https://us-south.ml.cloud.ibm.com",
            BOB_MODEL="ibm/granite-3-8b-instruct",
            GROQ_API_KEY="mock-groq-key",
            GEMINI_API_KEY="mock-gemini-key",
            PRIMARY_PROVIDER="bob",
            FALLBACK_ORDER=["bob", "groq", "gemini"]
        )
        self.provider = LLMProvider(self.settings)

    @patch("ibm_watsonx_ai.APIClient")
    @patch("ibm_watsonx_ai.foundation_models.ModelInference")
    def test_primary_bob_success(self, mock_model_inference, mock_api_client):
        # Mock watsonx.ai ModelInference generate response
        mock_instance = MagicMock()
        mock_instance.generate.return_value = {
            "results": [
                {
                    "generated_text": "Hello from Bob",
                    "input_token_count": 12,
                    "generated_token_count": 8
                }
            ]
        }
        mock_model_inference.return_value = mock_instance

        # Call generate
        response = self.provider.generate("Test prompt")

        # Assertions
        self.assertEqual(response.text, "Hello from Bob")
        self.assertEqual(response.provider_used, "bob")
        self.assertEqual(response.tokens_used, 20)
        self.assertFalse(response.fallback_triggered)
        mock_instance.generate.assert_called_once()

    @patch("ibm_watsonx_ai.APIClient")
    @patch("ibm_watsonx_ai.foundation_models.ModelInference")
    @patch("groq.Groq")
    def test_fallback_to_groq(self, mock_groq_class, mock_model_inference, mock_api_client):
        # Mock Bob to fail with a retryable connection error
        import requests
        mock_bob_instance = MagicMock()
        mock_bob_instance.generate.side_effect = requests.exceptions.ConnectionError("Watsonx endpoint unavailable")
        mock_model_inference.return_value = mock_bob_instance

        # Mock Groq to succeed
        mock_groq_instance = MagicMock()
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock(message=MagicMock(content="Hello from Groq"))]
        mock_completion.usage = MagicMock(total_tokens=25)
        mock_groq_instance.chat.completions.create.return_value = mock_completion
        mock_groq_class.return_value = mock_groq_instance

        # Patch sleep to make tests run instantly
        with patch("tenacity.nap.time.sleep", return_value=None):
            response = self.provider.generate("Test prompt")

        # Assertions
        self.assertEqual(response.text, "Hello from Groq")
        self.assertEqual(response.provider_used, "groq")
        self.assertEqual(response.tokens_used, 25)
        self.assertTrue(response.fallback_triggered)
        
        # Verify bob was retried (called twice)
        self.assertEqual(mock_bob_instance.generate.call_count, 2)
        # Verify groq was called once
        mock_groq_instance.chat.completions.create.assert_called_once()

    @patch("ibm_watsonx_ai.APIClient")
    @patch("ibm_watsonx_ai.foundation_models.ModelInference")
    @patch("groq.Groq")
    @patch("google.generativeai.GenerativeModel")
    @patch("google.generativeai.configure")
    def test_all_providers_fail(self, mock_gemini_config, mock_gemini_model, mock_groq_class, mock_model_inference, mock_api_client):
        # Mock Bob to fail (non-retryable auth exception for test convenience)
        mock_bob_instance = MagicMock()
        mock_bob_instance.generate.side_effect = ValueError("Invalid API key")
        mock_model_inference.return_value = mock_bob_instance

        # Mock Groq to fail
        mock_groq_instance = MagicMock()
        mock_groq_instance.chat.completions.create.side_effect = ValueError("Groq rate limit exceeded")
        mock_groq_class.return_value = mock_groq_instance

        # Mock Gemini to fail
        mock_gemini_instance = MagicMock()
        mock_gemini_instance.generate_content.side_effect = ValueError("Gemini internal error")
        mock_gemini_model.return_value = mock_gemini_instance

        # Call generate and assert exception is raised
        with patch("tenacity.nap.time.sleep", return_value=None):
            with self.assertRaises(CampusAIProviderError) as context:
                self.provider.generate("Test prompt")

        # Verify the errors are collected
        errors = context.exception.errors
        self.assertIn("bob", errors)
        self.assertIn("groq", errors)
        self.assertIn("gemini", errors)
        self.assertIn("ValueError", errors["bob"])
        self.assertIn("ValueError", errors["groq"])
        self.assertIn("ValueError", errors["gemini"])

if __name__ == "__main__":
    unittest.main()
