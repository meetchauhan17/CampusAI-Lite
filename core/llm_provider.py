import time
from typing import Any, List, Optional, Callable, Dict, Union
from pydantic import BaseModel, Field
from config.settings import Settings
from core.logger import logger
from tenacity import Retrying, AsyncRetrying, stop_after_attempt, wait_exponential, retry_if_exception
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.outputs import ChatResult, ChatGeneration

# Custom exceptions
class CampusAIProviderError(Exception):
    """Raised when all configured LLM providers fail."""
    def __init__(self, message: str, errors: Dict[str, str]):
        super().__init__(message)
        self.errors = errors

# Response Schema
class LLMResponse(BaseModel):
    text: str = Field(..., description="The generated completion text.")
    provider_used: str = Field(..., description="The name of the provider that succeeded ('bob', 'groq', 'gemini').")
    latency_ms: float = Field(..., description="The request latency in milliseconds.")
    tokens_used: Optional[int] = Field(None, description="The total number of tokens used (best-effort).")
    fallback_triggered: bool = Field(..., description="True if a fallback provider was used instead of the primary provider.")

# Exception filter for tenacity retries
def should_retry_exception(exc: Exception) -> bool:
    """
    Decide if we should retry the exception.
    Only retry on connection/rate-limit errors — not on authentication or client validation errors.
    """
    import requests
    if isinstance(exc, (requests.exceptions.ConnectionError, requests.exceptions.Timeout)):
        return True

    # Check for Groq-specific exceptions
    try:
        import groq
        if isinstance(exc, groq.APIConnectionError):
            return True
        if isinstance(exc, groq.RateLimitError):
            return True
        if isinstance(exc, groq.APIStatusError):
            return exc.status_code == 429 or exc.status_code >= 500
    except ImportError:
        pass

    # Check for Gemini/Google-specific exceptions
    try:
        from google.api_core.exceptions import GoogleAPICallError, ResourceExhausted, ServiceUnavailable
        if isinstance(exc, (ResourceExhausted, ServiceUnavailable)):
            return True
        if isinstance(exc, GoogleAPICallError):
            return exc.code == 429 or (exc.code and exc.code >= 500)
    except ImportError:
        pass

    # Watsonx / generic exceptions with status codes
    if hasattr(exc, "status_code"):
        code = getattr(exc, "status_code")
        if isinstance(code, int):
            return code == 429 or code >= 500

    # Fallback to string matching on message text
    msg = str(exc).lower()
    if any(k in msg for k in ["rate limit", "429", "timeout", "connection error", "503 service unavailable"]):
        return True

    return False

# Helper for PydanticAI request parsing
def extract_prompt_and_system(request: Any) -> tuple[str, Optional[str]]:
    """
    Helper function to extract user prompt and system prompt from PydanticAI ModelRequest.
    """
    system_prompt = None
    user_prompt = ""

    if hasattr(request, "instructions") and request.instructions:
        system_prompt = request.instructions

    if hasattr(request, "parts"):
        for part in request.parts:
            part_kind = getattr(part, "part_kind", None)
            if part_kind == "system-prompt":
                system_prompt = getattr(part, "content", system_prompt)
            elif part_kind == "user-prompt":
                content = getattr(part, "content", "")
                if isinstance(content, str):
                    user_prompt = content
                elif isinstance(content, (list, tuple)):
                    user_prompt = " ".join(
                        c if isinstance(c, str) else getattr(c, "content", "")
                        for c in content
                    )
            elif part_kind == "text":
                content = getattr(part, "content", "")
                user_prompt += f"\n{content}"

    return user_prompt.strip(), system_prompt


import threading

_bob_lock = threading.Lock()


class LLMProvider:
    """
    Unified LLM Client that manages authentication, lazy loading of SDKs,
    and automatic failover (fallback) between 'bob' (IBM Bob Shell CLI),
    'groq', and 'gemini'.
    """
    def __init__(self, settings: Settings):
        self.settings = settings

        # Lazy client placeholders
        self._groq_client = None
        self._groq_async_client = None
        self._gemini_configured = False

        # Register provider internal methods
        self._sync_providers = {
            "bob": self._call_bob,
            "groq": self._call_groq,
            "gemini": self._call_gemini
        }
        self._async_providers = {
            "bob": self._call_bob_async,
            "groq": self._call_groq_async,
            "gemini": self._call_gemini_async
        }

    def __deepcopy__(self, memo: dict):
        # AG2/AutoGen deep-copies llm_config when constructing ConversableAgent.
        # Our provider holds live httpx clients (CookieJar, RLock) that can't be
        # pickled.  Returning self is safe: the provider is stateless per-call.
        memo[id(self)] = self
        return self

    # Bob Shell subprocess helper
    def _run_bob_subprocess(self, prompt: str) -> str:
        """
        Calls the Bob Shell CLI (bob.cmd) via subprocess, pipes prompt through
        stdin, and extracts the response from ---output--- markers.
        """
        import subprocess, os
        if not self.settings.BOBSHELL_API_KEY:
            raise ValueError("Missing BOBSHELL_API_KEY for IBM Bob Shell provider.")

        env = {**os.environ, "BOBSHELL_API_KEY": self.settings.BOBSHELL_API_KEY}
        raw_model = self.settings.BOB_MODEL or "premium"
        
        # Model normalization mapping (as specified by Bob architecture)
        model_lower = raw_model.lower()
        if "granite" in model_lower:
            model = "gemini-3.5-flash"
        else:
            supported_models = [
                "premium", "premium-ide", "sonnet-4.5", "haiku-4.5",
                "explorer", "gpt-2026-5.4", "gemini-3.5-flash", "granite-8b-code-instruct"
            ]
            model = raw_model if raw_model in supported_models else "premium"

        # Synchronize subprocess calls to prevent git lock file conflicts in parallel threads
        with _bob_lock:
            result = subprocess.run(
                f'bob.cmd --output-format text --model "{model}"',
                input=prompt,
                capture_output=True,
                text=True,
                timeout=120,
                shell=True,
                env=env,
            )
        if result.returncode != 0:
            raise RuntimeError(f"Bob Shell exited {result.returncode}: {result.stderr.strip()}")

        # Extract content between ---output--- markers (odd-indexed segments)
        parts = result.stdout.split("---output---")
        if len(parts) >= 3:
            return "\n".join(p.strip() for p in parts[1::2] if p.strip())
        # Fallback: return all stdout if markers absent
        return result.stdout.strip()

    # Lazy initialization helpers
    def _get_groq_client(self) -> Any:
        if self._groq_client is None:
            if not self.settings.GROQ_API_KEY:
                raise ValueError("Missing GROQ_API_KEY for groq provider.")
            from groq import Groq
            self._groq_client = Groq(api_key=self.settings.GROQ_API_KEY)
        return self._groq_client

    def _get_groq_async_client(self) -> Any:
        if self._groq_async_client is None:
            if not self.settings.GROQ_API_KEY:
                raise ValueError("Missing GROQ_API_KEY for groq provider.")
            from groq import AsyncGroq
            self._groq_async_client = AsyncGroq(api_key=self.settings.GROQ_API_KEY)
        return self._groq_async_client

    def _configure_gemini(self) -> None:
        if not self._gemini_configured:
            if not self.settings.GEMINI_API_KEY:
                raise ValueError("Missing GEMINI_API_KEY for gemini provider.")
            import google.generativeai as genai
            genai.configure(api_key=self.settings.GEMINI_API_KEY)
            self._gemini_configured = True

    # Internal call methods
    def _call_bob(self, prompt: str, system: Optional[str], temperature: float, max_tokens: int) -> tuple[str, Optional[int]]:
        logger.info("[bob] Executing IBM Bob Shell generate call.")
        full_prompt = prompt
        if system:
            full_prompt = f"SYSTEM: {system}\nUSER: {prompt}"
        text = self._run_bob_subprocess(full_prompt)
        return text, None  # Bob Shell does not report token counts

    async def _call_bob_async(self, prompt: str, system: Optional[str], temperature: float, max_tokens: int) -> tuple[str, Optional[int]]:
        logger.info("[bob] Executing async IBM Bob Shell generate call (thread pool).")
        import asyncio
        loop = asyncio.get_event_loop()
        full_prompt = prompt
        if system:
            full_prompt = f"SYSTEM: {system}\nUSER: {prompt}"
        text = await loop.run_in_executor(None, self._run_bob_subprocess, full_prompt)
        return text, None

    def _call_groq(self, prompt: str, system: Optional[str], temperature: float, max_tokens: int) -> tuple[str, Optional[int]]:
        logger.info("[groq] Executing groq generate call.")
        client = self._get_groq_client()
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        model_name = getattr(self.settings, "GROQ_MODEL", "llama-3.3-70b-versatile")
        res = client.chat.completions.create(
            messages=messages,
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens
        )
        text = res.choices[0].message.content
        tokens_used = res.usage.total_tokens if res.usage else None
        return text, tokens_used

    async def _call_groq_async(self, prompt: str, system: Optional[str], temperature: float, max_tokens: int) -> tuple[str, Optional[int]]:
        logger.info("[groq] Executing async groq generate call.")
        client = self._get_groq_async_client()
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        model_name = getattr(self.settings, "GROQ_MODEL", "llama-3.3-70b-versatile")
        res = await client.chat.completions.create(
            messages=messages,
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens
        )
        text = res.choices[0].message.content
        tokens_used = res.usage.total_tokens if res.usage else None
        return text, tokens_used

    def _call_gemini(self, prompt: str, system: Optional[str], temperature: float, max_tokens: int) -> tuple[str, Optional[int]]:
        logger.info("[gemini] Executing gemini generate call.")
        self._configure_gemini()
        import google.generativeai as genai
        
        model_name = getattr(self.settings, "GEMINI_MODEL", "gemini-2.0-flash")
        model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=system
        )
        config = genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens
        )
        res = model.generate_content(prompt, generation_config=config)
        text = res.text
        tokens_used = res.usage_metadata.total_token_count if (hasattr(res, "usage_metadata") and res.usage_metadata) else None
        return text, tokens_used

    async def _call_gemini_async(self, prompt: str, system: Optional[str], temperature: float, max_tokens: int) -> tuple[str, Optional[int]]:
        logger.info("[gemini] Executing async gemini generate call.")
        self._configure_gemini()
        import google.generativeai as genai
        
        model_name = getattr(self.settings, "GEMINI_MODEL", "gemini-2.0-flash")
        model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=system
        )
        config = genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens
        )
        res = await model.generate_content_async(prompt, generation_config=config)
        text = res.text
        tokens_used = res.usage_metadata.total_token_count if (hasattr(res, "usage_metadata") and res.usage_metadata) else None
        return text, tokens_used

    # Public methods
    def generate(
        self,
        prompt: str,
        system: str = None,
        temperature: float = 0.3,
        max_tokens: int = 1024
    ) -> LLMResponse:
        """
        Synchronously generate completion text, using configured fallback order on failures.
        """
        fallback_order = self.settings.FALLBACK_ORDER
        primary_provider = fallback_order[0] if fallback_order else "bob"
        errors = {}

        retryer = Retrying(
            stop=stop_after_attempt(2),
            wait=wait_exponential(multiplier=1, min=2, max=10),
            retry=retry_if_exception(should_retry_exception),
            reraise=True
        )

        for provider in fallback_order:
            if provider not in self._sync_providers:
                logger.warning("Unknown provider '{}' in fallback order.", provider)
                continue

            logger.info("Attempting generation using provider: {}", provider)
            start_time = time.perf_counter()

            try:
                # Execute provider call wrapped with tenacity retry
                text, tokens_used = retryer(
                    self._sync_providers[provider],
                    prompt,
                    system,
                    temperature,
                    max_tokens
                )
                latency = (time.perf_counter() - start_time) * 1000.0
                fallback_triggered = (provider != primary_provider)

                logger.info("Generation successful using provider: {} (latency: {:.2f}ms)", provider, latency)
                return LLMResponse(
                    text=text,
                    provider_used=provider,
                    latency_ms=latency,
                    tokens_used=tokens_used,
                    fallback_triggered=fallback_triggered
                )
            except Exception as e:
                latency = (time.perf_counter() - start_time) * 1000.0
                logger.warning(
                    "Provider '{}' failed after retries. Error: {} (latency: {:.2f}ms)",
                    provider,
                    str(e),
                    latency
                )
                errors[provider] = f"{type(e).__name__}: {str(e)}"

        raise CampusAIProviderError(
            f"All LLM providers in fallback order failed. Errors: {errors}",
            errors=errors
        )

    async def generate_async(
        self,
        prompt: str,
        system: str = None,
        temperature: float = 0.3,
        max_tokens: int = 1024
    ) -> LLMResponse:
        """
        Asynchronously generate completion text, using configured fallback order on failures.
        """
        fallback_order = self.settings.FALLBACK_ORDER
        primary_provider = fallback_order[0] if fallback_order else "bob"
        errors = {}

        async_retryer = AsyncRetrying(
            stop=stop_after_attempt(2),
            wait=wait_exponential(multiplier=1, min=2, max=10),
            retry=retry_if_exception(should_retry_exception),
            reraise=True
        )

        for provider in fallback_order:
            if provider not in self._async_providers:
                logger.warning("Unknown async provider '{}' in fallback order.", provider)
                continue

            logger.info("Attempting async generation using provider: {}", provider)
            start_time = time.perf_counter()

            try:
                # Execute provider call wrapped with tenacity async retry
                text, tokens_used = await async_retryer(
                    self._async_providers[provider],
                    prompt,
                    system,
                    temperature,
                    max_tokens
                )
                latency = (time.perf_counter() - start_time) * 1000.0
                fallback_triggered = (provider != primary_provider)

                logger.info("Async generation successful using provider: {} (latency: {:.2f}ms)", provider, latency)
                return LLMResponse(
                    text=text,
                    provider_used=provider,
                    latency_ms=latency,
                    tokens_used=tokens_used,
                    fallback_triggered=fallback_triggered
                )
            except Exception as e:
                latency = (time.perf_counter() - start_time) * 1000.0
                logger.warning(
                    "Async provider '{}' failed after retries. Error: {} (latency: {:.2f}ms)",
                    provider,
                    str(e),
                    latency
                )
                errors[provider] = f"{type(e).__name__}: {str(e)}"

        raise CampusAIProviderError(
            f"All LLM providers in fallback order failed in async call. Errors: {errors}",
            errors=errors
        )

    # Adapter Wrappers
    def to_langchain_chat_model(self, temperature: float = 0.3, max_tokens: int = 1024) -> BaseChatModel:
        """
        Returns a LangChain BaseChatModel adapter wrapping this LLMProvider instance.
        """
        return CampusAIChatModel(provider=self, temperature=temperature, max_tokens=max_tokens)

    def to_pydantic_ai_callable(self) -> Callable[[Any], Any]:
        """
        Returns a plain async callable suitable for PydanticAI's FunctionModel wrapper.
        The callable takes a ModelRequest and returns a ModelResponse.
        """
        from pydantic_ai.messages import ModelResponse, TextPart
        
        async def call_pydantic_ai(request: Any) -> ModelResponse:
            user_prompt, system_prompt = extract_prompt_and_system(request)
            res = await self.generate_async(prompt=user_prompt, system=system_prompt)
            return ModelResponse(parts=[TextPart(content=res.text)])
            
        return call_pydantic_ai


# Custom LangChain BaseChatModel Implementation
class CampusAIChatModel(BaseChatModel):
    provider: Any = Field(..., description="The underlying LLMProvider instance.")
    temperature: float = 0.3
    max_tokens: int = 1024

    @property
    def _llm_type(self) -> str:
        return "campusai-provider"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        from langchain_core.messages import SystemMessage, HumanMessage
        from langchain_core.outputs import ChatGeneration

        system_prompt = None
        user_prompt = ""
        
        for msg in messages:
            if isinstance(msg, SystemMessage):
                system_prompt = msg.content
            elif isinstance(msg, HumanMessage):
                user_prompt = msg.content
            else:
                user_prompt += f"\n{msg.content}"

        res = self.provider.generate(
            prompt=user_prompt,
            system=system_prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )

        message = AIMessage(content=res.text)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        from langchain_core.messages import SystemMessage, HumanMessage
        from langchain_core.outputs import ChatGeneration

        system_prompt = None
        user_prompt = ""

        for msg in messages:
            if isinstance(msg, SystemMessage):
                system_prompt = msg.content
            elif isinstance(msg, HumanMessage):
                user_prompt = msg.content
            else:
                user_prompt += f"\n{msg.content}"

        res = await self.provider.generate_async(
            prompt=user_prompt,
            system=system_prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )

        message = AIMessage(content=res.text)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])
