import sys
import time
from config.settings import Settings
from core.llm_provider import LLMProvider

def main():
    print("--- Bob API Integration Tester ---")
    print("Loading settings...")
    
    settings = Settings()
    
    print(f"  PRIMARY_PROVIDER : {settings.PRIMARY_PROVIDER}")
    print(f"  BOB_MODEL        : {settings.BOB_MODEL}")
    print(f"  BOBSHELL_API_KEY : {'[SET]' if settings.BOBSHELL_API_KEY else '[MISSING]'}")
    
    if not settings.BOBSHELL_API_KEY:
        print("\n[Error] BOBSHELL_API_KEY is not set in your .env file!")
        sys.exit(1)
        
    print("\nInitializing LLMProvider...")
    provider = LLMProvider(settings)
    
    prompt = "Explain what an API is in one short sentence."
    print(f"\nSending test prompt to Bob: '{prompt}'")
    print("Waiting for response (this runs the Bob CLI under the hood)...")
    
    t0 = time.perf_counter()
    try:
        response = provider.generate(prompt=prompt)
        latency = time.perf_counter() - t0
        
        print("\n=== SUCCESS ===")
        print(f"Provider Used      : {response.provider_used}")
        print(f"Fallback Triggered : {response.fallback_triggered}")
        print(f"Latency            : {latency:.2f} seconds")
        print("\nResponse Text:")
        print("-" * 50)
        print(response.text)
        print("-" * 50)
        
    except Exception as exc:
        latency = time.perf_counter() - t0
        print(f"\n=== FAILURE === (took {latency:.2f} seconds)")
        print(f"Error: {exc}")
        print("\nTroubleshooting:")
        print("1. Verify your network/internet connection.")
        print("2. Run 'bob --help' in terminal to make sure Bob CLI is installed and in your PATH.")
        print("3. Check if your BOBSHELL_API_KEY is valid.")

if __name__ == "__main__":
    main()
