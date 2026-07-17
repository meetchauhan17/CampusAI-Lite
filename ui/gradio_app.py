import gradio as gr
from config.settings import Settings
from core.logger import logger
from core.llm_provider import LLMProvider

# Initialize configuration and provider
settings = Settings()
provider = LLMProvider(settings)

def handle_query(query: str, framework: str) -> str:
    """
    Process the user query using the selected agentic framework.
    """
    logger.info("Request received. Query: '{}' | Framework: '{}'", query, framework)
    
    try:
        # Using LLMProvider (which mocks primary provider "bob" / watsonx.ai)
        response = provider.generate(query)
        logger.info("Request completed successfully via {}", settings.PRIMARY_PROVIDER)
        return f"[{framework.upper()}] Assistant Response:\n\n{response}\n\n(Generated using primary model: {settings.BOB_MODEL})"
    except Exception as e:
        logger.error("Failed to execute query: {}", e)
        return f"Error executing query: {str(e)}"

# Define the Gradio Interface
with gr.Blocks(title="CampusAI-Lite Assistant") as app:
    gr.Markdown("# 🎓 CampusAI-Lite")
    gr.Markdown(
        "Welcome to **CampusAI-Lite**, an agentic university information assistant designed for "
        "the Capstone project comparing CrewAI, LangGraph, AutoGen, and BeeAI."
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            query_input = gr.Textbox(
                label="Search Document / Ask Question",
                placeholder="e.g., What are the enrollment prerequisites for AI courses?",
                lines=3
            )
            framework_select = gr.Dropdown(
                choices=["crewai", "langgraph", "autogen", "beeai"],
                value="langgraph",
                label="Select Framework Implementation"
            )
            submit_btn = gr.Button("Query Assistant", variant="primary")
            
        with gr.Column(scale=1):
            result_output = gr.Textbox(
                label="Result",
                interactive=False,
                lines=7
            )
            
    submit_btn.click(
        fn=handle_query,
        inputs=[query_input, framework_select],
        outputs=result_output
    )

if __name__ == "__main__":
    # Print warnings for missing API keys
    settings.validate_keys()
    
    logger.info("Starting Gradio App server...")
    app.launch(server_name="127.0.0.1", server_port=7860, share=False)
