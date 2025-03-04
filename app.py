import gradio as gr
from google import genai
from smolagents import DuckDuckGoSearchTool

search_tool = DuckDuckGoSearchTool()


def search_answer(question: str, api_key: str) -> str:
    """Fetches search results and generates an answer using Gemini AI."""
    result = search_tool(question)
    prompt = f"""
    Based on the search results, please provide a concise answer to the following question:

    Question: {question}
    Search Results: {result}

    Please ensure your response is:
    1. Directly based on the search results
    2. Brief and to the point
    """
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model='gemini-2.0-flash',
        contents=[prompt]
    )
    return response.text if response else "No response generated."

# Improved Gradio Interface with Better UI
custom_css = """
    .gradio-container {
        background: linear-gradient(to bottom right, #1a1a2e, #16213e);
        color: #e6e6e6;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .gradio-container h1, .gradio-container h2, .gradio-container h3 {
        color: #4cc9f0;
        font-weight: 600;
        text-shadow: 0px 2px 4px rgba(0, 0, 0, 0.3);
    }
    .gradio-container .prose p {
        color: #b8c1ec;
        font-size: 1.1em;
        margin-bottom: 1.5em;
    }
    .gradio-container input, .gradio-container textarea {
        background-color: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        color: #000000 !important;
        border-radius: 8px !important;
        transition: all 0.3s ease;
        font-weight: 700 !important;
    }
    .gradio-container input:focus, .gradio-container textarea:focus {
        border-color: #4cc9f0 !important;
        box-shadow: 0 0 0 2px rgba(76, 201, 240, 0.25) !important;
    }
    .gradio-container button {
        background: linear-gradient(to right, #4361ee, #4cc9f0) !important;
        border: none !important;
        color: white !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }
    .gradio-container button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(76, 201, 240, 0.3) !important;
    }
    .gradio-container .output-text {
        background-color: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 8px !important;
        padding: 15px !important;
        color: #ffffff !important;
        font-weight: 500 !important;
    }
    .footer {
        margin-top: 20px;
        text-align: center;
        color: #b8c1ec;
        font-size: 0.9em;
    }
    .container {
        max-width: 900px;
        margin: 0 auto;
        padding: 20px;
    }
    .header-icon {
        font-size: 2.5em;
        margin-bottom: 10px;
        background: linear-gradient(to right, #4361ee, #4cc9f0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
"""

# Add API key input field
api_key = gr.Textbox(
    placeholder="Enter your Gemini API Key",
    label="🔑 API Key",
    type="password",
    interactive=True,  
    elem_id="api-key-input"
)

demo = gr.Interface(
    fn=search_answer,
    inputs=[
        gr.Textbox(
            label="🔍 Ask a Question", 
            placeholder="Type your question here...", 
            lines=3,
            elem_id="question-input"
        ),
        api_key
    ],
    outputs=gr.Textbox(
        label="🤖 AI Answer", 
        interactive=False,
        elem_id="answer-output",
        elem_classes=["output-text"]
    ),
    title="✨ Smart AI Search Assistant",
    description="""<div class='header-icon'>🔮</div>
    <p>Enter your question below and get a concise answer based on real-time web search results.</p>
    <p>Our AI assistant will analyze the search data and provide you with the most relevant information.</p>""",
    article="""<div class='footer'>
    <p>Powered by Gemini AI and DuckDuckGo Search | Created with ❤️</p>
    </div>""",
    css=custom_css,
    theme=gr.themes.Base(),
    allow_flagging="never"
)

demo.launch(share=True)
