import os
from datetime import datetime, timedelta, timezone

import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI
from tavily import TavilyClient


# ------------------------------------------------------------
# Environment setup
# ------------------------------------------------------------

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError(
        "OPENAI_API_KEY is not set. Add it to your .env file or deployment secrets."
    )

if not TAVILY_API_KEY:
    raise ValueError(
        "TAVILY_API_KEY is not set. Add it to your .env file or deployment secrets."
    )

openai_client = OpenAI(api_key=OPENAI_API_KEY)
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)


# ------------------------------------------------------------
# Helper function for date handling
# ------------------------------------------------------------

def normalize_date_for_search(date: str) -> str:
    """
    Converts 'yesterday' into a YYYY-MM-DD date string.
    Tavily is a web search tool, so this date is used as part of the search query,
    not as a strict database-style filter.
    """

    date = date.strip()

    if date.lower() == "yesterday":
        yesterday = datetime.now(timezone.utc) - timedelta(days=1)
        return yesterday.strftime("%Y-%m-%d")

    # Validate YYYY-MM-DD format
    datetime.strptime(date, "%Y-%m-%d")
    return date


# ------------------------------------------------------------
# Reddit search using Tavily
# ------------------------------------------------------------

def subreddit_search(
    subreddit: str,
    date: str = "yesterday",
    max_results: int = 10
) -> str:
    """
    Searches Reddit posts from a specific subreddit using Tavily.

    Args:
        subreddit (str): Subreddit name without r/, for example "MicrosoftFabric".
        date (str): Date to search for. Supports "yesterday" or YYYY-MM-DD.
        max_results (int): Number of Tavily search results to retrieve.

    Returns:
        str: A formatted string containing Reddit search results.
    """

    subreddit = subreddit.replace("r/", "").strip()

    if not subreddit:
        return "No subreddit was provided."

    search_date = normalize_date_for_search(date)

    # Tavily searches the web, so the date is included in the query.
    # This is safer for deployment than Reddit's unauthenticated JSON endpoint,
    # but it may not return every post from the selected date.
    query = (
        f"site:reddit.com/r/{subreddit} "
        f"r/{subreddit} Reddit posts {search_date}"
    )

    response = tavily_client.search(
        query=query,
        include_domains=["reddit.com"],
        max_results=int(max_results),
        search_depth="advanced"
    )

    results = response.get("results", [])

    if not results:
        return f"No Reddit results found for r/{subreddit} around {search_date}."

    formatted_results = []

    for i, item in enumerate(results, start=1):
        title = item.get("title", "")
        content = item.get("content", "")
        url = item.get("url", "")

        formatted_results.append(
            f"""
Result {i}
Title: {title}
Content: {content}
URL: {url}
"""
        )

    return "\n\n".join(formatted_results)


# ------------------------------------------------------------
# OpenAI summarization
# ------------------------------------------------------------

def summarize_subreddit_posts(
    subreddit: str,
    date: str = "yesterday",
    preferences: str = "",
    max_results: int = 10
) -> str:
    """
    Searches Reddit using Tavily and summarizes the results using OpenAI.
    """

    posts = subreddit_search(
        subreddit=subreddit,
        date=date,
        max_results=max_results
    )

    if (
        posts.startswith("No Reddit results found")
        or posts.startswith("No subreddit")
    ):
        return posts

    # Avoid sending too much text to the model.
    max_chars = 30000

    if len(posts) > max_chars:
        posts = posts[:max_chars] + "\n\n[Additional results were truncated to stay within token limits.]"

    system_message = """
You are a Reddit subreddit summarization assistant.

Your job is to summarize Reddit search results retrieved from Tavily.

Important limitations:
- Tavily is a web search tool, not the official Reddit API.
- The results may not include every post from the selected subreddit or date.
- Reddit score and comment count may not be available.
- Be transparent if the available results are limited.

Rules:
- Summarize only the information provided in the retrieved results.
- Do not invent posts, comments, links, scores, trends, or opinions.
- Keep the summary concise, structured, and easy to read.
- Use the user's preferences only to decide what to emphasize in the summary.
"""

    user_message = f"""
Subreddit: r/{subreddit}
Date requested: {date}
User preferences for the summary: {preferences}

Retrieved Reddit search results:
{posts}

Please provide:
1. A short overall summary
2. Main topics discussed
3. Notable posts or links
4. Repeated issues, questions, or themes
5. A brief takeaway

When relevant, emphasize the user's preferences:
{preferences}

Be concise and clear.
Only return the final subreddit summary.
Do not explain the tool call process.
"""

    openai_response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
    )

    return openai_response.choices[0].message.content


# ------------------------------------------------------------
# Gradio styling
# ------------------------------------------------------------

custom_css = """
:root {
    --reddit-orange: #ff4500;
    --reddit-orange-dark: #d93a00;
    --reddit-bg: #fff7f3;
    --reddit-card: #ffffff;
    --reddit-text: #1c1c1c;
    --reddit-muted: #6b7280;
}

.gradio-container {
    background: linear-gradient(135deg, #fff7f3 0%, #fff1eb 45%, #ffffff 100%) !important;
    font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

#app-container {
    max-width: 980px;
    margin: 0 auto;
}

#hero {
    background: linear-gradient(135deg, #ff4500 0%, #ff7a1a 100%);
    color: white;
    padding: 32px;
    border-radius: 24px;
    box-shadow: 0 18px 40px rgba(255, 69, 0, 0.22);
    margin-bottom: 24px;
}

#hero h1 {
    font-size: 38px;
    margin-bottom: 8px;
}

#hero p {
    font-size: 17px;
    opacity: 0.95;
    margin-bottom: 0;
}

.input-card, .output-card {
    background: white;
    border-radius: 22px;
    padding: 22px;
    box-shadow: 0 12px 30px rgba(17, 24, 39, 0.08);
    border: 1px solid rgba(255, 69, 0, 0.12);
}

#submit-btn {
    background: linear-gradient(135deg, #ff4500 0%, #ff7a1a 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 16px !important;
    font-weight: 700 !important;
    font-size: 16px !important;
    padding: 12px 18px !important;
    box-shadow: 0 10px 22px rgba(255, 69, 0, 0.28) !important;
}

#submit-btn:hover {
    background: linear-gradient(135deg, #d93a00 0%, #ff6500 100%) !important;
    transform: translateY(-1px);
}

#clear-btn {
    border-radius: 16px !important;
}

.gr-textbox textarea,
.gr-textbox input {
    border-radius: 14px !important;
}

.gr-slider {
    border-radius: 14px !important;
}

#tips {
    background: #fff1eb;
    border-left: 5px solid #ff4500;
    padding: 14px 18px;
    border-radius: 16px;
    color: #3a1d12;
    margin-top: 12px;
}

footer {
    visibility: hidden;
}
"""


# ------------------------------------------------------------
# Gradio wrapper
# ------------------------------------------------------------

def gradio_summarize_subreddit(subreddit, date, preferences, max_results):
    """
    Wrapper function used by the Gradio interface.
    Handles user input validation and returns the final summary.
    """

    try:
        if not subreddit or not subreddit.strip():
            return "Please enter a subreddit name, for example `MicrosoftFabric`."

        if not date or not date.strip():
            date = "yesterday"

        return summarize_subreddit_posts(
            subreddit=subreddit,
            date=date,
            preferences=preferences,
            max_results=int(max_results)
        )

    except ValueError as e:
        return f"""
### Date format issue

Please use either:

- `yesterday`
- `YYYY-MM-DD`, for example `2026-05-18`

Error details: `{str(e)}`
"""

    except Exception as e:
        return f"""
### Something went wrong

Error details:

`{str(e)}`
"""


# ------------------------------------------------------------
# Gradio app
# ------------------------------------------------------------

with gr.Blocks(
    css=custom_css,
    title="Subreddit Pulse",
    theme=gr.themes.Soft(
        primary_hue="orange",
        secondary_hue="red",
        neutral_hue="slate"
    )
) as demo:

    with gr.Column(elem_id="app-container"):

        gr.HTML(
            """
            <div id="hero">
                <h1>🔥 Subreddit Pulse</h1>
                <p>
                    Search Reddit with Tavily and get a concise AI-powered summary of recent subreddit discussions.
                </p>
            </div>
            """
        )

        with gr.Row():

            with gr.Column(scale=1, elem_classes="input-card"):
                gr.Markdown("## Search settings")

                subreddit_input = gr.Textbox(
                    label="Subreddit",
                    value="MicrosoftFabric",
                    placeholder="Example: MicrosoftFabric, PowerBI, datascience",
                    info="Enter the subreddit name without r/"
                )

                date_input = gr.Textbox(
                    label="Date",
                    value="yesterday",
                    placeholder="yesterday or 2026-05-18",
                    info="Used as part of the Tavily search query"
                )

                preferences_input = gr.Textbox(
                    label="Summary preferences",
                    value="Focus on Power BI, Fabric, semantic models, data engineering, common issues, and highly discussed posts.",
                    placeholder="Example: focus on technical issues, questions, complaints, tutorials, or highly discussed posts",
                    lines=5,
                    info="Used only by OpenAI when creating the summary"
                )

                max_results_input = gr.Slider(
                    label="Number of Tavily results to retrieve",
                    minimum=5,
                    maximum=20,
                    value=10,
                    step=1,
                    info="More results may improve coverage but can take longer"
                )

                with gr.Row():
                    submit_btn = gr.Button(
                        "Summarize Subreddit 🚀",
                        elem_id="submit-btn",
                        scale=2
                    )

                    clear_btn = gr.ClearButton(
                        components=[
                            subreddit_input,
                            date_input,
                            preferences_input
                        ],
                        value="Clear",
                        elem_id="clear-btn",
                        scale=1
                    )

                gr.HTML(
                    """
                    <div id="tips">
                        <strong>Note:</strong> This version uses Tavily instead of direct Reddit access.
                        It is safer for deployment, but it may not retrieve every post or include Reddit score/comment counts.
                    </div>
                    """
                )

            with gr.Column(scale=2, elem_classes="output-card"):
                gr.Markdown("## Summary")

                output = gr.Markdown(
                    value="Your subreddit summary will appear here.",
                    label="Subreddit Summary"
                )

        gr.Examples(
            examples=[
                [
                    "MicrosoftFabric",
                    "yesterday",
                    "Focus on Power BI, Fabric, semantic models, data engineering, common issues, and highly discussed posts.",
                    10
                ],
                [
                    "PowerBI",
                    "yesterday",
                    "Focus on user problems, dashboard performance, DAX, semantic models, and practical tips.",
                    10
                ],
                [
                    "datascience",
                    "yesterday",
                    "Focus on career advice, project ideas, machine learning, and beginner questions.",
                    10
                ]
            ],
            inputs=[
                subreddit_input,
                date_input,
                preferences_input,
                max_results_input
            ],
            label="Try an example"
        )

        submit_btn.click(
            fn=gradio_summarize_subreddit,
            inputs=[
                subreddit_input,
                date_input,
                preferences_input,
                max_results_input
            ],
            outputs=output,
            show_progress="full"
        )


# Queue enables better request handling and visible loading/progress behaviour.
demo.queue()

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", 7860))
    )