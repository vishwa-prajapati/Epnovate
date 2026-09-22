import os
import json
from dotenv import load_dotenv
from fastmcp import FastMCP
from newsapi import NewsApiClient
from groq import Groq
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

load_dotenv()

mcp = FastMCP("Epnovate MCP Server")

newsapi = NewsApiClient(api_key=os.getenv("NEWSAPI_KEY"))
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")


@mcp.tool
def scrape_news_articles(query: str) -> str:
    all_articles = newsapi.get_everything(q=query, language="en", page_size=5)
    articles = all_articles.get("articles", [])
    headlines = [a["title"] for a in articles]
    return f"Found {len(articles)} articles for '{query}': {headlines}"


@mcp.tool
def get_market_benchmarks(query: str) -> str:
    benchmarks = {
        "adoption": 0.75,
        "sentiment": 0.85,
        "cost_index": 1.2
    }
    return f"Market benchmarks for '{query}': {benchmarks}"


@mcp.tool
def check_compliance_risks(query: str) -> str:
    risks = {
        "data_privacy": "low",
        "regulatory": "medium",
        "financial": "high"
    }
    return f"Compliance risks for '{query}': {risks}"


@mcp.tool
def synthesize_insights(summary_text: str) -> str:
    chat_completion = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": f"Synthesize this into a clear, well structured report summary:\n\n{summary_text}"
            }
        ],
    )
    return chat_completion.choices[0].message.content


@mcp.tool
def render_pdf_report(content: str, filename: str = "report.pdf") -> str:
    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter
    text_obj = c.beginText(50, height - 50)
    text_obj.setFont("Helvetica", 11)

    for line in content.split("\n"):
        chunks = [line[i:i + 95] for i in range(0, len(line), 95)] or [""]
        for chunk in chunks:
            text_obj.textLine(chunk)

    c.drawText(text_obj)
    c.save()
    return f"PDF report generated: {filename}"


TOOL_FUNCTIONS = {
    "scrape_news_articles": scrape_news_articles,
    "get_market_benchmarks": get_market_benchmarks,
    "check_compliance_risks": check_compliance_risks,
    "synthesize_insights": synthesize_insights,
    "render_pdf_report": render_pdf_report,
}

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "scrape_news_articles",
            "description": "Fetch recent news articles for a given topic.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_market_benchmarks",
            "description": "Get internal enterprise adoption, sentiment and cost index metrics for a topic.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_compliance_risks",
            "description": "Check regulatory and compliance risks tied to a topic.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "synthesize_insights",
            "description": "Merge news data and internal metrics into one coherent written summary. Call this after news, market benchmarks and compliance data have all been gathered.",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary_text": {
                        "type": "string",
                        "description": "All raw data gathered so far, combined into one string",
                    }
                },
                "required": ["summary_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "render_pdf_report",
            "description": "Compile the final synthesized summary into a PDF file on disk. Call this last, after synthesis is complete.",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {"type": "string"},
                    "filename": {"type": "string"},
                },
                "required": ["content"],
            },
        },
    },
]

SYSTEM_PROMPT = (
    "You are an orchestrator agent for a research report pipeline. "
    "Given a user query, decide which tools to call and in what order. "
    "Usually that means gathering news, market benchmarks and compliance risks, "
    "synthesizing them into one summary, then rendering that summary as a PDF. "
    "Call one tool at a time and use the results so far to decide the next step. "
    "Once the PDF has been rendered, reply with a short confirmation and no more tool calls."
)


def run_agent(query: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]

    while True:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",
        )

        message = response.choices[0].message
        messages.append(message)

        if not message.tool_calls:
            return message.content

        for tool_call in message.tool_calls:
            name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)
            print(f"calling {name} with {args}")

            result = TOOL_FUNCTIONS[name](**args)

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str(result),
            })


if __name__ == "__main__":
    query = input("Enter your query: ")
    final_message = run_agent(query)
    print(final_message)