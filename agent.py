"""
Autonomous Research Agent using LangChain
Assignment 2 - AI Research Agent
"""

import os
from datetime import datetime
from langchain_anthropic import ChatAnthropic
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import Tool
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ─────────────────────────────────────────────
# LLM Setup
# ─────────────────────────────────────────────
llm = ChatAnthropic(
    model="claude-opus-4-5",
    api_key=os.environ.get("ANTHROPIC_API_KEY"),
    max_tokens=4096,
)

# ─────────────────────────────────────────────
# Tool 1: Web Search (DuckDuckGo)
# ─────────────────────────────────────────────
search = DuckDuckGoSearchRun()
web_search_tool = Tool(
    name="web_search",
    func=search.run,
    description=(
        "Searches the web for current information on a topic. "
        "Use this to find recent news, statistics, and developments. "
        "Input should be a concise search query."
    ),
)

# ─────────────────────────────────────────────
# Tool 2: Wikipedia Knowledge Tool
# ─────────────────────────────────────────────
wiki_wrapper = WikipediaAPIWrapper(top_k_results=3, doc_content_chars_max=3000)
wiki_tool = WikipediaQueryRun(api_wrapper=wiki_wrapper)
wikipedia_tool = Tool(
    name="wikipedia_search",
    func=wiki_tool.run,
    description=(
        "Fetches detailed encyclopedic knowledge from Wikipedia. "
        "Use this for background information, definitions, history, and established facts. "
        "Input should be the topic or concept to look up."
    ),
)

tools = [web_search_tool, wikipedia_tool]

# ─────────────────────────────────────────────
# ReAct Agent Prompt
# ─────────────────────────────────────────────
react_prompt = PromptTemplate.from_template("""
You are an expert research agent. Your task is to conduct thorough research on a given topic
and gather comprehensive information to write a detailed report.

You have access to the following tools:
{tools}

Use the following format STRICTLY:

Question: the input topic you must research
Thought: think about what information you need to gather
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now have enough information to compile a comprehensive report
Final Answer: [COMPILED RESEARCH DATA]

Gather information covering:
1. Introduction and background of the topic
2. Current state and key developments
3. Key findings and insights (at least 5)
4. Major challenges and limitations
5. Future scope and predictions
6. Notable statistics and data points

Begin!

Question: Research the topic: {input}
Thought: {agent_scratchpad}
""")

# ─────────────────────────────────────────────
# Create ReAct Agent
# ─────────────────────────────────────────────
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=react_prompt,
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=8,
    handle_parsing_errors=True,
    return_intermediate_steps=True,
)

# ─────────────────────────────────────────────
# Report Generator
# ─────────────────────────────────────────────
report_prompt = PromptTemplate.from_template("""
You are a professional technical report writer. Using the research data provided below,
generate a comprehensive, well-structured report.

Topic: {topic}
Research Data: {research_data}
Date: {date}

Generate a detailed report with exactly this structure (use markdown headers):

# {topic}

## Cover Page
**Report Title:** {topic} - A Comprehensive Research Report  
**Prepared By:** AI Research Agent (LangChain + Claude)  
**Date:** {date}  
**Classification:** Academic Research

---

## 1. Introduction
[Write 3-4 paragraphs covering background, significance, and scope of the topic]

## 2. Key Findings
[List and explain at least 6 key findings with details and supporting evidence]

### Finding 1: [Title]
[Explanation]

### Finding 2: [Title]
[Explanation]

[Continue for all findings...]

## 3. Challenges
[Discuss 4-5 major challenges, obstacles, or limitations related to the topic]

## 4. Future Scope
[Discuss future trends, opportunities, and predictions for the next 5-10 years]

## 5. Conclusion
[Write a comprehensive 2-3 paragraph conclusion summarizing key points and significance]

## 6. References & Sources
[List the sources used during research]
""")

report_chain = report_prompt | llm | StrOutputParser()


# ─────────────────────────────────────────────
# Main Research Function
# ─────────────────────────────────────────────
def run_research_agent(topic: str) -> dict:
    """
    Run the full research agent pipeline for a given topic.
    Returns dict with raw research data and final report.
    """
    print(f"\n{'='*60}")
    print(f"  AUTONOMOUS RESEARCH AGENT")
    print(f"  Topic: {topic}")
    print(f"{'='*60}\n")

    # Step 1: Agent gathers research
    print("📡 Phase 1: Agent searching and gathering information...\n")
    result = agent_executor.invoke({"input": topic})
    research_data = result.get("output", "")

    # Step 2: Generate structured report
    print("\n📝 Phase 2: Generating structured report...\n")
    report = report_chain.invoke({
        "topic": topic,
        "research_data": research_data,
        "date": datetime.now().strftime("%B %d, %Y"),
    })

    return {
        "topic": topic,
        "research_data": research_data,
        "report": report,
        "timestamp": datetime.now().isoformat(),
    }


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # Sample topics
    topics = [
        "Impact of AI in Healthcare",
        "Quantum Computing and its Future Applications",
    ]

    for topic in topics:
        output = run_research_agent(topic)
        
        # Save report to file
        filename = topic.lower().replace(" ", "_") + "_report.md"
        with open(f"reports/{filename}", "w") as f:
            f.write(output["report"])
        
        print(f"\n✅ Report saved: reports/{filename}")
        print("\n" + "="*60 + "\n")
