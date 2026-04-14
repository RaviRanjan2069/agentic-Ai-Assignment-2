# 🤖 Autonomous Research Agent (LangChain)

> Assignment 2 — AI Agent that researches any topic and generates a structured report

## Architecture

```
User Input (Topic)
       ↓
  ReAct Agent (LangChain)
       ↓
  ┌──────────┬────────────────┐
  │Web Search│ Wikipedia Tool │
  │(DuckDuck)│   (Knowledge)  │
  └──────────┴────────────────┘
       ↓
  Research Data Compiled
       ↓
  Report Generator Chain
       ↓
  Structured Final Report (Markdown)
```

## Setup

```bash
# Clone and install
pip install -r requirements.txt

# Set your API key
export ANTHROPIC_API_KEY="your-key-here"

# Run the agent
python agent.py
```

## Usage

```python
from agent import run_research_agent

result = run_research_agent("Impact of AI in Healthcare")
print(result["report"])
```

## Tools Used

| Tool | Purpose |
|------|---------|
| `DuckDuckGoSearchRun` | Real-time web search for current news & stats |
| `WikipediaQueryRun` | Encyclopedic background knowledge |

## Output Structure

Every generated report includes:
- 📋 Cover Page
- 📌 Introduction
- 🔍 Key Findings (6+)
- ⚠️ Challenges
- 🚀 Future Scope
- ✅ Conclusion
- 📚 References

## Sample Topics
- "Impact of AI in Healthcare"
- "Quantum Computing and its Future Applications"
- "Climate Change and Renewable Energy"
- "Blockchain in Supply Chain Management"
