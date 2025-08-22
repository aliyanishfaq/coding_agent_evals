# LangGraph Agent Development Guide

## Core Requirements

**Always use LangGraph** for building agents. All agent implementations must:

- Export the compiled graph as `app` from `./agent.py`
- Include a `langgraph.json` configuration file
- Be deployment-ready

## File Structure
```
./agent.py          # Main agent file, exports: app
./langgraph.json    # LangGraph configuration
```

## Export Pattern
```python
# ... your agent implementation ...

app = graph.compile()  # Export compiled graph as 'app'
```

## Agent Instructions

The agent must accept a minimal state containing only a `messages` list with a `HumanMessage`:

```python
from agent import app

initial_state = {
    "messages": [HumanMessage("Your prompt here")]
}
result = app.invoke(initial_state)
```