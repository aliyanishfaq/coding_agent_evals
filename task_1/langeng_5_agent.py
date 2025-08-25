"""
LangGraph Deep Research Agent with Interactive Scoping and ReAct Search

This agent operates in two stages:
1. Interactive Scoping: Back-and-forth with user to clarify research scope
2. ReAct Research: Uses Tavily search to conduct research and generate report
"""

from typing import Annotated, Literal, TypedDict, Optional
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_anthropic import ChatAnthropic
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt
import os


# State schema for the agent
class ResearchState(TypedDict):
    """State schema for the research agent"""
    messages: Annotated[list[BaseMessage], add_messages]
    research_scope: str
    phase: Literal["scoping", "researching", "complete"]
    scope_confirmed: bool
    scoping_iterations: int  # Track number of scoping interactions


def scoping_agent(state: ResearchState) -> dict:
    """
    Interactive scoping agent that clarifies research requirements with the user.
    Uses interrupt() to enable back-and-forth conversation.
    """
    messages = state.get("messages", [])
    scope_confirmed = state.get("scope_confirmed", False)
    research_scope = state.get("research_scope", "")
    scoping_iterations = state.get("scoping_iterations", 0)
    
    # If scope is already confirmed, move to research phase
    if scope_confirmed:
        return {"phase": "researching"}
    
    # Initialize the LLM for scoping
    llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0.3)
    
    # System prompt for the scoping agent
    system_prompt = SystemMessage(content="""You are a research scoping assistant. Your role is to:
1. Understand what the user wants to research
2. Ask clarifying questions to define the scope clearly
3. Identify key aspects, boundaries, and specific areas of focus
4. Summarize the research scope when ready

Be conversational but focused. Ask one or two clarifying questions at a time.
After gathering enough information (usually 2-4 exchanges), provide a clear research scope summary and ask for confirmation.
Format your final scope summary clearly with bullet points.""")
    
    # Get the conversation history for context
    conversation = [system_prompt] + messages
    
    # Generate response
    response = llm.invoke(conversation)
    
    # Check if we should ask for confirmation (after a few iterations or if scope seems clear)
    response_text = response.content.lower()
    is_confirmation_request = any(phrase in response_text for phrase in [
        "confirm", "shall i proceed", "is this correct", "ready to begin", 
        "does this capture", "shall we proceed"
    ])
    
    # After 3+ iterations, encourage confirmation
    if scoping_iterations >= 3 and not is_confirmation_request:
        # Add a confirmation prompt
        response.content += "\n\nBased on our discussion, I believe I have a clear understanding of your research needs. Shall we proceed with this scope?"
        is_confirmation_request = True
    
    if is_confirmation_request:
        # Interrupt to get user confirmation
        user_input = interrupt({
            "message": response.content,
            "awaiting": "confirmation",
            "phase": "scoping_confirmation"
        })
        
        # Process user response
        user_text = user_input if isinstance(user_input, str) else ""
        user_text_lower = user_text.lower()
        
        if any(word in user_text_lower for word in ["yes", "confirm", "proceed", "correct", "looks good", "perfect"]):
            # Extract research scope from the conversation
            # Look for the most recent scope summary in the response
            scope_summary = response.content
            
            # Find the scope section if it exists
            if "research scope" in response_text or "scope:" in response_text:
                # Extract the scope section
                lines = response.content.split('\n')
                scope_lines = []
                in_scope = False
                for line in lines:
                    if any(marker in line.lower() for marker in ["research scope", "scope:", "will research", "will investigate"]):
                        in_scope = True
                    if in_scope:
                        scope_lines.append(line)
                if scope_lines:
                    scope_summary = '\n'.join(scope_lines)
            
            return {
                "messages": [response, HumanMessage(content=user_text)],
                "research_scope": scope_summary,
                "scope_confirmed": True,
                "phase": "researching",
                "scoping_iterations": scoping_iterations + 1
            }
        else:
            # User wants to clarify more
            return {
                "messages": [response, HumanMessage(content=user_text)],
                "scope_confirmed": False,
                "phase": "scoping",
                "scoping_iterations": scoping_iterations + 1
            }
    else:
        # Continue the scoping conversation
        user_input = interrupt({
            "message": response.content,
            "awaiting": "clarification",
            "phase": "scoping_discussion"
        })
        
        user_text = user_input if isinstance(user_input, str) else ""
        
        return {
            "messages": [response, HumanMessage(content=user_text)],
            "scope_confirmed": False,
            "phase": "scoping",
            "scoping_iterations": scoping_iterations + 1
        }


def research_agent_node(state: ResearchState) -> dict:
    """
    Research agent that uses ReAct pattern with Tavily search to conduct research.
    """
    research_scope = state.get("research_scope", "")
    messages = state.get("messages", [])
    
    if not research_scope:
        return {
            "messages": [AIMessage(content="No research scope defined. Please start with the scoping phase.")],
            "phase": "complete"
        }
    
    # Initialize Tavily search tool
    tavily_tool = TavilySearch(
        max_results=5,
        search_depth="advanced",
        include_answer=True,
        include_raw_content=False,
        include_images=False,
        name="tavily_search"
    )
    
    # Create the ReAct agent with Tavily search
    llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0.3)
    
    # Create a comprehensive research prompt
    research_prompt = f"""You are a deep research agent conducting comprehensive research based on the following scope:

{research_scope}

Instructions:
1. Use the tavily_search tool multiple times to gather comprehensive information
2. Search for different aspects and angles of the topic
3. Look for recent data, statistics, expert opinions, and multiple perspectives
4. Verify important information from multiple sources when possible
5. After gathering sufficient information, synthesize your findings into a detailed report

Your final report should include:
- Executive Summary
- Key Findings (with bullet points)
- Detailed Analysis
- Data and Statistics (if relevant)
- Different Perspectives or Viewpoints
- Conclusions and Insights
- Sources and References

Be thorough but concise. Focus on providing valuable, actionable insights."""
    
    # Create the ReAct agent with proper configuration
    react_agent = create_react_agent(
        model=llm,
        tools=[tavily_tool],
        prompt=research_prompt,
        name="research_agent"
    )
    
    # Prepare input for the ReAct agent
    research_input = {
        "messages": [HumanMessage(content=f"Please conduct comprehensive research based on this scope: {research_scope}")]
    }
    
    try:
        # Run the ReAct agent
        result = react_agent.invoke(research_input)
        
        # Extract the final report from the agent's messages
        if result.get("messages"):
            # Get the last AI message which should contain the final report
            final_message = None
            for msg in reversed(result["messages"]):
                if isinstance(msg, AIMessage) and not hasattr(msg, 'tool_calls'):
                    final_message = msg
                    break
            
            if not final_message:
                final_message = result["messages"][-1]
            
            # Add a header to the report
            report_content = f"""
# Research Report

## Research Scope
{research_scope}

---

{final_message.content if hasattr(final_message, 'content') else str(final_message)}
"""
            
            return {
                "messages": [AIMessage(content=report_content)],
                "phase": "complete"
            }
        else:
            return {
                "messages": [AIMessage(content="Research completed but no results were generated.")],
                "phase": "complete"
            }
    except Exception as e:
        error_message = f"An error occurred during research: {str(e)}"
        return {
            "messages": [AIMessage(content=error_message)],
            "phase": "complete"
        }


def route_phase(state: ResearchState) -> Literal["scoping", "research", "end"]:
    """
    Route to the appropriate phase based on the current state.
    """
    phase = state.get("phase", "scoping")
    scope_confirmed = state.get("scope_confirmed", False)
    
    if phase == "complete":
        return "end"
    elif scope_confirmed and phase == "researching":
        return "research"
    else:
        return "scoping"


# Build the graph
def create_research_graph():
    """
    Create the two-stage research agent graph with interactive scoping and ReAct research.
    
    This graph implements a two-phase research workflow:
    1. Scoping Phase: Interactive back-and-forth with user to clarify research requirements
    2. Research Phase: ReAct agent with Tavily search to conduct comprehensive research
    
    Returns:
        CompiledStateGraph: The compiled LangGraph agent ready for deployment
    """
    # Initialize the graph with our state schema
    graph = StateGraph(ResearchState)
    
    # Add nodes
    graph.add_node("scoping", scoping_agent)
    graph.add_node("research", research_agent_node)
    
    # Add edges
    graph.add_edge(START, "scoping")
    
    # Add conditional routing from scoping node
    graph.add_conditional_edges(
        "scoping",
        route_phase,
        {
            "scoping": "scoping",  # Continue scoping conversation
            "research": "research",  # Move to research phase
            "end": END  # End if needed
        }
    )
    
    # Research always leads to END
    graph.add_edge("research", END)
    
    # Add memory for persistence (required for interrupt support)
    memory = MemorySaver()
    
    # Compile the graph with checkpointer for interrupt support
    compiled_graph = graph.compile(checkpointer=memory)
    
    return compiled_graph


# Create and export the app (required by AGENTS.md)
app = create_research_graph()


# Helper function for testing the agent locally
if __name__ == "__main__":
    """
    Example usage of the Deep Research Agent.
    This demonstrates how to run the agent with interactive scoping and research phases.
    """
    import sys
    from langgraph.types import Command
    
    print("=" * 70)
    print("DEEP RESEARCH AGENT WITH INTERACTIVE SCOPING")
    print("=" * 70)
    print("\nThis agent will:")
    print("1. Work with you interactively to clarify your research scope")
    print("2. Conduct comprehensive web research using Tavily search")
    print("3. Generate a detailed research report\n")
    print("=" * 70)
    
    # Get initial query
    print("\nWhat would you like to research?")
    initial_query = input("> ")
    
    if not initial_query:
        print("No query provided. Exiting.")
        sys.exit(0)
    
    # Configure the session
    config = {"configurable": {"thread_id": "research-session-001"}}
    
    # Initial state
    initial_state = {
        "messages": [HumanMessage(content=initial_query)],
        "phase": "scoping",
        "scope_confirmed": False,
        "research_scope": "",
        "scoping_iterations": 0
    }
    
    print("\n" + "=" * 70)
    print("STARTING INTERACTIVE SCOPING PHASE")
    print("=" * 70)
    
    try:
        # Start the agent
        for event in app.stream(initial_state, config):
            # Check if we're at an interrupt point
            if "__interrupt__" in event:
                interrupts = event["__interrupt__"]
                for interrupt_data in interrupts:
                    if hasattr(interrupt_data, 'value'):
                        interrupt_info = interrupt_data.value
                        
                        # Display the agent's message
                        print("\n" + "-" * 50)
                        print("AGENT:")
                        print("-" * 50)
                        print(interrupt_info.get("message", ""))
                        
                        # Get user input
                        print("\n" + "-" * 50)
                        print("YOUR RESPONSE:")
                        user_input = input("> ")
                        
                        # Resume with user input
                        app.update_state(
                            config,
                            {"messages": [HumanMessage(content=user_input)]},
                            as_node="scoping"
                        )
            
            # Process node outputs
            for node_name, node_data in event.items():
                if node_name == "scoping" and node_data.get("phase") == "researching":
                    print("\n" + "=" * 70)
                    print("SCOPE CONFIRMED - STARTING RESEARCH PHASE")
                    print("=" * 70)
                    print("\nResearch Scope:")
                    print(node_data.get("research_scope", ""))
                    print("\nConducting research... This may take a moment...")
                
                elif node_name == "research" and node_data.get("phase") == "complete":
                    print("\n" + "=" * 70)
                    print("RESEARCH COMPLETE")
                    print("=" * 70)
                    
                    # Display the final report
                    messages = node_data.get("messages", [])
                    if messages:
                        last_message = messages[-1]
                        if hasattr(last_message, 'content'):
                            print(last_message.content)
                    
                    print("\n" + "=" * 70)
                    print("Thank you for using the Deep Research Agent!")
                    print("=" * 70)
    
    except KeyboardInterrupt:
        print("\n\nResearch interrupted by user. Exiting...")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback
        traceback.print_exc()






