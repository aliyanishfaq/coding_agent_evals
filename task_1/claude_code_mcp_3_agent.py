import os
from typing import TypedDict, List, Annotated
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearchResults
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages


class ResearchState(TypedDict):
    messages: Annotated[List, add_messages]
    research_brief: str
    research_complete: bool
    final_report: str


class ResearchAgent:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.tavily_tool = TavilySearchResults(
            max_results=5,
            search_depth="advanced",
            include_answer=True,
            include_raw_content=True
        )
        self.tools = [self.tavily_tool]
        self.llm_with_tools = self.llm.bind_tools(self.tools)

    def clarification_node(self, state: ResearchState) -> ResearchState:
        messages = state["messages"]
        
        if not any("research_brief" in str(msg.content).lower() for msg in messages):
            clarification_prompt = """You are a research scoping assistant. Your job is to help clarify and scope research topics through interactive questions.

The user has requested research on a topic. Ask 2-3 focused questions to understand:
1. The specific scope and focus of the research
2. The intended audience or use case
3. Any particular aspects they want emphasized

Keep your questions concise and practical. Once you have enough information, create a clear research brief."""

            response = self.llm.invoke([
                SystemMessage(content=clarification_prompt),
                *messages
            ])
            
            return {
                "messages": [response],
                "research_brief": state.get("research_brief", ""),
                "research_complete": False,
                "final_report": ""
            }
        
        # Check if we have enough information to create a brief
        brief_prompt = """Based on the conversation, create a concise research brief (2-3 sentences) that captures:
1. The main research topic
2. The scope and focus
3. The intended outcome

If you don't have enough information yet, ask one more clarifying question.
If you have enough information, start your response with "RESEARCH_BRIEF:" followed by the brief."""

        response = self.llm.invoke([
            SystemMessage(content=brief_prompt),
            *messages
        ])
        
        if "RESEARCH_BRIEF:" in response.content:
            brief = response.content.split("RESEARCH_BRIEF:")[1].strip()
            return {
                "messages": state["messages"] + [response],
                "research_brief": brief,
                "research_complete": False,
                "final_report": ""
            }
        
        return {
            "messages": state["messages"] + [response],
            "research_brief": state.get("research_brief", ""),
            "research_complete": False,
            "final_report": ""
        }

    def react_agent_node(self, state: ResearchState) -> ResearchState:
        brief = state["research_brief"]
        
        react_prompt = f"""You are a research agent conducting deep research based on this brief: {brief}

Use the search tool to gather comprehensive information. Follow this process:
1. Start with broad searches on the main topic
2. Drill down into specific aspects based on initial findings
3. Look for recent developments, key players, statistics, and trends
4. Gather diverse perspectives and sources

Make multiple searches as needed. When you have sufficient information, compile it into a detailed research report."""

        messages = [SystemMessage(content=react_prompt)]
        
        # Continue the ReAct loop until we have a comprehensive report
        max_iterations = 10
        iteration = 0
        
        while iteration < max_iterations:
            response = self.llm_with_tools.invoke(messages)
            messages.append(response)
            
            if response.tool_calls:
                tool_node = ToolNode(self.tools)
                tool_result = tool_node.invoke({"messages": [response]})
                messages.extend(tool_result["messages"])
            else:
                # No more tool calls, check if we should continue or finish
                continue_prompt = """Based on your research so far, determine if you need more information or if you're ready to write the final report.

If you need more information, make another search.
If you're ready, respond with "FINAL_REPORT:" followed by a comprehensive research report that includes:
1. Executive summary
2. Key findings
3. Detailed analysis
4. Current trends and developments
5. Conclusions and implications"""

                continue_response = self.llm_with_tools.invoke(messages + [HumanMessage(content=continue_prompt)])
                
                if "FINAL_REPORT:" in continue_response.content:
                    report = continue_response.content.split("FINAL_REPORT:")[1].strip()
                    return {
                        "messages": state["messages"] + [AIMessage(content=f"Research completed. Final report:\n\n{report}")],
                        "research_brief": brief,
                        "research_complete": True,
                        "final_report": report
                    }
                else:
                    messages.append(continue_response)
            
            iteration += 1
        
        # Fallback if max iterations reached
        return {
            "messages": state["messages"] + [AIMessage(content="Research process completed with available information.")],
            "research_brief": brief,
            "research_complete": True,
            "final_report": "Research completed with available data."
        }

    def should_continue(self, state: ResearchState) -> str:
        if not state["research_brief"]:
            return "clarify"
        elif not state["research_complete"]:
            return "research"
        else:
            return END

    def create_graph(self):
        workflow = StateGraph(ResearchState)
        
        workflow.add_node("clarify", self.clarification_node)
        workflow.add_node("research", self.react_agent_node)
        
        workflow.set_entry_point("clarify")
        
        workflow.add_conditional_edges(
            "clarify",
            self.should_continue,
            {
                "clarify": "clarify",
                "research": "research",
                END: END
            }
        )
        
        workflow.add_edge("research", END)
        
        return workflow.compile()


# Create and export the app
research_agent = ResearchAgent()
app = research_agent.create_graph()