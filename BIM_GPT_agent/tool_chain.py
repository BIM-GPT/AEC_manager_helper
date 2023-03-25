"""Agent for working with OpenDataBIM csv output and having other tools too."""
from typing import Any, List, Optional

from langchain.agents.agent import AgentExecutor

# redefine your prompt here
from prompt import SUFFIX, PREFIX
from langchain.agents.mrkl.base import ZeroShotAgent
from langchain.callbacks.base import BaseCallbackManager
from langchain.chains.llm import LLMChain
from langchain.llms.base import BaseLLM

# tools that we need
from langchain.tools.python.tool import PythonAstREPLTool
from langchain.agents import load_tools
from langchain.agents import load_tools

def create_OpenDataBIM_helper_agent(
    llm: BaseLLM,
    df: Any,
    callback_manager: Optional[BaseCallbackManager] = None,
    prefix: str = PREFIX,
    suffix: str = SUFFIX,
    input_variables: Optional[List[str]] = None,
    verbose: bool = False,
    **kwargs: Any,
) -> AgentExecutor:
    """Construct a pandas agent from an LLM and dataframe."""
    import pandas as pd

    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"Expected pandas object, got {type(df)}")
    if input_variables is None:
        input_variables = ["df", "input", "agent_scratchpad", "chat_history"]
    
    
    
    template = """This is a conversation between a human and a bot:

{chat_history}

Write a summary of the conversation for {input}:
"""
    prompt = PromptTemplate(
        input_variables=['chat_history', 'input'], 
        template=template
    )
    memory = ConversationBufferMemory(memory_key="chat_history")
    readonlymemory = ReadOnlySharedMemory(memory=memory)
    summry_chain = LLMChain(
        llm=llm, 
        prompt=prompt, 
        verbose=True, 
        memory=readonlymemory, # use the read-only memory to prevent the tool from modifying the memory
    )
    
    
    tools = [
             PythonAstREPLTool(locals={"df": df}), 
             # DON'T FORGET TO REMOVE KEY!!!
             load_tools(["google-serper"], 
                        llm=llm)[0],
             Tool(
                name = "Summary",
                func=summry_chain.run,
                description="useful for when you summarize a conversation. The input to this tool should be a string, representing who will read this summary."
             )
            ]
    prompt = ZeroShotAgent.create_prompt(
        tools, prefix=prefix, suffix=suffix, input_variables=input_variables
    )
    partial_prompt = prompt.partial(df=str(df.head()))
    llm_chain = LLMChain(
        llm=llm,
        prompt=partial_prompt,
        callback_manager=callback_manager,
    )
    tool_names = [tool.name for tool in tools]
    agent = ZeroShotAgent(llm_chain=llm_chain, allowed_tools=tool_names, **kwargs)
    return AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=verbose, memory=memory)
