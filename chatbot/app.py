"""
app.py
------
A chatbot that lets you ask plain-English questions about the flagged
transaction data. Built with Claude's tool-calling: the model decides
which data-query function to call based on your question, the function
runs against the real dataset, and Claude turns the result into a
natural-language answer.

Run:
    export ANTHROPIC_API_KEY="your-key-here"
    streamlit run chatbot/app.py
"""

import os
import json
import streamlit as st
import anthropic

from query_tools import TOOL_DEFINITIONS, TOOL_FUNCTIONS

MODEL = "claude-sonnet-5"

SYSTEM_PROMPT = (
    "You are a data analyst assistant for a fintech transaction monitoring "
    "dashboard. You answer questions about flagged/anomalous transactions "
    "using the tools available to you. Always call a tool to get real data "
    "before answering — never guess or make up numbers. Keep answers "
    "concise and business-friendly, as if reporting to a stakeholder."
)


def get_client():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        st.error(
            "No ANTHROPIC_API_KEY found. Set it with:\n\n"
            "export ANTHROPIC_API_KEY='your-key-here'\n\n"
            "then restart the app."
        )
        st.stop()
    return anthropic.Anthropic(api_key=api_key)


def run_tool(name, tool_input):
    func = TOOL_FUNCTIONS.get(name)
    if not func:
        return {"error": f"Unknown tool: {name}"}
    return func(**tool_input)


def get_response(client, messages):
    """
    Sends the conversation to Claude, handles any tool calls it makes by
    running the corresponding local function, feeds the results back, and
    repeats until Claude produces a final text answer.
    """
    while True:
        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            tools=TOOL_DEFINITIONS,
            messages=messages,
        )

        if response.stop_reason != "tool_use":
            final_text = "".join(
                block.text for block in response.content if block.type == "text"
            )
            return final_text

        # Claude wants to call one or more tools
        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                result = run_tool(block.name, block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result),
                })

        messages.append({"role": "user", "content": tool_results})


def main():
    st.set_page_config(page_title="Transaction Insights Chatbot", page_icon="💳")
    st.title("💳 Transaction Anomaly Chatbot")
    st.caption(
        "Ask about flagged transactions in plain English — "
        "e.g. \"Which location has the most anomalies?\""
    )

    client = get_client()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        if isinstance(msg["content"], str):
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    user_input = st.chat_input("Ask a question about the transaction data...")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = get_response(client, st.session_state.messages)
                st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
