"""
Multi-Agent Debate (by T.-W. Yoon, Mar. 2024)
"""

import streamlit as st
import os


def multi_agent_debate():
    """
    Let the two agents debate on a given subject.
    """

    st.write("## 📚 Multi-Agent Debate")
    st.write("#### Coming Soon...")

    with st.sidebar:
        st.write("---")
        st.write(
            "<small>**T.-W. Yoon**, Mar. 2024  \n</small>",
            "<small>[LangChain OpenAI Agent](https://langchain-openai-agent.streamlit.app/)  \n</small>",
            "<small>[OpenAI Assistants](https://assistants.streamlit.app/)  \n</small>",
            "<small>[TWY's Playground](https://twy-playground.streamlit.app/)  \n</small>",
            "<small>[Differential equations](https://diff-eqn.streamlit.app/)</small>",
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    multi_agent_debate()
