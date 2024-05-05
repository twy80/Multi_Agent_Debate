"""
Multi-Agent Debate (by T.-W. Yoon, Mar. 2024)
"""

import os, requests, datetime
import streamlit as st
from functools import partial
from tempfile import NamedTemporaryFile
from typing import List, Callable, Literal
from streamlit.runtime.uploaded_file_manager import UploadedFile
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.utilities import BingSearchAPIWrapper
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.tools import Tool
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import create_openai_tools_agent
# from langchain.agents import create_tool_calling_agent
from langchain.agents import AgentExecutor
from langchain.agents import load_tools
from langchain.pydantic_v1 import BaseModel, Field


def initialize_session_state_variables() -> None:
    """
    Initialize all the session state variables.
    """

    if "ready" not in st.session_state:
        st.session_state.ready = False

    if "bing_subscription_validity" not in st.session_state:
        st.session_state.bing_subscription_validity = False

    if "langchain_api_validity" not in st.session_state:
        st.session_state.langchain_api_validity = False

    if "language" not in st.session_state:
        st.session_state.language = "English"

    if "topic" not in st.session_state:
        st.session_state.topic = ""

    if "positive" not in st.session_state:
        st.session_state.positive = ""

    if "negative" not in st.session_state:
        st.session_state.negative = ""

    if "agent_descriptions" not in st.session_state:
        st.session_state.agent_descriptions = {}

    if "specified_topic" not in st.session_state:
        st.session_state.specified_topic = ""

    if "new_debate" not in st.session_state:
        st.session_state.new_debate = True

    if "conversations" not in st.session_state:
        st.session_state.conversations = []

    if "conversations4print" not in st.session_state:
        st.session_state.conversations4print = []

    if "simulator" not in st.session_state:
        st.session_state.simulator = None

    if "names" not in st.session_state:
        st.session_state.names = {}

    if "tools" not in st.session_state:
        st.session_state.tools = []

    if "retriever_tool" not in st.session_state:
        st.session_state.retriever_tool = None

    if "vector_store_message" not in st.session_state:
        st.session_state.vector_store_message = ""

    if "conclusions" not in st.session_state:
        st.session_state.conclusions = ""

    if "comments_key" not in st.session_state:
        st.session_state.comments_key = 0


class MySearchToolInput(BaseModel):
    query: str = Field(description="search query to look up")


def is_openai_api_key_valid(openai_api_key: str) -> bool:
    """
    Return True if the given OpenAI API key is valid.
    """

    headers = {
        "Authorization": f"Bearer {openai_api_key}",
    }
    response = requests.get(
        "https://api.openai.com/v1/models", headers=headers
    )

    return response.status_code == 200


def is_bing_subscription_key_valid(bing_subscription_key: str) -> bool:
    """
    Return True if the given Bing subscription key is valid.
    """

    try:
        bing_search = BingSearchAPIWrapper(
            bing_subscription_key=bing_subscription_key,
            bing_search_url="https://api.bing.microsoft.com/v7.0/search",
            k=1
        )
        bing_search.run("Where can I get a Bing subscription key?")
    except:
        return False
    else:
        return True


def check_api_keys() -> None:
    """
    Unset this flag to check the validity of the OpenAI API key.
    """

    st.session_state.ready = False


def append_period(text: str) -> str:
    """
    Append a '.' to the input text
    if it is nonempty and does not end with '.' or '?'.
    """

    if text and text[-1] not in (".", "?"):
        text += "."
    return text


def get_vector_store(uploaded_files: List[UploadedFile]) -> "FAISS":
    """
    Take a list of UploadedFile objects as input,
    and return a FAISS vector store.
    """

    if not uploaded_files:
        return None

    documents = []
    filepaths = []
    loader_map = {
        ".pdf": PyPDFLoader,
        ".txt": TextLoader,
        ".docx": Docx2txtLoader
    }
    try:
        for uploaded_file in uploaded_files:
            # Create a temporary file within the "files/" directory
            with NamedTemporaryFile(dir="files/", delete=False) as file:
                file.write(uploaded_file.getbuffer())
                filepath = file.name
            filepaths.append(filepath)

            file_ext = os.path.splitext(uploaded_file.name.lower())[1]
            loader_class = loader_map.get(file_ext)
            if not loader_class:
                st.error(f"Unsupported file type: {file_ext}", icon="🚨")
                for filepath in filepaths:
                    if os.path.exists(filepath):
                        os.remove(filepath)
                return None

            # Load the document using the selected loader.
            loader = loader_class(filepath)
            documents.extend(loader.load())

        with st.spinner("Vector DB in preparation..."):
            # Split the loaded text into smaller chunks for processing.
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                # separators=["\n", "\n\n", "(?<=\. )", "", " "],
            )
            doc = text_splitter.split_documents(documents)
            # Create a FAISS vector database.
            embeddings = OpenAIEmbeddings()
            vector_store = FAISS.from_documents(doc, embeddings)
    except Exception as e:
        vector_store = None
        st.error(f"An error occurred: {e}", icon="🚨")
    finally:
        # Ensure the temporary file is deleted after processing
        for filepath in filepaths:
            if os.path.exists(filepath):
                os.remove(filepath)

    return vector_store


def get_retriever() -> None:
    """
    Upload document(s), create a vector store, prepare a retriever tool,
    save the tool to the variable st.session_state.retriever_tool
    """

    st.write("")
    st.write("##### Document(s) to ask about")
    uploaded_files = st.file_uploader(
        label="Upload an article",
        type=["txt", "pdf", "docx"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    left, right = st.columns(2)
    if left.button(label="$\:\!$Create a vector DB$\,$"):
        # Create the vector store.
        vector_store = get_vector_store(uploaded_files)

        if vector_store is not None:
            retriever = vector_store.as_retriever()
            st.session_state.retriever_tool = create_retriever_tool(
                retriever,
                name="retriever",
                description=(
                    "Search for information about the uploaded documents. "
                    "For any questions about the documents, you must use "
                    "this tool!"
                ),
            )
            st.session_state.vector_store_message = "Vector DB is ready!"

    if st.session_state.vector_store_message:
        right.write(f":blue[{st.session_state.vector_store_message}]")


class DialogueAgent:
    """
    Class for an individual agent participating in the debate.
    """

    def __init__(
        self,
        name: str,
        system_message: SystemMessage,
        llm: ChatOpenAI,
        tools: List[str],
    ) -> None:
        self.name = name
        self.system_message = system_message
        self.llm = llm
        self.prefix = f"{self.name}: "
        self.tools = tools
        self.reset()

    def reset(self):
        self.message_history = ["\nHere is the conversation so far.\n"]

    def send(self) -> str:
        """
        Apply the llm to the message history and return the message string.
        """
        chat_prompt_list = [
            ("system", "You are a helpful assistant."),
            ("human", "{input}"),
        ]
        agent_prompt_list = chat_prompt_list + [
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ]
        chat_prompt = ChatPromptTemplate.from_messages(chat_prompt_list)
        agent_prompt = ChatPromptTemplate.from_messages(agent_prompt_list)

        if self.tools:
            agent = create_openai_tools_agent(
                self.llm, self.tools, agent_prompt
            )
            agent_executor = AgentExecutor(
                agent=agent, tools=self.tools, verbose=False
            )
        else:
            agent_executor = chat_prompt | self.llm

        output = agent_executor.invoke(
            {
                "input": "\n".join(
                    [self.system_message.content]
                    + self.message_history
                    + [self.prefix]
                )
            }
        )
        message = output["output"] if self.tools else output.content
        return message

    def receive(self, name: str, message: str) -> None:
        """
        Concatenate {message} spoken by {name} into message history
        """
        self.message_history.append(f"{name}: {message}\n")


class DialogueSimulator:
    """
    Class for simulating the debate.
    """

    def __init__(
        self,
        agents: List[DialogueAgent],
        selection_function: Callable[[int, List[DialogueAgent]], int],
    ) -> None:
        self.agents = agents
        self._step = 0
        self.select_next_speaker = selection_function

    def reset(self):
        for agent in self.agents:
            agent.reset()

    def inject(self, name: str, message: str):
        """
        Initiate the conversation with a {message} from {name}
        """
        for agent in self.agents:
            agent.receive(name, message)

        # increment time
        # self._step += 1

    def step(self) -> tuple[str, str]:
        # 1. choose the next speaker
        speaker_idx = self.select_next_speaker(self._step, self.agents)
        speaker = self.agents[speaker_idx]

        # 2. next speaker sends message
        try:
            with st.spinner(f"{speaker.name} is thinking..."):
                message = speaker.send()
        except Exception as e:
            st.error(f"An error occurred: {e}", icon="🚨")
            st.stop()

        # 3. everyone receives message
        for receiver in self.agents:
            receiver.receive(speaker.name, message)

        # 4. increment time
        self._step += 1

        return speaker.name, message


def select_next_speaker(step: int, agents: List[DialogueAgent]) -> int:
    """
    Return 0, 1, ..., or (the number of agents - 1) corresponding
    to the next speaker.
    """

    idx = (step) % len(agents)
    return idx


def run_simulator(no_of_rounds: int, simulator: DialogueSimulator) -> None:
    """
    Simulate a given number of rounds for the debate.
    What each participant speaks is saved to a session state list
    containing all the conversations so far.
    """

    max_iters = 2 * no_of_rounds
    iter = 0

    while iter < max_iters:
        name, message = simulator.step()
        color = "blue" if iter % 2 == 0 else "red"
        message4print = f"**:{color}[{name}]**: {message}"

        st.session_state.conversations.append(f"{name}: {message}")
        st.session_state.conversations4print.append(message4print)

        st.write(message4print)
        iter += 1


def generate_agent_description(
    name: str,
    conversation_description: str,
    language: Literal['English', 'Korean'],
    word_limit: int
) -> str:

    """
    Generate the description for a participant.
    """

    agent_specifier_prompt = [
        SystemMessage(
            content=(
                "You can add detail to the description of "
                "the conversation participant."
            )
        ),
        HumanMessage(
            content=(
                f"{conversation_description}\n"
                f"Please reply with a creative description of '{name}', "
                f"in {word_limit} words or less in {language}.\n"
                f"Speak directly to '{name}'.\n"
                "Give them a point of view.\n"
                "Do not add anything else."
            )
        ),
    ]
    agent_specifier_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=1.0)
    agent_description = agent_specifier_llm.invoke(agent_specifier_prompt)

    return agent_description.content


def generate_system_message(
    name: str,
    conversation_description: str,
    description: str,
    language: Literal['English', 'Korean'],
    word_limit: int
) -> str:

    """
    Generate the system message for a participant.
    """

    if description:
        description_statement = (
            f"Your description is as follows: {description}\n\n"
        )
    else:
        description_statement = ""

    generated_system_message = (
        f"{conversation_description}\n\n"
        f"Your name is '{name}'.\n\n"
        f"{description_statement}"
        "Your goal is to persuade your conversation partner "
        "of your point of view.\n\n"
        "DO look up information with your tool "
        "to refute your partner's claims.\n"
        "DO cite your sources.\n\n"
        "DO NOT fabricate fake citations.\n"
        "DO NOT cite any source that you did not look up.\n\n"
        "DO NOT restate something that has been said in the past.\n"
        "Do not add anything else.\n\nStop speaking the moment "
        "you finish speaking from your perspective.\n\n"
        f"Answer in {word_limit} words or less in {language}."
    )
    return generated_system_message


def get_participant_names(topic: str) -> List[str]:
    """
    Get the names of the positive and negative for the debate.
    """

    participants = ["positive", "negative"]
    participant_names = []

    for participant in participants:
        ex = "AI alarmist" if participant == "negative" else "AI accelerationist"
        name_specifier_prompt = [
            SystemMessage(content="You are a helpful moderator for a debate."),
            HumanMessage(
                content=(
                    "Here is the topic of conversation: "
                    f"{append_period(topic)}\n"
                    f"For the {participant} perspective on the topic, "
                    "write a name in three words or less. Start the name "
                    "with a capital letter and do not use ':' .\n"
                    "For example, for the topic 'The current impact of "
                    "automation and artificial intelligence on employment', "
                    f"'{ex}' could serve as an appropriate name for "
                    f"the {participant} side.\n"
                    "Use a common noun instead of a proper noun, "
                    "as shown in the example."
                )
            ),
        ]
        name_specifier_llm = ChatOpenAI(
            model="gpt-3.5-turbo", temperature=1.0
        )
        participant_name = name_specifier_llm.invoke(
            name_specifier_prompt
        ).content
        participant_names.append(participant_name)

    return participant_names


def continue_debate() -> None:
    """
    Unset the new debate flag to signal that the debate has been set up.
    """

    st.session_state.new_debate = False


def reset_debate() -> None:
    """
    Reset all the session state variables.
    """

    st.session_state.topic = ""
    st.session_state.language = "English"
    st.session_state.positive = ""
    st.session_state.negative = ""
    st.session_state.agent_descriptions = {}
    st.session_state.specified_topic = ""
    st.session_state.new_debate = True
    st.session_state.conversations = []
    st.session_state.conversations4print = []
    st.session_state.simulator = None
    st.session_state.names = {}
    st.session_state.tools = []
    st.session_state.retriever_tool = None
    st.session_state.vector_store_message = ""
    st.session_state.conclusions = ""
    st.session_state.comments_key = 0


def set_tools() -> None:
    """
    Set the tools for the agents. Tools that can be selected are
    bing_search, arxiv, and retrieval.
    """

    arxiv = load_tools(["arxiv"])[0]
    st.write("")
    st.write("**Tools**")
    if st.session_state.bing_subscription_validity:
        search = BingSearchAPIWrapper()
        bing_search = Tool(
            name="bing_search",
            description=(
                "A search engine for comprehensive, accurate, and trusted results. "
                "Useful for when you need to answer questions about current events. "
                "Input should be a search query."
            ),
            func=partial(search.results, num_results=5),
            args_schema=MySearchToolInput,
        )
        tool_options = ["Search", "ArXiv", "Retrieval"]
        tool_dictionary = {
            "Search": bing_search,
            "ArXiv": arxiv,
        }
    else:
        tool_options = ["ArXiv", "Retrieval"]
        tool_dictionary = {
            "ArXiv": arxiv,
        }

    st.session_state.selected_tools = st.multiselect(
        label="agent tools",
        options=tool_options,
        label_visibility="collapsed",
    )

    if "Retrieval" in st.session_state.selected_tools:
        # Get the retriever tool and save it to st.session_state.retriever_tool.
        get_retriever()
        if st.session_state.retriever_tool is not None:
            tool_dictionary["Retrieval"] = st.session_state.retriever_tool
        else:
            st.session_state.selected_tools.remove("Retrieval")

    st.session_state.tools = [
        tool_dictionary[key] for key in st.session_state.selected_tools
    ]


def set_debate() -> None:
    """
    Prepare the agents for the debate by setting the topic, the names and
    descriptions of the participants, and the questions for the debate.
    """

    st.write("**Topic of the debate**")
    topic = st.text_input(
        label="topic of the debate",
        placeholder="Enter your topic",
        value=st.session_state.topic,
        label_visibility="collapsed",
    )
    st.session_state.topic = topic.strip()
    st.write(
        "**Language** "
        "<small>used by the debaters</small>",
        unsafe_allow_html=True
    )
    st.session_state.language = st.radio(
        label="language",
        options=("Korean", "English"),
        label_visibility="collapsed",
        index=1,
        horizontal=True
    )

    # Set the tools for the agents
    set_tools()

    left, right = st.columns(2)
    left.write("**Word limit for question suggestions** (≥ 10)")
    # Word limit for task brainstorming
    description_word_limit = left.number_input(
        label="description_word_limit",
        min_value=10,
        max_value=500,
        value=20,
        step=10,
        label_visibility="collapsed"
    )

    right.write("**Word limit for each debate response** (≥ 50)")
    # Word limit for each debate
    st.session_state.word_limit = right.number_input(
        label="answer_word_limit",
        min_value=50,
        max_value=2000,
        value=100,
        step=50,
        label_visibility="collapsed"
    )

    if st.button("Suggest names for the debaters"):
        st.session_state.positive, st.session_state.negative = (
            get_participant_names(topic)
        )

    left, right = st.columns(2)
    left.write("**Name for the positive**")
    positive = left.text_input(
        label="name of the positive",
        value=st.session_state.positive,
        label_visibility="collapsed"
    )
    st.session_state.positive = positive

    right.write("**Name for the negative**")
    negative = right.text_input(
        label="name of the negative",
        value=st.session_state.negative,
        label_visibility="collapsed"
    )
    st.session_state.negative = negative

    st.session_state.names = {
        positive: st.session_state.tools,
        negative: st.session_state.tools,
    }
    conversation_description = (
        "Here is the topic of conversation: "
        f"{append_period(topic)}\nThe participants are: "
        f"{' and '.join(st.session_state.names.keys())}."
    )

    agent_descriptions, agent_system_messages = {}, {}

    if positive and negative:
        if st.button("Suggest descriptions for the debaters"):
            for name in st.session_state.names.keys():
                st.session_state.agent_descriptions[name] = (
                    generate_agent_description(
                        name,
                        conversation_description,
                        st.session_state.language,
                        description_word_limit
                    )
                )

        for name in st.session_state.names.keys():
            st.write(f"**Description for {name}**")
            agent_descriptions[name] = st.text_area(
                label=f"description for {name}",
                value=st.session_state.agent_descriptions.get(name, ""),
                label_visibility="collapsed"
            )
            st.session_state.agent_descriptions[name] = agent_descriptions[name]

        keys_to_delete = [
            key for key in st.session_state.agent_descriptions
            if key not in st.session_state.names
        ]
        for key in keys_to_delete:
            del st.session_state.agent_descriptions[key]

        for name in st.session_state.names.keys():
            agent_system_messages[name] = generate_system_message(
                name,
                conversation_description,
                agent_descriptions[name],
                st.session_state.language,
                st.session_state.word_limit
            )

        if st.button("Suggest questions for the debaters"):
            topic_specifier_prompt = [
                SystemMessage(content="You can make a topic more specific."),
                HumanMessage(
                    content=(
                        "Here is the topic of conversation: "
                        f"{append_period(topic)}\n"
                        "You are the moderator.\n"
                        "Please make the topic more specific.\n"
                        "Please reply with the specified quest in "
                        f"{description_word_limit} words or less in "
                        f"{st.session_state.language}.\n"
                        "Speak directly to the participants: "
                        f"{*st.session_state.names,}.\n"
                        "Do not add anything else."
                    )
                ),
            ]
            topic_specifier_llm = ChatOpenAI(
                model="gpt-3.5-turbo", temperature=1.0
            )
            st.session_state.specified_topic = topic_specifier_llm.invoke(
                topic_specifier_prompt
            ).content

        st.write("**Questions for the debaters**")
        specified_topic = st.text_area(
            label="questions for the debaters",
            value=st.session_state.specified_topic,
            label_visibility="collapsed",
        )
        st.session_state.specified_topic = specified_topic

    if st.session_state.specified_topic:
        if st.button("Prepare the debate"):
            agent_llm = ChatOpenAI(
                model=st.session_state.model, temperature=0.2
            )
            agents = [
                DialogueAgent(
                    name=name,
                    system_message=SystemMessage(content=system_message),
                    llm=agent_llm,
                    tools=tools,
                )
                for (name, tools), system_message in zip(
                    st.session_state.names.items(),
                    agent_system_messages.values()
                )
            ]
            st.session_state.simulator = DialogueSimulator(
                agents=agents, selection_function=select_next_speaker
            )
            st.session_state.simulator.reset()
            st.session_state.simulator.inject("Moderator", specified_topic)
            st.session_state.new_debate = False
            st.rerun()


def print_topic_debaters_questions() -> str:
    """
    Print the topic, the names and descriptions of the participants,
    and questions for the debate.
    """

    st.write("**Topic of the debate**")
    st.info(f"**{st.session_state.topic}**")
    st.write(
        "**Name for the positive**$\:$: "
        f"$~$:blue[{st.session_state.positive}]"
    )
    st.write(
        "**Name for the negative**: "
        f"$~$:blue[{st.session_state.negative}]"
    )
    agent_descriptions = st.session_state.agent_descriptions
    dict_name = "agent_descriptions"
    for name in agent_descriptions.keys():
        st.write(f"**Description for {name}**")
        st.info(agent_descriptions[name])

    st.write("**Moderator**: Here are the questions for the debaters")
    st.info(st.session_state.specified_topic)

    headers = (
        "Topic of the debate: "
        f"{st.session_state.topic}\n\n"
        "Name for the positive: "
        f"{st.session_state.positive}\n"
        "Name for the negative: "
        f"{st.session_state.negative}\n\n"
    )

    if agent_descriptions[st.session_state.positive]:
        headers += (
            f"Description for {st.session_state.positive}:\n"
            f"{locals()[dict_name][st.session_state.positive]}\n\n"
        )
    if agent_descriptions[st.session_state.negative]:
        headers += (
            f"Description for {st.session_state.negative}:\n"
            f"{locals()[dict_name][st.session_state.negative]}\n\n"
        )

    headers += f"Moderator: {st.session_state.specified_topic}\n\n"
    return headers


def conclude_debate() -> None:
    """
    End the debate by providing a summary of the points raised by
    each participant and making a concluding remark. Add this conclusion
    to the list of conversations.
    """

    word_limit = 2 * st.session_state.word_limit
    moderator_prompt = [
        SystemMessage(
            content=(
                "You are the Moderator. "
                "Your goal is to provide a comprehensive summary "
                "highlighting the key points raised by each participant, "
                "and then to conclude the debate in a productive manner. "
                "If there is a clear standout in terms of being more "
                "persuasive or convincing, mention this in your conclusion."
            )
        ),
        HumanMessage(
            content=(
                f"Answer in {word_limit} words or less "
                f"in {st.session_state.language}.\n\n"
                "Here is the complete conversation.\n\n"
                f"{st.session_state.complete_conversations}\n\n"
                "Moderator: "
            )
        ),
    ]
    moderator_llm = ChatOpenAI(
        model=st.session_state.model, temperature=0.2
    )
    with st.spinner("Moderator is thinking..."):
        st.session_state.conclusions = moderator_llm.invoke(
            moderator_prompt
        ).content

    st.session_state.conversations.append(
        f"Moderator: {st.session_state.conclusions}"
    )
    st.session_state.conversations4print.append(
        f"**Moderator**: {st.session_state.conclusions}"
    )


def multi_agent_debate() -> None:
    """
    Let two agents, equipped with tools such as bing search, arxiv,
    and retriever, debate on a given topic. The debate can be concluded
    with a remark and be downloaded.
    """

    st.write("## 📚 Multi-Agent Debate")

    # Initialize all the session state variables
    initialize_session_state_variables()

    # LLM for the debate
    st.session_state.model = "gpt-4-turbo-preview"

    with st.sidebar:
        st.write("")
        st.write("**API Key Selection**")
        choice_api = st.sidebar.radio(
            label="$\\hspace{0.25em}\\texttt{Choice of API}$",
            options=("Your keys", "My keys"),
            label_visibility="collapsed",
            horizontal=True,
        )

        if choice_api == "Your keys":
            validity = "(Verified)" if st.session_state.ready else ""
            st.write(
                "**OpenAI API Key** ",
                f"<small>:blue[{validity}]</small>",
                unsafe_allow_html=True
            )
            openai_api_key = st.text_input(
                label="$\\textsf{Your OPenAI API Key}$",
                type="password",
                placeholder="sk-",
                on_change=check_api_keys,
                label_visibility="collapsed",
            )
            if st.session_state.bing_subscription_validity:
                validity = "(Verified)"
            else:
                validity = ""
            st.write(
                "**Bing Subscription Key** ",
                f"<small>:blue[{validity}]</small>",
                unsafe_allow_html=True
            )
            bing_subscription_key = st.text_input(
                label="$\\textsf{Your Bing Subscription Key}$",
                type="password",
                value="",
                on_change=check_api_keys,
                label_visibility="collapsed",
            )
            authentication = True
        else:
            openai_api_key = st.secrets["OPENAI_API_KEY"]
            bing_subscription_key = st.secrets["BING_SUBSCRIPTION_KEY"]
            langchain_api_key = st.secrets["LANGCHAIN_API_KEY"]
            stored_pin = st.secrets["USER_PIN"]
            st.write("**Password**")
            user_pin = st.text_input(
                label="Enter password", type="password", label_visibility="collapsed"
            )
            authentication = user_pin == stored_pin

        os.environ["BING_SEARCH_URL"] = "https://api.bing.microsoft.com/v7.0/search"

    if authentication:
        if not st.session_state.ready:
            if is_openai_api_key_valid(openai_api_key):
                os.environ["OPENAI_API_KEY"] = openai_api_key
                st.session_state.ready = True

                if choice_api == "My keys":
                    os.environ["BING_SUBSCRIPTION_KEY"] = bing_subscription_key
                    st.session_state.bing_subscription_validity = True
                    os.environ["LANGCHAIN_API_KEY"] = langchain_api_key
                    current_date = datetime.datetime.now().date()
                    date_string = str(current_date)
                    os.environ["LANGCHAIN_PROJECT"] = "agent_debate_" + date_string
                else:
                    if (
                        bing_subscription_key and
                        is_bing_subscription_key_valid(bing_subscription_key)
                    ):
                        os.environ["BING_SUBSCRIPTION_KEY"] = bing_subscription_key
                        st.session_state.bing_subscription_validity = True
                    else:
                        st.session_state.bing_subscription_validity = False
                st.rerun()
            else:
                st.info(
                    """
                    **Enter your OpenAI and Bing Subscription Keys in the sidebar**

                    Get an OpenAI API Key [here](https://platform.openai.com/api-keys)
                    and a Bing Subscription Key [here](https://portal.azure.com/).
                    You can also follow instructions on
                    [this site](https://levelup.gitconnected.com/api-tutorial-how-to-use-bing-web-search-api-in-python-4165d5592a7e)
                    to get your Bing Subscription Key. If you do not plan to search
                    the internet, there is no need to enter your Bing Subscription key.
                    """
                )
                st.image("files/Streamlit_Debate_App.png")
                st.info(
                    """
                    This app is coded by T.-W. Yoon, a professor of systems theory at
                    Korea University. Take a look at some of his other projects:
                    - [LangChain OpenAI Agent](https://langchain-openai-agent.streamlit.app/)
                    - [OpenAI Assistants](https://assistants.streamlit.app/)
                    - [TWY's Playground](https://twy-playground.streamlit.app/)
                    - [Differential equations](https://diff-eqn.streamlit.app/)
                    """
                )
                st.stop()
    else:
        st.info("**Enter the correct password in the sidebar**")
        st.stop()

    with st.sidebar:
        if choice_api == "My keys":
            st.write("")
            st.write("**LangSmith Tracing**")
            langsmith = st.radio(
                label="LangSmith Tracing",
                options=("On", "Off"),
                label_visibility="collapsed",
                index=1,
                horizontal=True
            )
            os.environ["LANGCHAIN_TRACING_V2"] = (
                "True" if langsmith == "On" else "False"
            )

    if st.session_state.new_debate:
        set_debate()
    else:
        with st.sidebar:
            st.write("")
            st.write(f"**Model**: :blue[{st.session_state.model}]")
            st.write(f"**Language**: :blue[{st.session_state.language}]")
            st.write(f"**Word limit**: :blue[{st.session_state.word_limit}]")

            if st.session_state.selected_tools:
                used_tools = (
                    f":blue[{', '.join(st.session_state.selected_tools)}]"
                )
                if len(st.session_state.selected_tools) == 1:
                    st.write(f"**Tool**: {used_tools}")
                else:
                    st.write(f"**Tools**: {used_tools}")
            else:
                st.write(f"**Tool**: :blue[None]")

        headers = print_topic_debaters_questions()
        st.session_state.complete_conversations = (
            headers + "\n\n".join(st.session_state.conversations)
        )

        if st.session_state.conversations:
            label_debate = "$\,$Continue the debate$\,$"
            label_no_of_rounds = "Number of additional rounds"
            value_no_of_rounds = 1
        else:
            label_debate = "$~~~\,$Start the debate$~~~\,$"
            label_no_of_rounds = "Number of rounds in this debate"
            value_no_of_rounds = 5

        st.write("")
        for message in st.session_state.conversations4print:
            st.write(message)

        if not st.session_state.conclusions:
            st.write(f"**{label_no_of_rounds}**")
            c1, _, _ = st.columns(3)
            no_of_rounds = c1.number_input(
                label=f"{label_no_of_rounds}",
                min_value=1,
                max_value=10,
                value=value_no_of_rounds,
                step=1,
                label_visibility="collapsed",
            )

        if st.session_state.conversations and not st.session_state.conclusions:
            st.write("**Facilitative comments by the (human) moderator** (Optional)")
            facilitative_comments = st.text_input(
                label="facilitative_comments",
                value="",
                key="comments" + str(st.session_state.comments_key),
                label_visibility="collapsed",
            )
            if facilitative_comments:
                st.session_state.simulator.inject("Moderator", facilitative_comments)
                st.session_state.conversations.append(
                    f"Moderator: {facilitative_comments}"
                )
                st.session_state.conversations4print.append(
                    f"**Moderator**: {facilitative_comments}"
                )
                st.session_state.comments_key += 1

        left, right = st.columns(2)
        if not st.session_state.conclusions:
            if left.button(f"{label_debate}"):
                run_simulator(no_of_rounds, st.session_state.simulator)
                st.rerun()
            if st.session_state.conversations:
                if right.button("Conclude the debate$\,$"):
                    conclude_debate()
                    st.rerun()
            else:
                if right.button("$~\:$Back to the setting$~\:$"):
                    st.session_state.new_debate = True
                    st.rerun()

        left.download_button(
            label="Download the debate",
            data=st.session_state.complete_conversations,
            file_name="multi_agent_debate.txt",
            mime="text/plain"
        )
        right.button(
            label="$~~\,\:\!$Reset the debate$~~\,\,$",
            on_click=reset_debate
        )

    with st.sidebar:
        st.write("---")
        st.write(
            "<small>**T.-W. Yoon**, Mar. 2024  \n</small>",
            "<small>[LangChain LLM Agent](https://langchain-llm-agent.streamlit.app/)  \n</small>",
            "<small>[OpenAI Assistants](https://assistants.streamlit.app/)  \n</small>",
            "<small>[TWY's Playground](https://twy-playground.streamlit.app/)  \n</small>",
            "<small>[Differential equations](https://diff-eqn.streamlit.app/)</small>",
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    multi_agent_debate()
