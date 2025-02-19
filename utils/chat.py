import os
import streamlit as st
from langchain_community.vectorstores import DeepLake
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from streamlit_chat import message
import groq
from langchain_groq import ChatGroq
from utils.process import process_repository

# Define available Groq models
GROQ_MODELS = {
    "Mixtral 8x7B": "mixtral-8x7b-32768",
    "DeepSeek R1 Distill LLaMA 70B": "deepseek-r1-distill-llama-70b",
    "LLaMA3 8B": "llama3-8b-8192",
    "Qwen 2.5 32B": "qwen-2.5-32b"
}

def initialize_session_state():
    """Initialize session state variables."""
    if "generated" not in st.session_state:
        st.session_state["generated"] = []
    if "past" not in st.session_state:
        st.session_state["past"] = []
    if "groq_api_key" not in st.session_state:
        st.session_state["groq_api_key"] = ""
    if "activeloop_token" not in st.session_state:
        st.session_state["activeloop_token"] = ""
    if "db" not in st.session_state:
        st.session_state["db"] = None
    if "selected_model" not in st.session_state:
        st.session_state["selected_model"] = "mixtral-8x7b-32768"
    if "temperature" not in st.session_state:  # Add temperature control
        st.session_state["temperature"] = 0.7

def setup_sidebar():
    """Setup sidebar for API key inputs and configuration."""
    with st.sidebar:
        groq_key = st.text_input("Groq API Key", key="groq_api_key_input")
        activeloop_token = st.text_input("ActiveLoop Token", key="activeloop_token_input")

        # Save the tokens to session state
        st.session_state["activeloop_token"] = activeloop_token
        st.session_state["groq_key"] = groq_key

        # Model selection
        selected_model = st.selectbox(
            "Select Model",
            options=list(GROQ_MODELS.keys()),  # Use the keys from the GROQ_MODELS dictionary
            index=0,  # Default to the first model
            key="selected_model_input"
        )
        st.session_state["selected_model"] = GROQ_MODELS[selected_model]  # Store the selected model in session state

        data_source = st.radio(
            "Choose your data source:",
            ["Use Existing Dataset", "Process New Repository"],
            key="data_source_input"
        )

        if data_source == "Process New Repository":
            st.subheader("Repository Settings")
            
            # Add ActiveLoop username input
            activeloop_username = st.text_input(
                "ActiveLoop Username",
                placeholder="your-activeloop-username",
                key="activeloop_username_input"
            )
            
            repo_url = st.text_input(
                "GitHub Repository URL",
                placeholder="https://github.com/username/repository",
                key="repo_url_input"
            )
            
            file_extensions = st.multiselect(
                "File Extensions to Process",
                [".py", ".js", ".ts", ".html", ".css", ".md", ".txt", ".json", ".yaml", ".yml"],
                default=[".py", ".md", ".txt"],
                key="file_extensions_input"
            )
            
            dataset_name = st.text_input(
                "New Dataset Name",
                placeholder="my-repo-dataset",
                key="dataset_name_input"
            )

            if st.button("Process Repository", type="primary", key="process_repo_button"):
                if not repo_url or not dataset_name or not activeloop_username:
                    st.error("Please provide repository URL, dataset name, and ActiveLoop username.")
                else:
                    activeloop_dataset_path = f"hub://{activeloop_username}/{dataset_name}"
                    
                    with st.spinner("Processing repository..."):
                        success = process_repository(
                            repo_url=repo_url,
                            include_file_extensions=file_extensions,
                            activeloop_dataset_path=activeloop_dataset_path,
                            repo_destination=None
                        )
                        if success:
                            st.session_state["db"] = activeloop_dataset_path
                            st.success("Repository processed successfully!")

        else:  # Use Existing Dataset
            st.subheader("Existing Dataset Settings")
            existing_dataset_path = st.text_input(
                "Dataset Path",
                placeholder="hub://username/dataset-name",
                key="existing_dataset_path_input"
            )

            if st.button("Load Dataset", type="primary", key="load_dataset_button"):
                if existing_dataset_path:
                    st.session_state["db"] = existing_dataset_path  # Ensure this is a string
                    st.success("Dataset loaded successfully!")
                else:
                    st.error("Please provide a dataset path.")

        # Add Initialize Chat button
        if st.button("Initialize Chat", key="initialize_chat_button"):
            if st.session_state.get("db") is not None:
                initialize_database(st.session_state["db"])
                st.success("Chat initialized successfully!")
            else:
                st.error("Please configure your API keys and dataset before initializing the chat.")

def initialize_database(dataset_path):
    """Initialize the DeepLake database with the given dataset path."""
    try:
        activeloop_token = st.session_state.get("activeloop_token")
        
        if not activeloop_token:
            st.error("ActiveLoop token is not set.")
            return
        
        # Debugging output to check the dataset path
        # print(f"Dataset path: {dataset_path}, Type: {type(dataset_path)}")  # Debugging line
        
        # if not isinstance(dataset_path, str):
        #     st.error("Dataset path must be a string.")
        #     return
        
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
        
        db = DeepLake(
            dataset_path=dataset_path,
            read_only=True,
            embedding=embeddings,
            token=activeloop_token  # Ensure the token is passed here
        )
        st.session_state["db"] = db
    except Exception as e:
        st.error(f"Error initializing database: {str(e)}")

def generate_response(prompt):
    """Generate a response using Groq's API."""
    try:
        client = groq.Groq(api_key=st.session_state.get("groq_api_key"))
        completion = client.chat.completions.create(
            model=st.session_state["selected_model"],
            messages=[{"role": "user", "content": prompt}],
            temperature=st.session_state["temperature"]
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error generating response: {str(e)}"

def search_db(db, query):
    """Search for a response to the query in the DeepLake database."""
    try:
        retriever = db.as_retriever()
        retriever.search_kwargs = {
            "k": 10,
            "distance_metric": "cos",
            "fetch_k": 100,
        }
        
        model = ChatGroq(
            model_name=st.session_state["selected_model"],
            # temperature=st.session_state["temperature"]
        )
        qa = RetrievalQA.from_llm(
            model, 
            retriever=retriever,
            verbose=True
        )
        result = qa.invoke(query)
        
        # Format the response
        if isinstance(result, dict):
            response = result.get('result', result.get('answer', ''))
            response = ' '.join(response.split())
            return response.strip()  # Return only the result string
        
        return str(result).strip()  # Ensure we return only the result string
    except Exception as e:
        return f"Error searching database: {str(e)}"

def run_chat_app():
    """Run the chat application using the Streamlit framework."""
    if st.session_state.get("db") is not None:
        # Add a temperature slider in the chat interface
        st.sidebar.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
            key="temperature",
            help="Controls randomness in the response. Lower values make responses more focused and deterministic."
        )

        user_input = st.chat_input("Type your message here...")

        if user_input:
            with st.spinner("Thinking..."):
                output = search_db(st.session_state["db"], user_input)
                st.session_state.past.append(user_input)
                st.session_state.generated.append(output)

        # Display chat messages
        if st.session_state["generated"]:
            for i in range(len(st.session_state["generated"]) - 1, -1, -1):
                message(st.session_state["generated"][i], key=str(i))
                message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
    else:
        st.info("Please configure your API keys and dataset in the sidebar to start chatting.")

# if __name__ == "__main__":
#     run_chat_app()
