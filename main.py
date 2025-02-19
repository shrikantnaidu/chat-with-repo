import streamlit as st
from utils.process import process_repository
from utils.chat import run_chat_app, initialize_session_state, setup_sidebar, initialize_database

def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="AI Chat with Repository Memory",
        page_icon="💭",
        layout="wide"
    )

    st.title("💬 RepoChat")

    # Initialize session state
    initialize_session_state()

    # Setup sidebar
    setup_sidebar()

    # Run chat app (it will handle the database initialization internally)
    run_chat_app()

if __name__ == "__main__":
    main()
