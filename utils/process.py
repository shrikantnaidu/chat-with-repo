import deeplake
import os
import pathspec
import subprocess
from langchain_community.document_loaders import TextLoader
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import DeepLake
import streamlit as st
import shutil
import time
import stat
import errno

def clone_repository(repo_url, local_path):
    """Clone the specified git repository to the given local path."""
    try:
        subprocess.run(["git", "clone", repo_url, local_path], check=True)
        return True
    except subprocess.CalledProcessError:
        st.error(f"Failed to clone repository: {repo_url}")
        return False


def load_docs(root_dir, file_extensions=None):
    """
    Load documents from the specified root directory.
    Ignore dotfiles, dot directories, and files that match .gitignore rules.
    Optionally filter by file extensions.
    """
    docs = []

    # Load .gitignore rules
    gitignore_path = os.path.join(root_dir, ".gitignore")

    if os.path.isfile(gitignore_path):
        with open(gitignore_path, "r") as gitignore_file:
            gitignore = gitignore_file.read()
        spec = pathspec.PathSpec.from_lines(
            pathspec.patterns.GitWildMatchPattern, gitignore.splitlines()
        )
    else:
        spec = None

    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Remove dot directories
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]

        for file in filenames:
            file_path = os.path.join(dirpath, file)

            # Skip dotfiles
            if file.startswith("."):
                continue

            # Skip files that match .gitignore rules
            if spec and spec.match_file(file_path):
                continue

            if file_extensions and os.path.splitext(file)[1] not in file_extensions:
                continue

            try:
                loader = TextLoader(file_path, encoding="utf-8")
                docs.extend(loader.load_and_split())
            except Exception as e:
                st.warning(f"Skipping file {file_path}: {str(e)}")
    return docs


def split_docs(docs):
    """Split the input documents into smaller chunks."""
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    return text_splitter.split_documents(docs)


def create_deeplake_dataset(activeloop_dataset_path, activeloop_token):
    """Create an empty DeepLake dataset with the specified path and token."""
    try:
        ds = deeplake.empty(
            activeloop_dataset_path,
            token=activeloop_token,
            overwrite=True,
        )
        
        ds.create_tensor("ids")
        ds.create_tensor("metadata")
        ds.create_tensor("embedding")
        ds.create_tensor("text")
        return True
    except Exception as e:
        st.error(f"Failed to create DeepLake dataset: {str(e)}")
        return False


def get_unique_repo_path(base_path="repos"):
    """Generate a unique repository path based on timestamp."""
    timestamp = int(time.time())
    return f"{base_path}_{timestamp}"

def handle_remove_readonly(func, path, exc):
    """Handle read-only files when removing directories."""
    excvalue = exc[1]
    if func in (os.rmdir, os.remove, os.unlink) and excvalue.errno == errno.EACCES:
        # Add write permission to the file
        os.chmod(path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
        # Try again
        func(path)
    else:
        raise excvalue

def safe_remove_directory(path, max_retries=5, delay=1):
    """Safely remove a directory with retries, especially for Windows Git repos."""
    for i in range(max_retries):
        try:
            if os.path.exists(path):
                # Use the custom error handler for read-only files
                shutil.rmtree(path, onerror=handle_remove_readonly)
            return True
        except Exception as e:
            if i == max_retries - 1:  # Last attempt
                st.warning(f"Could not remove directory {path} after {max_retries} attempts: {str(e)}")
                return False
            time.sleep(delay)  # Wait before retrying
    return False

def clean_old_repos(keep_latest=5):
    """Clean up old repository directories, keeping the most recent ones."""
    try:
        # List all repo directories
        repo_dirs = [d for d in os.listdir() if d.startswith("repos_")]
        
        # Sort by creation time (newest first)
        repo_dirs.sort(key=lambda x: os.path.getctime(x), reverse=True)
        
        # Remove old directories
        for old_dir in repo_dirs[keep_latest:]:
            safe_remove_directory(old_dir)
    except Exception as e:
        st.warning(f"Error while cleaning old repositories: {str(e)}")

def process_repository(repo_url, include_file_extensions, activeloop_dataset_path, repo_destination):
    """
    Process a git repository by cloning it, filtering files, splitting documents,
    creating embeddings, and storing everything in a DeepLake dataset.
    """
    try:
        activeloop_token = st.session_state.get("activeloop_token")
        if not activeloop_token:
            st.error("ActiveLoop token not found. Please configure it in the sidebar.")
            return False
            
        with st.spinner("Creating DeepLake dataset..."):
            if not create_deeplake_dataset(activeloop_dataset_path, activeloop_token):
                return False

        # Clean old repositories and get a unique path for the new one
        clean_old_repos()
        unique_repo_path = get_unique_repo_path()

        with st.spinner("Cloning repository..."):
            if not clone_repository(repo_url, unique_repo_path):
                safe_remove_directory(unique_repo_path)  # Clean up if clone fails
                return False

        with st.spinner("Loading and processing documents..."):
            docs = load_docs(unique_repo_path, include_file_extensions)
            if not docs:
                st.error("No documents found in the repository.")
                safe_remove_directory(unique_repo_path)  # Use safe removal
                return False
            
            texts = split_docs(docs)

        with st.spinner("Creating embeddings and storing in DeepLake..."):
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-mpnet-base-v2"
            )
            
            db = DeepLake(
                dataset_path=activeloop_dataset_path,
                embedding=embeddings,
                token=activeloop_token
            )
            db.add_documents(texts)
        
        # Clean up the repository directory after successful processing
        safe_remove_directory(unique_repo_path)  # Use safe removal
        
        st.session_state["db"] = db
        st.success("Repository processed successfully!")
        return True
    
    except Exception as e:
        # Clean up the repository directory if an error occurred
        if 'unique_repo_path' in locals():
            safe_remove_directory(unique_repo_path)  # Use safe removal
        st.error(f"An error occurred while processing the repository: {str(e)}")
        return False
