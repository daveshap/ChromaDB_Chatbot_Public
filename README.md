# ChromaDB Chatbot

Public version of my ChromaDB chatbot that keeps track of user profile and historical topics. Should be mostly ready to go right out of the box. 

## Setup

1. Install chromadb and openai (in `requirements.txt` file)
2. Update `user_profile.txt` file with your initial information
3. Update `key_openai.txt` with your OpenAI API key

## Usage

- Main chat client: `python chat.py`
- Take a look in your KB: `python chromadb_peek.py`

## Code Explanation

This Python script serves as the implementation of a chatbot that leverages the OpenAI's GPT-4 model. It additionally integrates the chatbot with a persistent knowledge base using the ChromaDB library. Here's an overview of how the different parts of the script function:

1. **Utility Functions**: The script starts with several utility functions to handle file operations and to interact with OpenAI's API. These include functions for saving and opening files, and a function to run the chatbot, managing retries in case of exceptions.
2. **Main Application**: The script's main operation is contained within a continuous loop (`while True:`), enabling continuous interaction with the user. This loop does the following:
   - **Instantiates the ChromaDB client** for persistent storage and knowledge base management.
   - **Initiates the chatbot** by loading OpenAI's API key and preparing a conversation list.
   - **Captures user input** and adds it to the conversation list. The input is also logged in a separate file for record-keeping.
   - **Searches the knowledge base** for relevant content based on the current conversation and updates the chatbot's default system message accordingly.
   - **Generates a response** from the chatbot based on the conversation so far, which includes the updated default system message and the user's input.
   - **Updates the user profile** based on the user's recent messages, using the chatbot's response as the updated profile.
   - **Updates the knowledge base** with the most recent conversation, either adding a new entry or updating an existing entry. If an existing entry becomes too long, it's split into two separate entries.

The script logs all interactions with the OpenAI API and updates to the knowledge base, providing a record of the chatbot's operations and aiding in debugging and optimization efforts. The use of the ChromaDB library allows for scalable storage and retrieval of the chatbot's knowledge base, accommodating a growing number of conversations and data points.

But seriously just look at the code, it's pretty straight forward. 

## Contributing

You're welcome to submit a pull request to make mild changes or fix bugs. Any substantial refactors will be rejected. If you want to take this work and modify it, please just work on your own fork. This repo will eventually be made a public readonly archive. 
