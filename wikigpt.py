#!/bin/env python3

import os
import readline
import atexit
import argparse
import pathlib

from transformers import GPT2TokenizerFast
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain


def readline_setup():
    histfile = os.path.join(os.path.expanduser("~"), ".wikigpt_history")
    histlen = 5000

    try:
        readline.read_history_file(histfile)
        # default history len is -1 (infinite), which may grow unruly
        readline.set_history_length(histlen)
    except FileNotFoundError:
        pass
    atexit.register(readline.write_history_file, histfile)

def init_argparse():
    parser = argparse.ArgumentParser(description="wikigpt",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-p", "--prompt", default = "wikigpt > ", help = "Customize the prompt")
    parser.add_argument("--add-path", dest = "paths", type = pathlib.Path,
                        action = "append", required = True)
    parser.add_argument("-v", "--verbose", default = False, action='store_true')

    args = parser.parse_args()
    global ARGS
    ARGS = args

    return args

def init():
    if not "OPENAI_API_KEY" in os.environ:
        raise SystemExit("OPENAI_API_KEY env missing")

    readline_setup()
    init_argparse()

def read_file_contents(filepath):
    # Full path to the file
    with open(filepath, mode='r', encoding="UTF-8") as file:
        file_contents = file.read()
        return file_contents

def process_path(path):
    # Initialize tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    def count_tokens(text: str) -> int:
        return len(tokenizer.encode(text))

    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 512,
        chunk_overlap  = 24,
        length_function = count_tokens,
    )

    # Array to hold all chunks
    all_chunks = []

    # Iterate over all files in the folder
    for dirpath, _, fnames in os.walk(path):
        # Only process md files
        ext = ".md"

        if ARGS.verbose:
            print("Reading dir %s..." % dirpath)

        for filename in fnames:
            if filename.endswith(ext):
                if ARGS.verbose:
                    print("\tReading file %s... " % filename)
                file_contents = read_file_contents(os.path.join(dirpath, filename))

                # Split the text into chunks
                chunks = text_splitter.create_documents([file_contents])

                # Add chunks to the array
                all_chunks.append(chunks)

    # Return the array of chunks
    return all_chunks

def init_chunks(paths):
    return [ chunks for path in paths for chunks in process_path(path)]

def init_embeddings(chunks):
    embeddings = OpenAIEmbeddings()
    db =  FAISS.from_documents(chunks[0], embeddings)
    for chunk in chunks[1:]:
        db_temp = FAISS.from_documents(chunk, embeddings)
        db.merge_from(db_temp)
    return ConversationalRetrievalChain.from_llm(OpenAI(temperature=0.1), db.as_retriever())

def get_question(prompt):
    return input(prompt).lower().lstrip().rstrip()

def submit_query(question, answerer, history=[]):
    result = answerer({"question": question, "chat_history": history})
    history.append((question, result['answer']))
    return result['answer']

def loop():
    all_chunks = init_chunks(ARGS.paths)
    answerer = init_embeddings(all_chunks)

    while True:
        question = get_question(ARGS.prompt)
        match question:
            case "":
                continue
            case "exit":
                break
            case _:
                print(submit_query(question, answerer))


def main():
    init()
    loop()

if __name__ == "__main__":
    main()
