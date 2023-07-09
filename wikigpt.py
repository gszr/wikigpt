#import textract
from transformers import GPT2TokenizerFast
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
import os
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from bs4 import BeautifulSoup
from markdown import markdown

def process_vimwiki_folder(dir_name,txt_folder_name):
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
    for dirpath, dnames, fnames in os.walk(dir_name):
        # Only process md files
        ext = ".md"
        print("Reading dir %s..." % dirpath)
        for filename in fnames:
            if filename.endswith(ext):
                print("\tReading file %s... " % filename)
                # Full path to the file
                filepath = os.path.join(dirpath, filename)
                file = open(filepath, mode='r')
                file_contents = file.read()
                file.close()

                # Extract text from the PDF file
                #doc = textract.process(filepath)
                html = markdown(file_contents)
                text = ''.join(BeautifulSoup(html).findAll(text=True))

                # Split the text into chunks
                chunks = text_splitter.create_documents([text])

                # Add chunks to the array
                all_chunks.append(chunks)

    # Return the array of chunks
    return all_chunks

# Create embeddings 
if not "OPENAI_API_KEY" in os.environ:
    raise SystemExit("OPENAI_API_KEY env missing")

embeddings = OpenAIEmbeddings()

# Store embeddings to vector db
all_chunks = process_vimwiki_folder("/home/gs/wiki", "./text");
db =  FAISS.from_documents(all_chunks[0], embeddings) 
for chunk in all_chunks[1:]:
    db_temp = FAISS.from_documents(chunk, embeddings)
    db.merge_from(db_temp)
    
chat_history = []
qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0.1), db.as_retriever())

while True:
    # Get user query
    query = input("Enter a query (type 'exit' to quit): ")
    if query.lower() == "exit":      
        break

    result = qa({"question": query, "chat_history": chat_history})
    chat_history.append((query, result['answer']))
    print(result['answer'])

print("Exited!!!")
