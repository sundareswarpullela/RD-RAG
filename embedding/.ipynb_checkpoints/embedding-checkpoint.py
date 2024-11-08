from llama_index.core import SimpleDirectoryReader, VectorStoreIndex

files_path = "data-curation/data/files"

reader = SimpleDirectoryReader(input_dir=files_path)
documents = reader.load_data()
index = VectorStoreIndex.from_documents(documents)

query_engine = index.as_query_engine()
response = query_engine.query("What did the author do growing up?")
print(response)