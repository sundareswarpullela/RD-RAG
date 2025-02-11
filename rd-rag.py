import subprocess
import argparse
import logging


from embedding.embedding import Embedder, embed_bioasq
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Mode selector")
    subparser = parser.add_subparsers(dest = "command", required=True)


    embed_parser = subparser.add_parser("embed",  help="Command to generate embeddings")
    embed_parser.add_argument("model", type=str, help="Embedder model")

    db_parser = subparser.add_parser("rundb", help="Command to run vector db")
    # db_parser.add_argument("port", type=int, help="Port to host vector DB")

    generate_parser = subparser.add_parser("generate", help="Command to generate results")
    generate_parser.add_argument("model", type=str, help="Embedder model")

    data_path =  "data-curation/data/source_files/filtered_rare_disease_data.json"
    vector_db_path = "vectordb"
    args = parser.parse_args()
    # print(args.model)

    if args.command == "embed":
        if args.model not in Embedder.embedder_map:
            raise ValueError(f"Invalid model name {args.model}")
        else:
            log.info(f"Embedding model: {args.model}")
            embedder = Embedder(args.model)
            embed_bioasq(embedder, data_path)


    elif args.command == "rundb":
        # log.info(f"Vector DB model: {args.model}")
        # log.info(f"Port: {args.port}")
        subprocess.run(["chroma", "run", "--path", f"{vector_db_path}"])

    elif args.command == "generate":
        pass

    else:
        raise ValueError(f"Invalid command {args.command}")

