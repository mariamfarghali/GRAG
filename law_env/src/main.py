from src.trial3 import read_file, write_chunks_to_file, TextChunker, Embedder, Tokenizer
import asyncio
import os


async def test_chunking_and_embedding():
    # File path
    filename = r"D:\\GRAG\\law_env\\raw_data\\newLaw.txt"

    # Initialize splitter, chunker, embedder
    tokenizer_mode = Tokenizer("xlm-roberta-base")

    chunker = TextChunker(tokenizer=tokenizer_mode, max_tokens=400, min_tokens=50)
    embedder = Embedder(embedder_model="LLukas22/paraphrase-multilingual-mpnet-base-v2-embedding-all")

    # Read document
    content = read_file(filename)

    # Chunk text
    chunks = await chunker.chunk_text(content)

    # Write chunks for inspection
    output_path = os.path.join(os.path.dirname(filename), "chunked_output3.txt")
    write_chunks_to_file(chunks, output_path, tokenizer_mode)

    # Summary
    print(f"\nOriginal document token count: {tokenizer_mode.count_tokens(content)}")
    print(f"Number of chunks created: {len(chunks)}")

    # Token details
    for i, chunk in enumerate(chunks):
        token_count = tokenizer_mode.count_tokens(chunk)
        status = "OK"
        if token_count > 400:
            status = "ERROR: Exceeds 400 tokens!"
        elif token_count <= 50:
            status = "WARNING: Has <=50 tokens!"
        print(f"Chunk {i+1}: {token_count} tokens - {status}")

    # Embed chunks
    embeddings = await embedder.embed(chunks)
    print(f"\nEmbeddings ready: {embeddings.shape}")


if __name__ == "__main__":
    asyncio.run(test_chunking_and_embedding())
