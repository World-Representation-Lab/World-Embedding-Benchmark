from datasets import load_dataset


dataset = load_dataset(
    "World-Representation-Lab/World-Embedding-Solid-Retrieval",
    split="test",
)
print(dataset)
