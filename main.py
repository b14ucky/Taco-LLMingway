from src.model import GPT
from src.train import Trainer
from src.tokenizer import Tokenizer
from src.logger import get_logger
from src.utils import load_dataset_from_txt, create_sequences, get_dataset
from pathlib import Path
from torch.utils.data import DataLoader


DATASET_PATH = Path("data/raw")

CONTEXT_LEN = 32
EMBED_DIM = 64
N_HEADS = 2
FFN_DIM = 128
N_BLOCKS = 2
DROPOUT = 1
BATCH_SIZE = 16
LR = 1e-3

N_EPOCHS = 2
FINAL_MODEL_PATH = Path("model/taco-llmingway.pth")
CHECKPOINT_DIR = Path("checkpoints")

logger = get_logger(__name__)


def main() -> None:
    dataset_txt = load_dataset_from_txt(DATASET_PATH, logger=logger)
    if dataset_txt is None:
        return None

    dataset_txt = dataset_txt.replace("\n", " ")
    logger.debug(f"Number of characters in dataset: {len(dataset_txt)}")
    logger.debug(f"Number of words in dataset: {len(dataset_txt.split(" "))}")

    tokenizer = Tokenizer(dataset_txt, tokenization_mode="character")
    vocab_size = len(tokenizer)

    tokens = tokenizer.encode(dataset_txt)
    dataset = get_dataset(*create_sequences(tokens, context_len=CONTEXT_LEN))
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)
    logger.debug(f"Dataset len: {len(dataset)}")
    logger.debug(f"Vocab size: {vocab_size}")

    model = GPT(
        embed_dim=EMBED_DIM,
        vocab_size=vocab_size,
        context_len=CONTEXT_LEN,
        n_heads=N_HEADS,
        ffn_dim=FFN_DIM,
        n_blocks=N_BLOCKS,
        dropout=DROPOUT,
    )

    logger.debug(model)

    trainer = Trainer(
        model=model,
        dataloader=dataloader,
        logger=logger,
        lr=LR,
    )

    trainer.train(
        n_epochs=N_EPOCHS,
        final_model_path=FINAL_MODEL_PATH,
        checkpoint_dir=CHECKPOINT_DIR,
    )


if __name__ == "__main__":
    main()
