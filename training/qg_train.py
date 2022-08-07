import argparse
import datasets
#from transformers import T5Config, T5ForConditionalGeneration, T5Tokenizer
from transformers import MT5Config, MT5ForConditionalGeneration, MT5Tokenizer

from dataset import QGDataset
from trainer import Trainer

from datasets import load_dataset

from sklearn.model_selection import train_test_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataloader_workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--qg_model", type=str, default="google/mt5-small")
    parser.add_argument("--pad_mask_id", type=int, default=-100)
    parser.add_argument("--pin_memory", dest="pin_memory", action="store_true", default=False)
    parser.add_argument("--save_dir", type=str, default="./mt5-base-question-generator")
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--valid_batch_size", type=int, default=32)
    return parser.parse_args()


def get_tokenizer(checkpoint: str) -> MT5Tokenizer:
    tokenizer = MT5Tokenizer.from_pretrained(checkpoint)
    tokenizer.add_special_tokens(
        {'additional_special_tokens': ['<answer>', '<context>']}
    )
    return tokenizer


def get_model(checkpoint: str, device: str, tokenizer: MT5Tokenizer) -> MT5ForConditionalGeneration:
    config = MT5Config(decoder_start_token_id=tokenizer.pad_token_id)
    model = MT5ForConditionalGeneration(config).from_pretrained(checkpoint)
    model.resize_token_embeddings(len(tokenizer))
    model = model.to(device)
    return model


if __name__ == "__main__":
    args = parse_args()
    tokenizer = get_tokenizer(args.qg_model)
    data_files = {"train": ["train_en.csv", "train_zh.csv"], "validation": ["validation_en.csv", "validation_zh.csv"]}
    dataset = load_dataset("csv", data_files=data_files)
    #dataset = datasets.load_dataset("iarfmoose/question_generator")
    print(dataset['train'])
    print(dataset['validation'])
    

    train_set = QGDataset(dataset["train"], args.max_length, args.pad_mask_id, tokenizer)
    valid_set = QGDataset(dataset["validation"], args.max_length, args.pad_mask_id, tokenizer)

    print(train_set)
    print(valid_set)

    model = get_model(args.qg_model, args.device, tokenizer)
    trainer = Trainer(
        dataloader_workers=args.dataloader_workers,
        device=args.device,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        model=model,
        pin_memory=args.pin_memory,
        save_dir=args.save_dir,
        tokenizer=tokenizer,
        train_batch_size=args.train_batch_size,
        train_set=train_set,
        valid_batch_size=args.valid_batch_size,
        valid_set=valid_set
    )
    trainer.train()
