from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
import os

DATA_DIR="data"
TOK_DIR="tokenizers"

def tokenizer_49k():
    ''' Trains a BPE tokenizer with a vocab size of 49,152 using the climbmix dataset.
    The trained tokenizer is saved as tokenizer_49k.json in the current directory.

    Configuration Details:
    - Model: BPE (Byte Pair Encoding)
    - Pre-tokenizer: ByteLevel (splits on byte level, allowing for better handling of code and special characters)
    - Special Tokens: <PAD>, <UNK>, <BOS>, <EOS>
    - Vocabulary Size: 49,152 tokens
    - Output File: tokenizers/tokenizer_49k.json

    Dataset Requirements:
    - data/climbmix.txt: Training data (e.g., code or technical documentation)
    
    
    Raises:
        FileNotFoundError: If required data files are missing in the data directory
        PermissionError: If unable to write to the tokenizers directory
    '''

    tok=Tokenizer(BPE())
    tok.pre_tokenizer=ByteLevel()
    trainer=BpeTrainer(
        vocab_size=49_152,
        special_tokens=["<PAD>","<UNK>","<BOS>","<EOS>"],
        show_progress=True,
    )
    tok.train(
        files=[
            os.path.join(DATA_DIR,"climbmix.txt"),
        ],
        trainer=trainer
    )
    tok.save(os.path.join(TOK_DIR, "tokenizer_49k.json"))

def tokenizer_32k():
    ''' Trains a BPE tokenizer with a vocab size of 32,000 using the climbmix dataset.
    The trained tokenizer is saved as tokenizer_32k.json in the current directory.
    
    Configuration Details:
    - Model: BPE (Byte Pair Encoding)
    - Pre-tokenizer: ByteLevel (splits on byte level, allowing for better handling of code and special characters)
    - Special Tokens: <PAD>, <UNK>, <BOS>, <EOS>
    - Vocabulary Size: 32,768 tokens
    - Output File: tokenizers/tokenizer_32k.json
    
    Dataset Requirements:
    - data/climbmix.txt: Training data (e.g., code or technical documentation)

    
    Raises:
        FileNotFoundError: If required data files are missing in the data directory
        PermissionError: If unable to write to the tokenizers directory
    '''

    tok=Tokenizer(BPE())
    tok.pre_tokenizer=ByteLevel()
    trainer=BpeTrainer(
        vocab_size=32_768,
        special_tokens=["<PAD>","<UNK>","<BOS>","<EOS>"],
        show_progress=True,
    )
    tok.train(
        files=[
            os.path.join(DATA_DIR,"climbmix.txt"),
        ],
        trainer=trainer
    )
    tok.save(os.path.join(TOK_DIR, "tokenizer_32k.json"))

if __name__=="__main__":
    os.makedirs(TOK_DIR,exist_ok=True)

    print("Training Tokenizer with vocab size 49k...")
    tokenizer_49k()

    print("------------------------------------------------------------------------------")

    print("Training Tokenizer with vocab size 32k...")
    tokenizer_32k()
