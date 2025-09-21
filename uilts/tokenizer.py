import os
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Union, Iterable, Optional,Iterator
import regex

def to_bytes_tuple(word: str) -> Tuple[bytes]:
        l = list(word.encode("utf-8"))
        l = [bytes([x]) for x in l]
        return tuple(l)

class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        """
        Construct a BPE tokenizer from a given vocabulary, list of merges, and (optionally) special tokens.
        
        Args:
            vocab: A dictionary mapping token IDs to their byte representations.
            merges: A list of tuples representing BPE merge operations.
            special_tokens: Optional list of strings that should be treated as unbreakable tokens.
        """
        self.PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        
        self.vocab = vocab
        self.byte_to_token_id = {v: k for k, v in vocab.items()}
        
        self.merges = merges
        self.merges_priority_map = {pair: i for i, pair in enumerate(self.merges)}
        
        self.special_tokens = special_tokens if special_tokens else []
        
        # self.special_tokens_byte = [token.encode("utf-8") for token in self.special_tokens_byte]
        
        # # Ensure special tokens are in the vocabulary
        # for token_byte in self.special_tokens_byte:
        #     if token_byte not in self.byte_to_token_id:
        #         # Add to vocab if not already present
        #         new_id = len(self.vocab)
        #         self.vocab[new_id] = token_byte
        #         self.byte_to_token_id[token_byte] = new_id
        

    def from_files(cls, vocab_filepath,merges_filepath,special_tokens=None):    
        pass
    
    
    
    def _apply_merges(self, word_byte_tuple: tuple[bytes, ...]) -> list[bytes]:
        """
        Apply BPE merges to a sequence of bytes.
        
        Args:
            byte_tuple: A tuple of single-byte tokens.
            
        Returns:
            A list of merged byte tokens after applying all applicable merges.
        """
        word_byte: list[bytes] = list(word_byte_tuple)
        
        while len(word_byte) > 1:
            # 记录所有合并对
            pairs = set()
            for i in range(len(word_byte) - 1):
                pair = (word_byte[i], word_byte[i+1])
                if pair in self.merges_priority_map:
                    pairs.add(pair)
            
            if not pairs:
                break # 如果剩下的合并对都不在merges字典中，就表示没有应该合并的合并对了，直接返回

            # 找到最佳合并对
            best_pair = min(pairs, key=lambda pair: self.merges_priority_map[pair])

            # 应用最佳合并对
            new_word_byte = []
            i = 0
            while i < len(word_byte):
                if i < len(word_byte) - 1 and (word_byte[i], word_byte[i+1]) == best_pair:
                    new_word_byte.append(word_byte[i] + word_byte[i+1])
                    i += 2
                else:
                    new_word_byte.append(word_byte[i])
                    i += 1
            word_byte = new_word_byte
        return word_byte
    
    def _encode_chunk(self, chunk: str) -> list[int]:
        """
        Encode a chunk of text, which may be a special token or regular text.
        
        Args:
            chunk: The input text chunk to encode.
            
        Returns:
            A list of integer token IDs representing the encoded chunk.
        """
        if chunk in self.special_tokens:
            # 如果chunk是特殊符号，直接编码
            return [self.byte_to_token_id[chunk.encode('utf-8')]]
        else:
            token_ids = []
            # 如果chunk是普通文本，使用BPE算法处理
            # 首先，使用PAT正则表达式将chunk分割为"单词"
            for word in regex.findall(self.PAT, chunk):
                if not word:
                    continue
                # 把word转换为字节片段
                word_byte_tuple = to_bytes_tuple(word)
                # 把word按照merge列表合并
                word_byte_merged = self._apply_merges(word_byte_tuple)
                
                # 将合并后的字节片段转换为token id
                token_ids.extend([self.byte_to_token_id[b] for b in word_byte_merged])
            return token_ids
    
    def encode(self, text: str) -> list[int]:
        """
        Encode an input text string into a sequence of token IDs.
        
        Args:
            text: The input text to encode.
            
        Returns:
            A list of integer token IDs representing the encoded text.
        """
        if not text:
            return []

        # 创建一个正则表达式模式来分割特殊符号
        # 按照长度降序排序，确保更长的符号（例如"<|eot|><|eot|>") 在更短的符号（例如"<|eot|>")之前被匹配
        sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
        special_token_pattern = '|'.join(map(regex.escape, sorted_special_tokens))

        if self.special_tokens:
            # 按照特殊符号分割text，保持特殊符号作为分隔符
            chunks = regex.split(f'({special_token_pattern})', text)
        else:
            chunks = [text]

        token_ids = []
        for chunk in chunks:
            token_ids.extend(self._encode_chunk(chunk))
        return token_ids
    
    def encode_iterable(self, iterable: Iterable[str]) -> iter:
        """
        Given an iterable of strings (e.g., a file handle), yield token IDs lazily.
        
        Args:
            iterable: An iterable source of text chunks.
            
        Yields:
            Token IDs generated by processing the input iterable.
        """
        for chunk in iterable:
            yield from self.encode(chunk)
            """
            两种写法等价
            ids = self.encode(chunk)
                for id in ids:
                    yield id
         """
     
    def decode(self, ids: list[int]) -> str:
        """
        Decode a sequence of token IDs back into a human-readable string.
        
        Args:
            ids: A list of integer token IDs.
            
        Returns:
            The decoded string representation of the input token IDs.
        """
        text_bytes = b''.join([self.vocab[token_id] for token_id in ids])
        return text_bytes.decode("utf-8", errors="replace")
        
