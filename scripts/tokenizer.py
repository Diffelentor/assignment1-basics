import os
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Union, Iterable, Optional,Iterator
import regex
import json
import torch
import numpy as np

def gpt2_bytes_to_unicode() -> dict[int, str]:
    """
    Returns a mapping between every possible byte (an integer from 0 to 255) to a
    printable unicode string character representation. This function is taken
    from the GPT-2 code.

    For example, `chr(0)` is `\x00`, which is an unprintable character:

    >>> chr(0)
    '\x00'
    >>> print(chr(0))

    As a result, this function returns a dictionary `d` where `d[0]` returns `Ā`.
    The bytes that are visually printable keep their original string representation [1].
    For example, `chr(33)` returns `!`, and so accordingly `d[33]` returns `!`.
    Note in particular that the space character `chr(32)` becomes `d[32]`, which
    returns 'Ġ'.

    For unprintable characters, the function shifts takes the integer representing
    the Unicode code point of that character (returned by the Python `ord`) function
    and shifts it by 256. For example, `ord(" ")` returns `32`, so the the space character
    ' ' is shifted to `256 + 32`. Since `chr(256 + 32)` returns `Ġ`, we use that as the
    string representation of the space.

    This function can simplify the BPE implementation and makes it slightly easier to
    manually inspect the generated merges after they're serialized to a file.
    """
    # These 188 integers can used as-is, since they are not whitespace or control characters.
    # See https://www.ssec.wisc.edu/~tomw/java/unicode.html.
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    # now get the representations of the other 68 integers that do need shifting
    # each will get mapped chr(256 + n), where n will grow from 0...67 in the loop
    # Get printable representations of the remaining integers 68 integers.
    n = 0
    for b in range(2**8):
        if b not in bs:
            # If this integer isn't in our list of visually-representable
            # charcters, then map it to the next nice character (offset by 256)
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    characters = [chr(n) for n in cs]
    d = dict(zip(bs, characters))
    return d

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
        self.vocab_size = len(vocab)
        self.byte_to_token_id = {v: k for k, v in vocab.items()}
        
        self.merges = merges
        self.merges_priority_map = {pair: i for i, pair in enumerate(self.merges)}
        
        self.special_tokens = special_tokens if special_tokens else []
        if "<|endoftext|>" in self.special_tokens:
            self.eos_token = "<|endoftext|>"
            self.eos_token_id = self.byte_to_token_id[self.eos_token.encode('utf-8')]
        
         
    @classmethod
    def from_files(cls, vocab_filepath,merges_filepath,special_tokens=None):    
        """
        从 vocab 和 merges 文件中构造 Tokenizer
        vocab: json 文件，id -> token(byte string base64/utf-8)
        merges: txt 文件，第一行是头（如 "#version: ..."), 后面每行是两个token拼接
        """

        # Compare the learned merges to the expected output merges
        gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
        with open(merges_filepath, encoding="utf-8") as f:
            gpt2_merges = [tuple(line.rstrip().split(" ")) for line in f]
            merges = [
                (
                    bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
                    bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
                )
                for merge_token_1, merge_token_2 in gpt2_merges
            ]

        # Compare the vocab to the expected output vocab
        with open(vocab_filepath, encoding="utf-8") as f:
            gpt2_vocab = json.load(f)
            vocab = {
                gpt2_vocab_index: bytes([gpt2_byte_decoder[token] for token in gpt2_vocab_item])
                for gpt2_vocab_item, gpt2_vocab_index in gpt2_vocab.items()
            }   

        # 3. 实例化 tokenizer
        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)
    
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
    
    def encode_tensor(self, text: str, device=None) -> torch.Tensor:
        ids = self.encode(text)
        return torch.tensor(ids, device=device)
    
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
     
    def decode(self, ids: list[int] | torch.Tensor, skip_special_tokens: bool = False) -> str:
        """
        Decode a sequence of token IDs back into a human-readable string.
        
        Args:
            ids: A list of integer token IDs.
            
        Returns:
            The decoded string representation of the input token IDs.
        """
        
        # 如果输入是 tensor，先 flatten 再转成 list
        if isinstance(ids, torch.Tensor):
            ids = ids.flatten().tolist()

        # 过滤掉特殊 token
        if skip_special_tokens and self.special_tokens:
            special_token_ids = {self.byte_to_token_id[t.encode("utf-8")] for t in self.special_tokens}
            ids = [i for i in ids if i not in special_token_ids]

        # 按顺序拼接字节
        text_bytes = b''.join([self.vocab[token_id] for token_id in ids])
        return text_bytes.decode("utf-8", errors="replace")

       
    @classmethod 
    def from_txt(
        cls,
        input_path: str | os.PathLike,
        vocab_size: int,
        special_tokens: list[str],
        **kwargs,
    ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

        # 初始化词表和合并规则
        vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}  # 单字节字符初始化
        next_id = 256

        # 处理特殊标记
        for token in special_tokens:
            vocab[next_id]=token.encode("utf-8")
            next_id+=1
        
        # 读取输入文件
        with open(input_path, "r", encoding="utf-8") as f:
            text = f.read()
        
        # 第3步对语料库里的文段进行预分词pre-tokenization：分割文本时保存标点和空格，得到“单词”列表['Hello', ',', ' world', '!', ' This', ' is', ' a', ' test', '.']
        token_seq_frequency_table = defaultdict(int) # 这里的tokens是token组，例如tuple(['h', 'e', 'l', 'l', 'o'])
        chunks = regex.split("|".join(map(regex.escape, special_tokens)), text) #首先按照特殊字符进行大分割，比如<endoftext>按照章节分割
        # 然后在大分割里小分割，按照空格和标点
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        for chunk in chunks:
            for word in regex.findall(PAT, chunk):
                word_bytes = word.encode("utf-8") #对每一个单词进行编码，并转换为bytes
                bytes_list = [bytes([x]) for x in word_bytes] #e.g. ['h', 'e', 'l', 'l', 'o']
                token_seq_frequency_table[tuple(bytes_list)] += 1 #统计每个token出现的频率

        merges: List[Tuple[bytes, bytes]] = [] # 用于存储合并操作记录
     
        # BPE 训练
        while len(vocab) < vocab_size:
            # 统计相邻 token 的频率
            pair_counts = defaultdict(int)
            for token, cnt in token_seq_frequency_table.items():
                for i in range(len(token) - 1):
                    pair_counts[(token[i], token[i+1])] += cnt

            # 如果没有更多的对可以合并，停止训练
            if not pair_counts:
                break

            # 找到最频繁的对
            max_count = max(pair_counts.values())
            # 找出所有频率最高的对，可能不止一个
            candidates = [k for k, v in pair_counts.items() if v == max_count]
            # 在候选者中，选择字节序最大的那个
            best_pair = max(candidates)

            # 将最频繁的对加入词表
            new_token = best_pair[0] + best_pair[1]
            vocab[next_id] = new_token #best_pair的两个字符合并成一个新token
            merges.append(best_pair)
            next_id += 1

            affected_token_seq = []
            for token_seq, freq in token_seq_frequency_table.items():
                has_pair = any(token_seq[i:i+2] == best_pair for i in range(len(token_seq) - 1))
                if has_pair:
                    affected_token_seq.append((token_seq, freq))
            #从受影响的token中出发,每个token就是token_frequency_table的key
            for token_seq, freq in affected_token_seq:
                new_seq = []
                i = 0
                while i < len(token_seq):
                    # 检查当前位置是否是最佳对的开始
                    if i < len(token_seq) - 1 and (token_seq[i], token_seq[i+1]) == best_pair:
                        new_seq.append(new_token)
                        i += 2
                    else:
                        new_seq.append(token_seq[i])
                        i += 1
                new_seq=tuple(new_seq)
                del token_seq_frequency_table[token_seq]
                token_seq_frequency_table[new_seq] += freq
            
        # 返回词表和合并规则
        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)
    
    def save_vocab_merges(self, vocab_filepath: str, merges_filepath: str) -> None:
        """
        将 Tokenizer 的 vocab 和 merges 保存到文件，便于以后用 from_files 加载。

        Args:
            vocab_filepath: 保存 vocab 的 JSON 文件路径
            merges_filepath: 保存 merges 的 TXT 文件路径
        """
        gpt2_encoder = gpt2_bytes_to_unicode()

        # === 保存 vocab.json ===
        vocab_to_save = {}
        for token_id, token_bytes in self.vocab.items():
            token_str = "".join(gpt2_encoder[b] for b in token_bytes)
            vocab_to_save[token_str] = token_id

        with open(vocab_filepath, "w", encoding="utf-8") as f:
            json.dump(vocab_to_save, f, ensure_ascii=False, indent=2)

        # === 保存 merges.txt ===
        with open(merges_filepath, "w", encoding="utf-8") as f:
            for a, b in self.merges:
                a_str = "".join(gpt2_encoder[byte] for byte in a)
                b_str = "".join(gpt2_encoder[byte] for byte in b)
                f.write(f"{a_str} {b_str}\n")
                
    def save_token_ids(self, token_ids_path, text_path):
        np.save(token_ids_path, np.array(self.encode(text_path)))
