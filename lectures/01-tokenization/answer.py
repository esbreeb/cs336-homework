"""第一讲作业的答案模版：与 bpe.py 同一套接口，函数体已填好。

在编辑器里对 answer.py 和 bpe.py 做 diff，就能看到要写进留空文件的实现。
运行：python3 answer.py
通过时打印 ok。
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass


def get_compression_ratio(string: str, indices: list[int]) -> float:
    """UTF-8 字节数 / token 数。indices 为空时返回 0.0。"""
    if not indices:
        return 0.0
    return len(string.encode("utf-8")) / len(indices)


class Tokenizer:
    def encode(self, string: str) -> list[int]:
        raise NotImplementedError

    def decode(self, indices: list[int]) -> str:
        raise NotImplementedError


class CharacterTokenizer(Tokenizer):
    """一个 Unicode 字符对应一个 token，token id 是 ord(char)。"""

    def encode(self, string: str) -> list[int]:
        return [ord(char) for char in string]

    def decode(self, indices: list[int]) -> str:
        return "".join(chr(index) for index in indices)


class ByteTokenizer(Tokenizer):
    """一个 UTF-8 字节对应一个 token，token id 落在 0..255。"""

    def encode(self, string: str) -> list[int]:
        return list(string.encode("utf-8"))

    def decode(self, indices: list[int]) -> str:
        return bytes(indices).decode("utf-8")


def merge(indices: list[int], pair: tuple[int, int], new_index: int) -> list[int]:
    """从左到右、不重叠地把 pair 换成 new_index。

    命中后跳过这两个位置，不回头。pair 不出现时，返回内容相同的新列表。
    例：merge([97, 97, 97, 97], (97, 97), 256) == [256, 256]。
    """
    merged: list[int] = []
    index = 0
    while index < len(indices):
        if (
            index + 1 < len(indices)
            and indices[index] == pair[0]
            and indices[index + 1] == pair[1]
        ):
            merged.append(new_index)
            index += 2
        else:
            merged.append(indices[index])
            index += 1
    return merged


@dataclass(frozen=True)
class BPETokenizerParams:
    """训练结果。vocab 的 key 是 token id，value 是该 token 的原始字节。

    merges 的插入顺序就是训练顺序：先插入的 merge 先被 encode 重放。
    """

    vocab: dict[int, bytes]
    merges: dict[tuple[int, int], int]


def train_bpe(string: str, num_merges: int) -> BPETokenizerParams:
    """在单条字符串上训练 BPE，最多做 num_merges 次合并。

    - 初始 token 是 UTF-8 字节。vocab[0..255] 固定为单字节。
    - 每一轮统计当前序列里全部相邻 pair 的出现次数（重叠的也计数）。
    - 次数最高的 pair 若有多个，选本轮从左到右最先被记到的那个。
    - 新 token id 从 256 起递增。vocab[new] 是左右两个 token 的字节拼接。
    - 用 merge 替换整条序列，再进入下一轮。
    - 序列里已经没有相邻 pair 时提前停止。
    """
    indices = list(string.encode("utf-8"))
    merges: dict[tuple[int, int], int] = {}
    vocab: dict[int, bytes] = {token_id: bytes([token_id]) for token_id in range(256)}
    for _ in range(num_merges):
        counts: dict[tuple[int, int], int] = defaultdict(int)
        for left, right in zip(indices, indices[1:]):
            counts[(left, right)] += 1
        if not counts:
            break
        pair = max(counts, key=counts.get)
        new_index = 256 + len(merges)
        merges[pair] = new_index
        vocab[new_index] = vocab[pair[0]] + vocab[pair[1]]
        indices = merge(indices, pair, new_index)
    return BPETokenizerParams(vocab=vocab, merges=merges)


class BPETokenizer(Tokenizer):
    def __init__(self, params: BPETokenizerParams):
        self.params = params

    def encode(self, string: str) -> list[int]:
        """先变成字节 id，再按 merges 的插入顺序逐条重放。"""
        indices = list(string.encode("utf-8"))
        for pair, new_index in self.params.merges.items():
            indices = merge(indices, pair, new_index)
        return indices

    def decode(self, indices: list[int]) -> str:
        """把每个 token id 换成 vocab 里的字节，拼起来再按 UTF-8 解码。"""
        return b"".join(self.params.vocab[index] for index in indices).decode("utf-8")


def _self_check() -> None:
    hello = "Hello, 🌍! 你好!"

    characters = CharacterTokenizer()
    assert characters.encode(hello) == [ord(char) for char in hello]
    assert characters.decode(characters.encode(hello)) == hello
    assert characters.encode("") == []
    assert characters.decode([]) == ""
    assert len(hello.encode("utf-8")) == 20
    assert len(hello) == 13
    assert get_compression_ratio(hello, characters.encode(hello)) == 20 / 13

    byte_tokenizer = ByteTokenizer()
    assert byte_tokenizer.encode(hello) == list(hello.encode("utf-8"))
    assert byte_tokenizer.decode(byte_tokenizer.encode(hello)) == hello
    assert get_compression_ratio(hello, byte_tokenizer.encode(hello)) == 1.0
    assert get_compression_ratio("", []) == 0.0

    assert merge([97, 97, 97, 97], (97, 97), 256) == [256, 256]
    assert merge([97, 97, 97], (97, 97), 256) == [256, 97]
    assert merge([1, 2, 1, 2], (1, 2), 9) == [9, 9]
    assert merge([1, 2, 3], (9, 9), 7) == [1, 2, 3]

    empty = train_bpe("ab ab ab", 0)
    assert empty.merges == {}
    assert len(empty.vocab) == 256
    assert empty.vocab[0] == bytes([0])
    assert empty.vocab[255] == bytes([255])

    single = train_bpe("a", 5)
    assert single.merges == {}
    assert BPETokenizer(single).encode("a") == [97]
    assert BPETokenizer(single).decode([97]) == "a"
    blank = train_bpe("", 4)
    assert blank.merges == {}
    assert BPETokenizer(blank).encode("") == []
    assert BPETokenizer(blank).decode([]) == ""

    one_pair = train_bpe("ab", 1)
    assert list(one_pair.merges.items()) == [((97, 98), 256)]
    assert one_pair.vocab[256] == b"ab"
    assert len(one_pair.vocab) == 257

    # "ab ab ab"：第二次 (256, 32) 与 (32, 256) 次数相同，取从左到右先出现的 (256, 32)。
    spaced = train_bpe("ab ab ab", 3)
    assert list(spaced.merges.items()) == [
        ((97, 98), 256),
        ((256, 32), 257),
        ((257, 257), 258),
    ]
    assert spaced.vocab[256] == b"ab"
    assert spaced.vocab[257] == b"ab "
    assert spaced.vocab[258] == b"ab ab "
    assert BPETokenizer(spaced).encode("ab ab ab") == [258, 256]

    # 并列时选从左到右最先出现的 pair：l-o 与 o-w 都出现 4 次。
    low = train_bpe("low low low low", 1)
    assert list(low.merges) == [(108, 111)]
    assert low.vocab[256] == b"lo"

    repeated = train_bpe("aaaa", 2)
    assert list(repeated.merges.items()) == [((97, 97), 256), ((256, 256), 257)]
    assert repeated.vocab[256] == b"aa"
    assert repeated.vocab[257] == b"aaaa"
    assert BPETokenizer(repeated).encode("aaaa") == [257]
    assert BPETokenizer(repeated).decode([257]) == "aaaa"

    params = train_bpe("the cat in the hat", 3)
    assert list(params.merges.items()) == [
        ((116, 104), 256),
        ((256, 101), 257),
        ((257, 32), 258),
    ]
    assert params.vocab[256] == b"th"
    assert params.vocab[257] == b"the"
    assert params.vocab[258] == b"the "
    tokenizer = BPETokenizer(params)
    assert tokenizer.encode("the cat in the hat") == [
        258, 99, 97, 116, 32, 105, 110, 32, 258, 104, 97, 116,
    ]
    assert tokenizer.decode(tokenizer.encode("the cat in the hat")) == "the cat in the hat"
    assert tokenizer.encode("that") == [256, 97, 116]
    assert tokenizer.encode("the quick brown fox") == [
        258, 113, 117, 105, 99, 107, 32, 98, 114, 111, 119, 110, 32, 102, 111, 120,
    ]
    assert tokenizer.decode(tokenizer.encode("the quick brown fox")) == "the quick brown fox"

    trained = train_bpe(hello, 6)
    encoded = BPETokenizer(trained).encode(hello)
    assert BPETokenizer(trained).decode(encoded) == hello
    assert len(encoded) < len(hello.encode("utf-8"))

    print("ok")


if __name__ == "__main__":
    _self_check()
