"""第一讲作业的留空代码。题目在 problem.md，完整实现在 answer.py。

只实现标了 NotImplementedError 的函数，不要改类的字段、函数签名和文末自测。
行为与 CS336 Lecture 1 的最小 BPE 一致：从 UTF-8 字节出发，反复合并最高频相邻 pair。
不做 GPT-2 预分词，不处理 special token，也不要求速度。
"""

from __future__ import annotations

from dataclasses import dataclass


def get_compression_ratio(string: str, indices: list[int]) -> float:
    """UTF-8 字节数 / token 数。indices 为空时返回 0.0。"""
    raise NotImplementedError


class Tokenizer:
    def encode(self, string: str) -> list[int]:
        raise NotImplementedError

    def decode(self, indices: list[int]) -> str:
        raise NotImplementedError


class CharacterTokenizer(Tokenizer):
    """一个 Unicode 字符对应一个 token，token id 是 ord(char)。"""

    def encode(self, string: str) -> list[int]:
        raise NotImplementedError

    def decode(self, indices: list[int]) -> str:
        raise NotImplementedError


class ByteTokenizer(Tokenizer):
    """一个 UTF-8 字节对应一个 token，token id 落在 0..255。"""

    def encode(self, string: str) -> list[int]:
        raise NotImplementedError

    def decode(self, indices: list[int]) -> str:
        raise NotImplementedError


def merge(indices: list[int], pair: tuple[int, int], new_index: int) -> list[int]:
    """从左到右、不重叠地把 pair 换成 new_index。

    命中后跳过这两个位置，不回头。pair 不出现时，返回内容相同的新列表。
    例：merge([97, 97, 97, 97], (97, 97), 256) == [256, 256]。
    """
    raise NotImplementedError


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
    - 每一轮统计当前序列里全部相邻 pair 的出现次数（重叠的也计数）。可以用 collections.defaultdict。
    - 次数最高的 pair 若有多个，选本轮从左到右最先被记到的那个。
    - 新 token id 从 256 起递增。vocab[new] 是左右两个 token 的字节拼接。
    - 用 merge 替换整条序列，再进入下一轮。
    - 序列里已经没有相邻 pair 时提前停止，不要报错。
    """
    raise NotImplementedError


class BPETokenizer(Tokenizer):
    def __init__(self, params: BPETokenizerParams):
        self.params = params

    def encode(self, string: str) -> list[int]:
        """先变成字节 id，再按 merges 的插入顺序逐条重放。"""
        raise NotImplementedError

    def decode(self, indices: list[int]) -> str:
        """把每个 token id 换成 vocab 里的字节，拼起来再按 UTF-8 解码。"""
        raise NotImplementedError


def _self_check() -> None:
    hello = "Hello, 🌍! 你好!"

    characters = CharacterTokenizer()
    assert characters.encode(hello) == [ord(char) for char in hello]
    assert characters.decode(characters.encode(hello)) == hello
    assert characters.encode("") == []
    assert characters.decode([]) == ""
    assert get_compression_ratio(hello, characters.encode(hello)) == len(hello.encode("utf-8")) / len(hello)

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
