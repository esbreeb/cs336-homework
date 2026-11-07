# 第一讲作业：概览与分词

对应 [CS336 Spring 2025](https://cs336.stanford.edu/spring2025/) Lecture 1（Overview, tokenization）。讲义代码是 [`lecture_01.py`](https://github.com/stanford-cs336/spring2025-lectures/blob/main/lecture_01.py)。

这一份停在讲座里的最小分词器。Assignment 1 另外要求 GPT-2 预分词、special token，以及 Transformer 的训练。

## 三个文件

| 文件 | 做什么 |
| --- | --- |
| `problem.md` | 题目。 |
| [`answer.py`](answer.py) | 答案模版。完整可运行的实现，用来对照。 |
| [`bpe.py`](bpe.py) | 留空的代码。把标了 `NotImplementedError` 的函数填上。 |

`answer.py` 和 `bpe.py` 的接口一致。在编辑器里对这两个文件做 diff，差的就是要写的函数体。

不要改 `bpe.py` 的函数签名、`BPETokenizerParams` 的字段，以及文件末尾的 `_self_check`。可以自己加 import。不要调用 `tiktoken` 或 Hugging Face 的 tokenizer。

```bash
python3 answer.py   # 答案，应打印 ok
python3 bpe.py      # 留空文件填完后也应打印 ok
```

## 背景

语言模型给出的是 token 序列上的概率。tokenizer 负责两件事：`encode` 把字符串变成 token id，`decode` 把 id 变回字符串。vocabulary size 是可能出现的 token id 的个数。

压缩率用字节数衡量序列被缩短了多少：

```text
compression ratio = UTF-8 字节数 / token 数
```

token 序列为空时，压缩率定义为 0。

三种直接的切分都不好用。按 Unicode 字符切，词表大约有 15 万，很多字符几乎不出现。按字节切，词表只有 256，但压缩率恒为 1，序列长，而 attention 的计算量随长度二次增长。按词切，词的个数不固定，罕见词学不好，没见过的词只能落到一个 UNK 上。

BPE（Gage 1994；Sennrich 等 2016；GPT-2 沿用）在语料上学习词表：常见的字节片段合成一个 token，罕见片段仍由多个 token 拼成。本作业做讲座里的版本，整段字符串一起训练，不先做预分词。

训练规则：

1. 把字符串编成 UTF-8，每个字节是一个 0 到 255 的 token。`vocab[i] = bytes([i])`。
2. 统计当前序列里每一对相邻 token 的出现次数。重叠的相邻对也要计数。`aaaa` 里 pair `(97, 97)` 出现 3 次。
3. 取得分最高的 pair。并列时，选这一轮从左到右扫描时最先被记到的那个。按扫描顺序写入普通 dict，再 `max(counts, key=counts.get)`，得到的就是这个 pair。
4. 新 token id 从 256 开始，每成功合并一次加一。`vocab[new] = vocab[left] + vocab[right]`。
5. 用 `merge` 替换整条序列：从左到右，命中 pair 就换成新 id 并跳过这两个位置，不重叠、不回头。所以 `aaaa` 计了 3 次，替换之后却是两个新 token：`[256, 256]`。
6. 重复 2–5，直到做满 `num_merges`，或者序列里已经没有相邻 pair。后一种情况直接停止。

`encode` 把新字符串变成字节 id，再按 `merges` 的插入顺序把训练得到的合并逐条重放。`decode` 按 `vocab` 取出字节，拼起来，用 UTF-8 解码。

`answer.py` 里用断言记下这些结果：`Hello, 🌍! 你好!` 是 20 个字节、13 个字符，character 压缩率是 20/13，byte 压缩率是 1；`"ab ab ab"` 三次合并依次得到 `(97, 98) → 256`、`(256, 32) → 257`、`(257, 257) → 258`。第二次 `(256, 32)` 和 `(32, 256)` 次数相同，取从左到右先出现的那对。`"aaaa"` 里 `(97, 97)` 计 3 次，一次 `merge` 得到 `[256, 256]`。

## 要写的代码

在 [`bpe.py`](bpe.py) 里实现：

- `get_compression_ratio`
- `CharacterTokenizer.encode` / `decode`
- `ByteTokenizer.encode` / `decode`
- `merge`
- `train_bpe`
- `BPETokenizer.encode` / `decode`

`CharacterTokenizer` 用 `ord` / `chr`。`ByteTokenizer` 用 UTF-8。BPE 的训练和编码遵守上一节的规则。`vocab` 始终包含 0 到 255 的单字节，再加上每个新 token。

`bpe.py` 末尾的自测检查往返、空串、不重叠替换、提前停止、并列时的选择，以及 `"the cat in the hat"` 上 3 次合并。`"ab ab ab"` 的逐步结果在 `answer.py` 里。
