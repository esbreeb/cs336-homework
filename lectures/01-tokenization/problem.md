# 第一讲作业：概览与分词

对应 [CS336 Spring 2025](https://cs336.stanford.edu/spring2025/) Lecture 1（Overview, tokenization）。讲义代码是 [`lecture_01.py`](https://github.com/stanford-cs336/spring2025-lectures/blob/main/lecture_01.py)。

这一份停在讲座里的最小分词器。Assignment 1 另外要求 GPT-2 预分词、special token，以及 Transformer 的训练。

## 要交的两个文件

| 文件 | 做什么 |
| --- | --- |
| [`answer.md`](answer.md) | 简答模版。把表和空行填上。 |
| [`bpe.py`](bpe.py) | 留空的代码。只填标了 `NotImplementedError` 的函数。 |

不要改 `bpe.py` 的函数签名、`BPETokenizerParams` 的字段，以及文件末尾的 `_self_check`。可以自己加 import。不要调用 `tiktoken` 或 Hugging Face 的 tokenizer。

填完后在本目录运行：

```bash
python3 bpe.py
```

通过时打印 `ok`，退出码为 0。

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

## 第一部分：简答

写在 [`answer.md`](answer.md)。手算题先写结果，再用你自己的 `train_bpe` 对照。自测里没有这道题的答案。

1. `encode`、`decode` 和 vocabulary size 各指什么？
2. 对字符串 `Hello, 🌍! 你好!`，填 UTF-8 字节数、字符数、character tokenizer 的压缩率、byte tokenizer 的压缩率。再各用一句话写 character、byte、word 三种切分的主要问题。压缩率可以写成分数或小数。
3. 词表里的 UNK 会给解码和 perplexity 带来什么问题？
4. 对 `"ab ab ab"` 做 3 次 merge。每一步写出 pair（两个整数）、出现次数、新 token id，以及 merge 之后的整条序列。并列时用上面的规则。
5. 对 `"aaaa"`：pair `(97, 97)` 的计数是多少？调用一次 `merge(..., 256)` 之后序列是什么？为什么这两个数字不一样？
6. 为什么 `encode` 必须按训练顺序重放 `merges`？用第 4 题里第二次合并所依赖的 token id 说明，颠倒顺序之后哪些合并不会发生。
7. 相对 byte tokenizer，BPE 用更大的词表换来了什么？为什么讲座仍把分词叫作 necessary evil？

## 第二部分：代码

在 [`bpe.py`](bpe.py) 里实现：

- `get_compression_ratio`
- `CharacterTokenizer.encode` / `decode`
- `ByteTokenizer.encode` / `decode`
- `merge`
- `train_bpe`
- `BPETokenizer.encode` / `decode`

`CharacterTokenizer` 用 `ord` / `chr`。`ByteTokenizer` 用 UTF-8。BPE 的训练和编码遵守上一节的规则。`vocab` 始终包含 0 到 255 的单字节，再加上每个新 token。

自测会检查往返、空串、不重叠替换、提前停止、并列时的选择，以及 `"the cat in the hat"` 上 3 次合并的结果。第 4 题的 `"ab ab ab"` 不在自测里。
