import re, math
import itertools
from collections import Counter
from dataclasses import dataclass, field
from typing import Self
import struct
import queue


def sliding_window(sentence, n):
    for i in range(len(sentence) - n + 1):
        yield sentence[i : i + n]


@dataclass
class VocabTrie:
    char: str | None = None
    parent: Self | None = None

    id: int | None = None  # int for words (not necessarily leaf node)
    children: dict[str, Self] = field(default_factory=dict)

    def __hash__(self):
        return id(self)

    def __len__(self):
        is_word = 1 if self.id is not None else 0
        return is_word + sum([len(c) for c in self.children.values()])

    def __contains__(self, key):
        return self.get(key) is not None

    def get(self, word: str, default=None):
        if word is None:
            return default

        if word == "":
            return self.id if self.id is not None else default

        c = word[0]
        key = word[1:]

        child = self.children.get(c)
        if child is None:
            return default

        return child.get(key, default)

    def then(self, char: str):
        if char in self.children:
            return self.children[char]

        self.children[char] = VocabTrie(char, self)
        return self.children[char]


@dataclass
class TrieNode:
    word: str | None = None
    parent: Self | None = None

    backoff_weight: float = 0.0
    prob: float = 0.0
    count = 1
    continues: set = field(default_factory=set)
    children: dict[str, Self] = field(default_factory=dict)

    def __hash__(self):
        return id(self)

    def get(self, word: str, default=None):
        return self.children.get(word, default)

    # Used for building the model in a functional style
    def then(self, word: str, increment=True):
        if word in self.children:
            if increment:
                self.children[word].count += 1
            return self.children[word]

        self.children[word] = TrieNode(word, self)
        return self.children[word]

    # Used for calculating P_continues(word)
    def follows(self, word: str):
        self.continues.update({word})


class KenLMModel:
    # n=1 -> unigram
    # n=2 -> bigram
    # n=3 -> trigram etc.
    def __init__(self, n=3, unk_threshold=1):
        self.n = n
        self.unk_threshold = unk_threshold
        self.vocab = VocabTrie()
        self.model = TrieNode()

    def train(self, filename, length, count_thresholds=None):
        # Pass 1: count raw words
        raw_counts = Counter()
        with open(filename) as f:
            for i, line in itertools.islice(enumerate(f), length):
                line = line.lower()
                words = re.sub(r"\[.*?\]", "", line).split()
                raw_counts.update(words)

        # Build vocabulary and prune rare words
        vocab = {w for w, c in raw_counts.items() if c > self.unk_threshold}
        vocab.update({"<s>", "</s>", "<unk>"})

        # Build vocab trie
        for i, w in enumerate(vocab):
            v = self.vocab
            for c in w:
                v = v.then(c)
            v.id = i

        with open(filename) as f:
            for _, line in itertools.islice(enumerate(f), length):
                line = line.lower()

                # Remove [text] from subitles used for direction and split into array
                words = re.sub(r"\[.*?\]", "", line).split()
                if not len(words) or words[-1] == ":":
                    continue

                # Replace rare words with <unk>
                words = [w if w in self.vocab else "<unk>" for w in words]
                # Prepend <s> and append </s>
                words = ["<s>"] * max(self.n - 1, 1) + words + ["</s>"]

                for seq in sliding_window(words, self.n):
                    node = self.model
                    for h in seq:
                        node = node.then(h)

                    self.model.then(seq[-1], False).follows(seq[-2])

        # Prune based on count
        if count_thresholds:
            nodes = queue.Queue()
            nodes.put((self.model, 0))

            # Breadth-first traversal
            while not nodes.empty():
                node, depth = nodes.get()
                if node.word:
                    vocab.update(node.word)

                node.children = {
                    w: n
                    for w, n in node.children.items()
                    if n.count >= count_thresholds[depth]
                }
                for n in node.children.values():
                    nodes.put((n, depth + 1))

        # Calculate P_continuation(w) for the unigram
        total_bigrams = sum(len(w.continues) for w in self.model.children.values())
        for w in self.model.children.values():
            w.prob = len(w.continues) / total_bigrams

        def kneyser_ney_smoothing(node):
            total_tokens = sum(c.count for c in node.children.values())
            discounts = []
            for c in node.children.values():
                if c.count == 1:
                    discount = 0.75
                elif c.count == 2:
                    discount = 1.0
                else:
                    discount = 1.25

                discounts.append(discount)
                c.prob = max(c.count - discount, 0) / total_tokens

            node.backoff_weight = (
                sum(discounts) / total_tokens if len(node.children) else 0.0
            )

            if len(node.children):
                for w in node.children.values():
                    kneyser_ney_smoothing(w)

        for w in self.model.children.values():
            kneyser_ney_smoothing(w)

    def score(self, seq):
        w = seq[-1]
        if len(seq) == 1:
            return self.model.get(w, self.model.get("<unk>")).prob

        backoff_weight = 0.0
        node = self.model
        for h in seq:
            c = node.get(h)
            if c is None:
                prob = 0.0
                break

            prob = c.prob
            backoff_weight = node.backoff_weight  # Track the parent's backoff_weight
            node = c

        if prob == 0:
            backoff_weight = 1.0

        backoff_seq = seq[1:]
        return prob + backoff_weight * self.score(backoff_seq)

    def perplexity(self, lines: list[str]):
        log_probabilities = []
        for line in lines:
            line = line.lower()
            words = re.sub(r"\[.*?\]", "", line).split()
            if not len(words) or words[-1] == ":":
                continue

            # Replace any unseen words with <unk>
            words = [w if w in self.vocab else "<unk>" for w in words]
            words = ["<s>"] * max(self.n - 1, 1) + words + ["</s>"]

            for seq in sliding_window(words, self.n):
                prob = math.log(self.score(seq))
                log_probabilities.append(prob)

        return math.exp(-sum(log_probabilities) / len(log_probabilities))

    # Quantization saves ~15% space, seems less worthwhile for already small bin files
    def save(self, filename, quantized=False):
        # Header: node count, pointer count.
        # Same values for now, but keeping separate in case of later compression/decoupling
        HEADER_STRUCT = struct.Struct("<ii")
        # Node: prob, backoff, child_start, child_count
        NODE_STRUCT = (
            struct.Struct("<ffii") if not quantized else struct.Struct("<eeii")
        )
        # Pointer: word_id, node_index
        POINTER_STRUCT = struct.Struct("<ii")

        nodes = []
        pointers = {}

        node_queue = queue.Queue()

        # Breadth-first flattening of the trie
        node_queue.put(self.model)

        to_id = lambda n: self.vocab.get(n.word, -1) if n is not None else -1

        while not node_queue.empty():
            node = node_queue.get()
            pointers[node] = (to_id(node), len(nodes))

            # Skip for root node
            if to_id(node) != -1:
                parent_pointer = pointers[node.parent][1]
                if nodes[parent_pointer][2] == -1:
                    nodes[parent_pointer][2] = len(nodes)

            nodes.append([node.prob, node.backoff_weight, -1, len(node.children)])

            for _, c in sorted(
                node.children.items(), key=lambda c: self.vocab.get(c[0])
            ):
                node_queue.put(c)

        pointers_flat = [tup for tup in pointers.values()]

        with open(filename, "wb") as f:
            f.write(HEADER_STRUCT.pack(len(nodes), len(pointers)))
            for n in nodes:
                f.write(NODE_STRUCT.pack(*n))
            for p in pointers_flat:
                f.write(POINTER_STRUCT.pack(*p))

        # Do the same for the vocab trie

        # Header: node count, character count
        HEADER_STRUCT = struct.Struct("<ii")
        # Node: char_index (ptr to char pool), child start, child count, word_id (-1 if not a word)
        NODE_STRUCT = struct.Struct("<iiii")

        chars = set({})
        nodes = []
        pointers = {}

        node_queue = queue.Queue()
        node_queue.put(self.vocab)
        while not node_queue.empty():
            node = node_queue.get()
            if node.char is not None:
                chars.add(node.char)
            pointers[node] = len(nodes)

            nodes.append(
                [
                    node.char,
                    -1,
                    len(node.children),
                    node.id if node.id is not None else -1,
                ]
            )

            if node.parent is not None:
                parent_pointer = pointers[node.parent]
                if nodes[parent_pointer][1] == -1:
                    nodes[parent_pointer][1] = len(nodes)

            for c in node.children.values():
                node_queue.put(c)

        chars = {c: i for i, c in enumerate(sorted(chars))}
        for n in nodes:
            n[0] = chars[n[0]] if n[0] is not None else 0

        with open(f"{filename}.vocab", "wb") as f:
            f.write(HEADER_STRUCT.pack(len(nodes), len(chars)))
            for n in nodes:
                f.write(NODE_STRUCT.pack(*n))
            for c in chars.keys():
                # Null-terminating characters
                f.write(c.encode("utf-8") + b"\0")


def main():
    train_len = 3000000
    test_len = 100000

    model = KenLMModel(n=5, unk_threshold=2)
    model.train(filename="en.tok", length=train_len, count_thresholds=[1, 2, 5, 10, 20])

    val_set = []
    with open("en.tok") as f:
        for line in itertools.islice(f, train_len, train_len + test_len):
            val_set.append(line)

    score = model.perplexity(val_set)
    print(f"Vocab: {len(model.vocab)}")
    print(f"Score: {score}")

    model.save("model.bin")


if __name__ == "__main__":
    main()
