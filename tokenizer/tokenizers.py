import argparse
import os
import re
import unicodedata
import random
import pickle
import json
import heapq
import time
from pathlib import Path
from collections import defaultdict, Counter
from tqdm import tqdm 

random.seed(69)
BPE_MERGES = 10000 # Default


class Tokenizer:
    def train(self, data):
        pass
    
    def tokenize(self, text):
        raise NotImplementedError
    
    def save(self, path):
        pass
    
    def load(self, path):
        pass


class WhitespaceTokenizer(Tokenizer):
    def tokenize(self, text):
        return [t for t in re.split(r'(\s+)', text) if t and not t.isspace()]


class RegexTokenizer(Tokenizer):
    def __init__(self):
        self.pattern = re.compile(r"[\w\u0900-\u097F]+|[^\w\s]", re.UNICODE)
    
    def tokenize(self, text):
        return self.pattern.findall(text)


class customBPE(Tokenizer):
    def __init__(self, num_merges=BPE_MERGES, min_pair_frequency=2):
        self.num_merges = num_merges
        self.min_pair_frequency = min_pair_frequency
        self.merges = {}  # (left, right) -> merged string
        self.vocab = {}
        self._token_cache = {}
        self.pre_tokenizer = RegexTokenizer()

    def _build_word_frequencies(self, data):
        token_frequencies = Counter()
        for line in tqdm(data, desc="Pre-tokenizing words"):
            words = self.pre_tokenizer.tokenize(line)
            for word in words:
                token_frequencies[tuple(list(word) + ["</w>"])] += 1
        return token_frequencies

    @staticmethod
    def _get_pairs_with_positions(sequence):
        """Returns dict of pair -> list of index positions in sequence"""
        pairs = defaultdict(list)
        for i in range(len(sequence) - 1):
            pairs[(sequence[i], sequence[i + 1])].append(i)
        return pairs

    @staticmethod
    def _merge_sequence_fast(sequence, pair):
        left, right = pair
        merged = left + right
        new_seq = []
        i = 0
        n = len(sequence)
        while i < n:
            if i < n - 1 and sequence[i] == left and sequence[i + 1] == right:
                new_seq.append(merged)
                i += 2
            else:
                new_seq.append(sequence[i])
                i += 1
        return tuple(new_seq)

    def _build_vocab(self, sequences):
        tokens = {"특수": 0}
        for sequence in sequences:
            for token in sequence:
                if token == "</w>":
                    continue
                clean_token = token.replace("</w>", "")
                if clean_token and clean_token not in tokens:
                    tokens[clean_token] = len(tokens)
        return tokens

    def train(self, data):
        print("Training Optimized BPE Tokenizer...")
        t0 = time.time()
        word_frequencies = self._build_word_frequencies(data)

        # Sequences and their occurrence frequencies
        sequences = list(word_frequencies.keys())
        frequencies = [word_frequencies[sequence] for sequence in sequences]
        print(f"Unique words / initial sequence types: {len(sequences):,}")

        # Index pair counts and inverted index (pair -> set of word_ids)
        pair_counts = Counter()
        pair_to_words = defaultdict(set)

        for word_id, sequence in enumerate(sequences):
            freq = frequencies[word_id]
            for i in range(len(sequence) - 1):
                p = (sequence[i], sequence[i + 1])
                pair_counts[p] += freq
                pair_to_words[p].add(word_id)

        # Max-heap for constant time O(1) top pair lookup
        heap = [(-count, pair) for pair, count in pair_counts.items() if count >= self.min_pair_frequency]
        heapq.heapify(heap)

        print(f"Starting up to {self.num_merges:,} merges...")
        progress = tqdm(total=self.num_merges, desc="Merging pairs")

        while len(self.merges) < self.num_merges and heap:
            neg_count, best_pair = heapq.heappop(heap)
            count = -neg_count
            current_count = pair_counts.get(best_pair, 0)

            # Lazy heap invalidation check
            if count != current_count:
                continue
            if current_count < self.min_pair_frequency:
                break

            self.merges[best_pair] = best_pair[0] + best_pair[1]
            affected_word_ids = pair_to_words.pop(best_pair, set())
            del pair_counts[best_pair]

            left, right = best_pair

            for word_id in affected_word_ids:
                old_sequence = sequences[word_id]
                freq = frequencies[word_id]

                # Fast check if pair still exists in word
                if left not in old_sequence:
                    continue

                # Remove old pairs
                for i in range(len(old_sequence) - 1):
                    p = (old_sequence[i], old_sequence[i + 1])
                    pair_counts[p] -= freq
                    if pair_counts[p] <= 0:
                        del pair_counts[p]
                        pair_to_words[p].discard(word_id)

                # Merge sequence
                new_sequence = self._merge_sequence_fast(old_sequence, best_pair)
                sequences[word_id] = new_sequence

                # Add new pairs
                for i in range(len(new_sequence) - 1):
                    p = (new_sequence[i], new_sequence[i + 1])
                    pair_counts[p] += freq
                    pair_to_words[p].add(word_id)
                    new_c = pair_counts[p]
                    if new_c >= self.min_pair_frequency:
                        heapq.heappush(heap, (-new_c, p))

            progress.update(1)

        progress.close()
        self.merge_ranks = {pair: idx for idx, pair in enumerate(self.merges.keys())}
        self.vocab = self._build_vocab(sequences)
        self._token_cache.clear()
        print(f"BPE Training Complete in {time.time() - t0:.2f}s! Learned {len(self.merges):,} merges, Vocab: {len(self.vocab):,} tokens.")

    def tokenize(self, text):
        words = self.pre_tokenizer.tokenize(text)
        result = []
        
        if not hasattr(self, 'merge_ranks'):
            self.merge_ranks = {pair: idx for idx, pair in enumerate(self.merges.keys())}
        
        for word in words:
            cached = self._token_cache.get(word)
            if cached is not None:
                result.extend(cached)
                continue

            tokens = list(word) + ["</w>"]
            
            while len(tokens) > 1:
                pairs = [(tokens[i], tokens[i+1]) for i in range(len(tokens) - 1)]
                best_pair = None
                earliest_rank = float('inf')
                
                for pair in pairs:
                    rank = self.merge_ranks.get(pair)
                    if rank is not None and rank < earliest_rank:
                        earliest_rank = rank
                        best_pair = pair
                
                if best_pair is None:
                    break
                
                left, right = best_pair
                merged_tokens = []
                i = 0
                n = len(tokens)
                while i < n:
                    if i < n - 1 and tokens[i] == left and tokens[i+1] == right:
                        merged_tokens.append(left + right)
                        i += 2
                    else:
                        merged_tokens.append(tokens[i])
                        i += 1
                
                tokens = merged_tokens
            
            cleaned_tokens = [t.replace('</w>', '') for t in tokens if t != '</w>']
            cleaned_tokens = [token for token in cleaned_tokens if token]
            self._token_cache[word] = cleaned_tokens
            result.extend(cleaned_tokens)
        
        return result

    def save(self, path):
        model = {
            "num_merges": self.num_merges,
            "min_pair_frequency": self.min_pair_frequency,
            "merges": self.merges,
            "vocab": self.vocab,
        }
        with open(path, 'wb') as f:
            pickle.dump(model, f)

    def load(self, path):
        with open(path, 'rb') as f:
            model = pickle.load(f)
        if isinstance(model, dict) and "merges" in model:
            self.num_merges = model.get("num_merges", self.num_merges)
            self.min_pair_frequency = model.get("min_pair_frequency", self.min_pair_frequency)
            self.merges = model["merges"]
            self.vocab = model.get("vocab", {})
        else:
            self.merges = model
            self.vocab = {}
        self.merge_ranks = {pair: idx for idx, pair in enumerate(self.merges.keys())}
        self._token_cache.clear()

    def save_artifacts(self, output_dir, name):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        model_path = output_dir / f"{name}.pkl"
        vocab_path = output_dir / f"{name}_vocab.json"
        merges_path = output_dir / f"{name}_merges.txt"
        config_path = output_dir / f"{name}_config.json"

        self.save(model_path)

        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)

        with open(merges_path, "w", encoding="utf-8") as f:
            for left, right in self.merges:
                f.write(f"{left} {right}\n")

        config = {
            "type": "custom_bpe",
            "num_merges": self.num_merges,
            "learned_merges": len(self.merges),
            "min_pair_frequency": self.min_pair_frequency,
            "vocab_size": len(self.vocab),
            "model_file": model_path.name,
            "vocab_file": vocab_path.name,
            "merges_file": merges_path.name,
        }
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

        return {
            "model": model_path,
            "vocab": vocab_path,
            "merges": merges_path,
            "config": config_path,
        }


class BPETokenizer(customBPE):
    """BPE tokenizer with save_pretrained and HF Hub compatibility."""
    
    def save_pretrained(self, save_directory):
        """Save tokenizer model and vocabulary to directory."""
        output_dir = Path(save_directory)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model using existing method
        self.save(output_dir / f"bpe_{self.num_merges}.pkl")
        
        # Save vocab as JSON
        vocab_path = output_dir / f"bpe_{self.num_merges}_vocab.json"
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)
        
        # Save merges as txt
        merges_path = output_dir / f"bpe_{self.num_merges}_merges.txt"
        with open(merges_path, "w", encoding="utf-8") as f:
            for left, right in self.merges:
                f.write(f"{left} {right}\n")
    
    @classmethod
    def from_pretrained(cls, load_directory, name="bpe"):
        """Load tokenizer from directory."""
        load_dir = Path(load_directory)
        
        # Try to load the pkl file
        pkl_path = None
        for f in load_dir.iterdir():
            if f.name.endswith('.pkl') and f.name.startswith(f"{name}_"):
                pkl_path = f
                break
        
        if pkl_path is None:
            raise FileNotFoundError(f"No tokenizer pkl found in {load_directory}")
        
        tokenizer = cls()
        tokenizer.load(str(pkl_path))
        return tokenizer