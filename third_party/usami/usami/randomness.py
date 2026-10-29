"""Versioned SHA-256 counter sampling, independent of Python's random implementation."""
import hashlib


class StableRandom:
    def __init__(self, seed: int):
        self.seed = str(seed)
        self.counter = 0

    def randbelow(self, limit: int) -> int:
        if limit < 1 or limit > 1 << 256:
            raise ValueError("sampling limit must be in [1, 2**256]")
        ceiling = (1 << 256) - ((1 << 256) % limit)
        while True:
            block = hashlib.sha256(f"usami-rng-v1:{self.seed}:{self.counter}".encode()).digest()
            self.counter += 1
            value = int.from_bytes(block, "big")
            if value < ceiling:
                return value % limit

    def randint(self, low: int, high: int) -> int:
        return low + self.randbelow(high - low + 1)

    def choice(self, values):
        return values[self.randbelow(len(values))]

    def random(self) -> float:
        return self.randbelow(1 << 53) / (1 << 53)

    def sample(self, values, count: int):
        pool = list(values)
        if count < 0 or count > len(pool):
            raise ValueError("invalid sample size")
        result = []
        for _ in range(count):
            result.append(pool.pop(self.randbelow(len(pool))))
        return result

    def shuffle(self, values) -> None:
        for i in range(len(values) - 1, 0, -1):
            j = self.randbelow(i + 1)
            values[i], values[j] = values[j], values[i]
