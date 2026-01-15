from collections import Counter


class Vocab:
    def __init__(self, answers, min_freq=1, specials=["<unk>"]):
        norm_answers = [ans.lower().strip() for ans in answers]
        counter = Counter(norm_answers)

        self.itos = specials.copy()
        for ans, freq in counter.items():
            if freq >= min_freq:
                self.itos.append(ans)
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def encode(self, ans):
        return self.stoi.get(ans.lower().strip(), self.stoi["<unk>"])

    def decode(self, idx):
        return self.itos[idx]

    def __len__(self):
        return len(self.itos)
