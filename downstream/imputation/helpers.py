class EpochReport:
    def __init__(self):
        self.rows = {"train": [], "val": [], "test": []}

    def add(self, phase: str, epoch: int, **data):
        self.rows[phase].append({"phase": phase, "epoch": epoch, **data})

    def as_dict(self):
        return self.rows
