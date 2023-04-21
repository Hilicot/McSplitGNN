

class VertexPair:
    v: int
    w: int

    def __init__(self, v: int, w: int):
        self.v = v
        self.w = w

    def get(self):
        return self.v, self.w