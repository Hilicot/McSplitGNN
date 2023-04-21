class Bidomain:
    def __init__(self, l:int, r:int, left_len:int, right_len:int, is_adjacent:bool):
        self.l = l
        self.r = r
        self.left_len = left_len
        self.right_len = right_len
        self.is_adjacent = is_adjacent

    def get_max_len(self):
        return max(self.left_len, self.right_len)