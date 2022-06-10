
class Cycle(object):
    def __init__(self):
        self.forward = []
        self.backward = []
        self.num_other_frames = -1

    def __str__(self):
        s = "Forward:"
        for i, step in enumerate(self.forward):
            s += " {} -> {}|".format(step["src"], step["tgt"])

        s += "\nBackward:"
        for i, step in enumerate(self.backward):
            s += " {} -> {}|".format(step["src"], step["tgt"])

        return s

    def __repr__(self):
        return "Cycle:\n{}".format(str(self))

    def __copy__(self):
        return self.copy()

    @property
    def num_forward_steps(self):
        return len(self.forward)

    @property
    def num_backward_steps(self):
        return len(self.backward)

    def copy(self):
        copy = self.__class__()
        copy.forward = self.forward.copy()
        copy.backward = self.backward.copy()
        copy.num_other_frames = self.num_other_frames
        return copy

    def steps(self):
        for step in self.forward_steps():
            step = list(step) + [True]
            yield tuple(step)

        for step in self.backward_steps():
            step = list(step) + ["", False]
            yield tuple(step)

    def forward_steps(self):
        trace = ""
        for step in self.forward:
            trace += "{}->{};".format(step["src"], step["tgt"])
            yield step["src"], step["tgt"], trace

    def backward_steps(self):
        for step in self.backward:
            yield step["src"], step["tgt"]

    @classmethod
    def create(cls, forward, backward, num_other_frames):
        instance = cls()
        instance.forward = forward
        instance.backward = backward
        instance.num_other_frames = num_other_frames
        return instance

    @classmethod
    def from_string_spec(cls, num_other_frames: int, forward: str, backward: str):
        instance = cls()
        assert num_other_frames >= 1

        def parse_steps(s):
            s = s.replace(" ", "")
            assert ";" in s
            step_tokens = [s for s in s.split(";") if len(s) > 0]

            steps = []
            for token in step_tokens:
                # grad = not token.endswith("'")
                token = token.replace("'", "")

                try:
                    src, tgt = token.split("->")
                except ValueError as err:
                    print("Error while parsing token: '{}'".format(token))
                    raise err

                assert src[0] == "[" and src[-1] == "]", "Invalid src spec: \"{}\". Tgt: \"{}\". Token: \"{}\"".format(src, tgt, token)
                src_t = [int(t) for t in src[1:-1].split(",")]

                if "[" in tgt or "]" in tgt:
                    raise ValueError("Square brackets must not be present in target frame spec.")

                steps.append({"src": src_t, "tgt": int(tgt)})

            return steps

        instance.forward = parse_steps(forward)
        instance.backward = parse_steps(backward)
        instance.num_other_frames = num_other_frames

        return instance


def _test():
    cycle = Cycle.from_string_spec(
        num_other_frames=6,
        forward="[0]->1'; [0,1]->2'; [0,1,2]->3; [1,2,3]->4'; [2,3,4]->5'; [3,4,5]->6;",
        backward="[6]->5'; [5,6]->4'; [4,5,6]->3; [3,4,5]->2'; [2,3,4]->1'; [1,2,3]->0;"
    )

    print("CYCLE:")
    print(cycle)


if __name__ == '__main__':
    _test()
