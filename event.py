class Event:
    def __init__(self):
        self._handlers = []

    def add(self, h):
        self._handlers.append(h)

    def remove(self, h):
        self._handlers.remove(h)

    def __call__(self, *args, **kwargs):
        for h in self._handlers:
            h(*args, **kwargs)
