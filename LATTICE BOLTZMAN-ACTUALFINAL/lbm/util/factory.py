class Factory:
    """
    Generic factory method for creating objects at runtime.
    Class constructors are first registered and may then be created.
    """
    def __init__(self):
        self.keys = {}

    def register(self, key, constructor):
        self.keys[key] = constructor

    def create(self, key, **kwargs):
        constructor = self.keys.get(key)
        if not constructor:
            raise ValueError(key)
        return constructor(**kwargs)
