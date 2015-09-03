class FeatureMap(object):

    def __init__(self):
        self.feature_map = {}
        self.rev_feature_map = {}
        self.max_i = 0

    def add_if_new(self, key):
        if not key in self.feature_map:
            self.feature_map[key] = self.max_i
            self.rev_feature_map[self.max_i] = key
            self.max_i += 1

    def __getitem__(self, key):
        self.add_if_new(key)
        return self.feature_map[key]

    def __setitem__(self, key, value):
        self.add_if_new(key)
        self.feature_map[key] = value

    instances = {}

    @staticmethod
    def get(name):
        if not name in FeatureMap.instances:
            FeatureMap.instances[name] = FeatureMap()
        return FeatureMap.instances[name]
