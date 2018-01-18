from abc import ABC, abstractclassmethod

class BaseModel(ABC):
    def __init__(self, loader, columns):
        super().__init__()
        self.init_model(loader, columns)

    def init_model(self, loader, columns):
        self.loader = loader
        self.columns = columns
        self.model_pathname = loader.save_dir + "/" + loader.model_name
        self.model_savename = self.model_pathname + ".h5"


    @abstractclassmethod
    def build_model(self):
        pass


    @abstractclassmethod
    def fit(self, feature_data, label_data):
        pass


    @abstractclassmethod
    def validate(self, feature_data, label_data):
        pass


    @abstractclassmethod
    def predicate(self, feature_data, label_data):
        pass


    @abstractclassmethod
    def test(self, feature_data, label_data):
        pass


    @abstractclassmethod
    def summary(self):
        pass
