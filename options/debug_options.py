from .base_options import BaseOptions
import copy

"""Debug options for the project."""


class DebugOptionsObject:
    def __init__(self, opt):
        # copy all the options from the base options
        for k, v in opt.__dict__.items():
            self.__dict__[k] = copy.deepcopy(v)

        # set the new/overrided options
        self.log_level = 'debug'
        self.log_file = None
        self.log_stdout = True
        self.quiet = True

        self.phase = 'train'
        self.device = 'cpu'
        self.dataset_format = "ascii"
        self.timeout = 10
        self.max_iter = 0
        self.plot_graphs_on_loading = False  # Plot graphs when they are loaded into the model. (For debugging purposes)


class DebugOptions(BaseOptions):
    def __init__(self):
        super().__init__()

    def initialize(self):
        super().initialize()

    def parse(self):
        super().parse()
        return DebugOptionsObject(self.opt)
