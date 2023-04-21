from options.base_options import BaseOptions,null_coalescence

class TestOptions(BaseOptions):
    def initialize(self):
        super().initialize()

    def parse(self):
        super().parse()
        
        self.opt.phase = 'test'

        return self.opt
