from options.base_options import BaseOptions, null_coalescence

class TrainOptions(BaseOptions):
    def initialize(self):
        super().initialize()
        self.parser.add_argument('--promise_mode', default='diverse')

    def parse(self):
        super().parse()
        
        self.opt.phase = 'train'
        self.opt.scalable = null_coalescence(self.opt.scalable, True)
        self.opt.num_nodes_degree_max = null_coalescence(self.opt.num_nodes_degree_max, 20*self.opt.num_bds_max)

        if self.opt.load_model:
           self.opt.recursion_threshold = null_coalescence(self.opt.recursion_threshold, -1)
            
        if len(self.opt.dataset_list) == 1:
            self.opt.total_runtime = null_coalescence(self.opt.total_runtime, 360)
        elif self.opt.load_model:
            self.opt.total_runtime = null_coalescence(self.opt.total_runtime, 6000)
            
        return self.opt