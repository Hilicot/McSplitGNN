from options.base_options import BaseOptions,null_coalescence

class TestOptions(BaseOptions):
    def initialize(self):
        super().initialize()
        self.parser.add_argument('--promise_mode', default='P')

    def parse(self):
        super().parse()
        
        self.opt.phase = 'test'
        
        if self.opt.load_model:
            self.opt.recursion_threshold = null_coalescence(self.opt.recursion_threshold, 10000 if len(self.opt.dataset_list) == 1 else 7500)
                    
        if len(self.opt.dataset_list) == 1:
            self.opt.num_nodes_degree_max = null_coalescence(self.opt.num_nodes_degree_max, 3*self.opt.num_bds_max)
        else:
            self.opt.num_nodes_degree_max = null_coalescence(self.opt.num_nodes_degree_max, 20*self.opt.num_bds_max)
        
        return self.opt
