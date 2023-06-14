from options.base_options import BaseOptions, null_coalescence

class TrainOptions(BaseOptions):
    def initialize(self):
        super().initialize()

        self.parser.add_argument(
            "--epochs",
            type=int,
            default=10,
            help="Specify the number of epochs to train for.",
        )

    def parse(self):
        super().parse()


        self.opt.phase = 'train'

        return self.opt