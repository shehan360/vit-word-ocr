import torch.nn as nn

from modules.vitstr import create_vitstr


class Model(nn.Module):
    def __init__(self, opt):
        super(Model, self).__init__()
        self.vitstr= create_vitstr(num_tokens=opt.num_class, model=opt.TransformerModel)

    def forward(self, input, text, is_train=True, seqlen=25):
        prediction = self.vitstr(input, seqlen=seqlen)
        return prediction