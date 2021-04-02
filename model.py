import torch.nn as nn
import backbone as backbone
import torch.nn.functional as F
import torch
import numpy as np
from torch.autograd import Function

def softXEnt (input, target):
    logprobs = torch.nn.functional.log_softmax(input, dim=1)
    return  -(target * logprobs).sum() / input.shape[0]

class GradientReverseLayer(Function):
    @staticmethod
    def forward(ctx, inps, domain, alpha):
        ctx.alpha = alpha
        ctx.domain = domain
        return inps
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()*ctx.alpha, None, None

class MDDNet(nn.Module):
    def __init__(self, base_net='ResNet50', use_bottleneck=True, bottleneck_dim=1024, width=1024, class_num=10):
        super(MDDNet, self).__init__()
        ## set base network
        self.base_network = backbone.network_dict[base_net]()
        self.use_bottleneck = use_bottleneck
        self.grl_layer = GradientReverseLayer.apply
        self.bottleneck_layer_list = [nn.Linear(self.base_network.output_num(), bottleneck_dim), nn.BatchNorm1d(bottleneck_dim), nn.ReLU(), nn.Dropout()]
        self.bottleneck_layer = nn.Sequential(*self.bottleneck_layer_list)
        self.classifier_layer_list = [nn.Linear(width, class_num, bias=True)]
        self.classifier_layer = nn.Sequential(*self.classifier_layer_list)
        self.classifier_layer_2_list = [nn.Linear(width, class_num, bias=True)]
        self.classifier_layer_2 = nn.Sequential(*self.classifier_layer_2_list)
        self.softmax = nn.Softmax(dim=1)

        ## initialization
        self.bottleneck_layer[0].weight.data.normal_(0, 0.005)
        self.bottleneck_layer[0].bias.data.fill_(0.1)
        # for dep in range(2):
        #     self.classifier_layer_2[dep * 3].weight.data.normal_(0, 0.01)
        #     self.classifier_layer_2[dep * 3].bias.data.fill_(0.0)
        #     self.classifier_layer[dep * 3].weight.data.normal_(0, 0.01)
        #     self.classifier_layer[dep * 3].bias.data.fill_(0.0)


        ## collect parameters
        self.parameter_list = [{"params":self.base_network.parameters()},
                               {"params":self.bottleneck_layer.parameters()},
                               {"params":self.classifier_layer.parameters()},
                               {"params":self.classifier_layer_2.parameters()}]

    def forward(self, inputs, domain=None, alpha=0.01):
        features = self.base_network(inputs)
        if self.use_bottleneck:
            features = self.bottleneck_layer(features)
        features_adv = self.grl_layer(features, domain, alpha)
        outputs_adv = self.classifier_layer_2(features_adv)
        
        outputs = self.classifier_layer(features)
        softmax_outputs = self.softmax(outputs)

        return features, outputs, softmax_outputs, outputs_adv

class MDD(object):
    def __init__(self, base_net='ResNet50', width=1024, class_num=10, use_bottleneck=True, use_gpu=True, srcweight=3):
        self.c_net = MDDNet(base_net, use_bottleneck, width, width, class_num)

        self.use_gpu = use_gpu
        self.is_train = False
        self.class_num = class_num
        if self.use_gpu:
            self.c_net = self.c_net.cuda()
        self.srcweight = srcweight


    def get_loss(self, inputs, labels_source, domain, alpha=0.01):
        class_criterion = nn.CrossEntropyLoss()
        class_criterion2 = softXEnt

        _, outputs, _, outputs_adv = self.c_net(inputs, domain=domain, alpha=alpha)

        src_idx = domain==0
        tgt_idx = domain==1

        classifier_loss = class_criterion(outputs[src_idx], labels_source[src_idx])

        target_adv = outputs.argmax(1)
        classifier_loss_adv_src = class_criterion(outputs_adv[src_idx], target_adv[src_idx])

        logloss_tgt = torch.log(1.00001 - F.softmax(outputs_adv[tgt_idx], dim=1))
        classifier_loss_adv_tgt = F.nll_loss(logloss_tgt, target_adv[tgt_idx])

        transfer_loss = self.srcweight * classifier_loss_adv_src + classifier_loss_adv_tgt
        return classifier_loss, transfer_loss

    def predict(self, inputs):
        _, _, softmax_outputs,_= self.c_net(inputs)
        return softmax_outputs

    def get_parameter_list(self):
        return self.c_net.parameter_list

    def set_train(self, mode):
        self.c_net.train(mode)
        self.is_train = mode

