import torch
from torch.autograd import Function

class GradientReversalFunctionBKP(Function):
    """
    Gradient Reversal Layer from:
    Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)
    Forward pass is the identity function. In the backward pass,
    the upstream gradients are multiplied by -lambda (i.e. gradient is reversed)
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_outputs):
        lambda_ = ctx.lambda_
        lambda_ = grad_outputs.new_tensor(lambda_)
        dx = lambda_ * grad_outputs
        # pdb.set_trace()
        print(dx)
        return dx, None
    # @staticmethod
    # def backward(ctx, *grad_outputs):
    #     lambda_ = ctx.lambda_
    #     lambda_ = grad_outputs.new_tensor(lambda_)
    #     dx = ctx.mask * lambda_ * grad_outputs
    #     return dx, None


class GradientReversalFunction(Function):
    """
    Gradient Reversal Layer from:
    Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)
    Forward pass is the identity function. In the backward pass,
    the upstream gradients are multiplied by -lambda (i.e. gradient is reversed)
    """

    @staticmethod
    def forward(ctx, x, mask, lambda_):
        ctx.mask = mask
        ctx.lambda_ = lambda_
        #ctx.att = att
        return x.clone()

    @staticmethod
    def backward(ctx, grad_outputs):
        lambda_ = ctx.lambda_
        lambda_ = grad_outputs.new_tensor(lambda_)
        #att = ctx.att
        #pdb.set_trace()
        dx = ctx.mask.unsqueeze(1).repeat([1, grad_outputs.shape[1]]) * lambda_ * grad_outputs
        #print(dx)
        return dx, None, None
    # @staticmethod
    # def backward(ctx, *grad_outputs):
    #     lambda_ = ctx.lambda_
    #     lambda_ = grad_outputs.new_tensor(lambda_)
    #     dx = ctx.mask * lambda_ * grad_outputs
    #     return dx, None

class GradientReversal(torch.nn.Module):
    def __init__(self, lambda_=1):
        super(GradientReversal, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        #print('x', x)
        return GradientReversalFunction.apply(x[0], x[1], self.lambda_),x[2]