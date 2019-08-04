import torch
import contextlib

class EarlyStopException(Exception):
    pass

class LoopComponent():
    r"""
    Base class for loop components. Each method is a callback to be
    invoked by a `TrainTestLoop` instance, which is passed as an input.
    If the loop instance has multiple components, on each iteration their
    callbacks will be invoked in the following order:
    `
        Iteration
    +------->-------+
    |               |
    |   +-----------v-----------+-----------------------+--
    |   |   Loop component #1   |   Loop component #2   |   ...
    |   +-----------------------+-----------------------+--
    |   |   on_train_begin -----+-> on_train_begin -----+-> ...
    |   |     on_epoch_begin ---+---> on_epoch_begin ---+---> ...
    |   |       on_batch_begin -+-----> on_batch_begin -+-----> ...
    |   |            :          |            :          |
    |   |       on_batch_end ---+----->  on_batch_end --+-----> ...
    |   |     on_epoch_end -----+---> on_epoch_end -----+---> ...
    |   |   on_train_end -------+-> on_train_end -------+-> ...    :
    |   +-----------------------+-----------------------+--        |
    |                                                              v
    +-------------------------------<------------------------------+
    `
    """
    def on_train_begin(self, loop):   pass  # called by loop at start of training
    def on_epoch_begin(self, loop):   pass  # called by loop at start of each epoch
    def on_batch_begin(self, loop):   pass  # called by loop at start of each batch
    def on_grads_reset(self, loop):   pass  # called by loop to zero out gradients
    def on_forward_pass(self, loop):  pass  # called by loop to compute forward pass
    def on_loss_compute(self, loop):  pass  # called by loop to compute model loss
    def on_backward_pass(self, loop): pass  # called by loop to compute backward pass
    def on_optim_step(self, loop):    pass  # called by loop to compute/schedule optim
    def on_batch_end(self, loop):     pass  # called by loop at end of each batch
    def on_epoch_end(self, loop):     pass  # called by loop at end of each epoch
    def on_train_end(self, loop):     pass  # called by loop at end of training

class TrainTestLoop():
    r"""
    Composable loop for training and testing PyTorch models. On each
    iteration of the loop, computations are performed by one or more
    `LoopComponent` instances that access and modify loop state. The
    number and order of loop components can be modified at any time.

    Args:
        model: `torch.nn.Module` object containing the model.
        components: iterable of `LoopComponent` instances that perform
            computations on each iteration, in order of invocation.
        train_data: iterable for which len() returns length.
        valid_data: iterable for which len() returns length.

    Methods:
        train(n_epochs): train model for n_epochs: int.
        test(train_data): test model on previously unseen train_data:
            iterable for which len() returns length.
        stop(): stop early and, if training and validating, invoke the
            'on_train_end' callbacks of all loop components. Any
            component of the loop can call stop() at any time.

    Sample usage:
        >>> loop = TrainTestLoop(model, components, train_data, valid_data)
        >>> loop.train(n_epochs)
        >>> loop.test(test_data)
        >>> print(*vars(loop), sep='\n')  # vars holding loop state
    """
    def __init__(self, model, components, train_data, valid_data):
        self.model, self.components, self.train_data, self.valid_data = (model, list(components), train_data, valid_data)
        self.epoch_num = 0

    def _components_do(self, *args):
        for callback in [getattr(comp, arg) for arg in args for comp in self.components]:
            callback(self)

    def _run_epoch(self, data, desc):
        self.n_batches, self.epoch_desc = (len(data), desc)
        self.is_training, self.is_validating, self.is_testing = [desc == s for s in ('train', 'valid', 'test')]
        assert [self.is_training, self.is_validating, self.is_testing].count(True) == 1
        self.model.train() if self.is_training else self.model.eval()
        with torch.no_grad() if not self.is_training else contextlib.suppress():
            self._components_do('on_epoch_begin')
            if self.is_training: self._components_do('on_grads_reset')
            for self.batch_num, self.batch in enumerate(iter(data)):
                self._components_do('on_batch_begin', 'on_forward_pass', 'on_loss_compute')
                if self.is_training:
                    self._components_do('on_backward_pass', 'on_optim_step')
                    self.optim_step_num += 1
                self._components_do('on_batch_end')
            self._components_do('on_epoch_end')

    def train(self, n_epochs):
        self.n_epochs = n_epochs
        self.n_optim_steps, self.optim_step_num = (self.n_epochs * len(self.train_data), 0)
        self._components_do('on_train_begin')
        for _ in range(n_epochs):
            try:
                self._run_epoch(self.train_data, 'train')
                self._run_epoch(self.valid_data, 'valid')
                self.epoch_num += 1
            except EarlyStopException: break
        self._components_do('on_train_end')
        
    def test(self, test_data):
        try:
            self.n_epochs = 1
            self._run_epoch(test_data, 'test')
        except EarlyStopException: pass

    def stop(self):
        raise EarlyStopException
