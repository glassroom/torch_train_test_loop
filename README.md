# torch_train_test_loop

Composable training/testing of PyTorch deep learning models with minimal overhead:

```python
from torch_train_test_loop import TrainTestLoop, LoopComponent

class MainLoop(LoopComponent):

    def on_train_begin(self, loop):
        loop.loss_func = CrossEntropyLoss()
        loop.optimizer = MyOptimizer(loop.model.parameters(), lr=3e-4)
        loop.scheduler = MyScheduler(loop.optimizer, loop.n_optim_steps)

    def on_grads_reset(self, loop):
        loop.model.zero_grad()

    def on_forward_pass(self, loop):
        model, batch = (loop.model, loop.batch)
        loop.scores = model(batch.data)

    def on_loss_compute(self, loop):
        scores, labels = (loop.scores, loop.batch.labels)
        loop.loss = loop.loss_func(scores, labels)

    def on_backward_pass(self, loop):
        loop.loss.backward()

    def on_optim_step(self, loop):
        loop.optimizer.step()
        loop.scheduler.step()

    def on_train_end(self, loop):
        print("Done.")

class SaveModel(LoopComponent):

    def on_train_end(self, loop):
        torch.save(loop.model.state_dict(), './model_state.pth')
        print("Saved.")

loop = TrainTestLoop(model=my_model, components=[MainLoop(), SaveModel()],
                     train_data=my_train_data, valid_data=my_valid_data)
loop.train(n_epochs=10)
```
Output:
```
Done.
Saved.
```

## Why?

We were unable to find a simple, composable, standalone tool for manipulating training loops *without* the overhead and complexity of a full-blown framework.

If you regularly find yourself digging through code path dependencies to figure out how to try something new in your training loop, this tool is for you. It tries to do the bare minimum necessary for composing loops without getting in your way.  Also, the code is meant to be easy to understand/modify -- under 70 lines of Python excluding comments.

## Installing

```
pip install git+https://github.com/glassroom/torch_train_test_loop
```

Alternatively, you can download a single file to your project directory: [torch_train_test_loop.py](torch_train_test_loop/torch_train_test_loop.py).

The only dependency is PyTorch.

## Overview

**torch_train_test_loop** consists of only two classes, `TrainTestLoop` and `LoopComponent`, that work together:

* `TrainTestLoop` contains logic for running training/validation and testing loops: It manages epochs and batches, iterates over datasets, sets a torch.no_grad() context for validating and testing, changes model state to train() or eval() as necessary, and manages other variables that control loop state. All other computations are performed by invoking callbacks of one or more `LoopComponent` instances at predefined points on each iteration.

* `LoopComponent` contains callback methods invoked by a `TrainTestLoop` instance at predefined points on each iteration. For a list of predefined callback methods, see the [class definition](torch_train_test_loop.py). If a loop has multiple components, their callbacks are invoked in the following order:

```
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
|   |         <all others> -+-------> <all others> -+-------> ...
|   |            :          |            :          |
|   |       on_batch_end ---+----->  on_batch_end --+-----> ...
|   |     on_epoch_end -----+---> on_epoch_end -----+---> ...
|   |   on_train_end -------+-> on_train_end -------+-> ...    :
|   +-----------------------+-----------------------+--        |
|                                                              v
+-------------------------------<------------------------------+
```

Each additional component thus adds new a layer of functionality to all predefined callback points in a loop.

## Dynamic Manipulation of Loop Components

Loop instances store components in a standard Python list, so they can be dynamically inserted, deleted, replaced, and reordered at any time:

```python
# Create a train/test loop.
my_components = (MyInitializer(), MyBatchProcessor(), MyOptimManager(), MyStats())
loop = TrainTestLoop(my_model, my_components, train_data, valid_data)

# Insert a new component in the second position to preprocess batches.
loop.components.insert(1, MyPreprocessing())

# Delete the last component.
del loop.components[-1]
```

## Loop Variables

The following variables are controlled by the loop instance and available to its components:

```
Variable             Description
===================  =====================================================
loop.model           PyTorch model
loop.components      list of LoopComponent instances
loop.train_data      iterable of training data
loop.valid_data      iterable of validation data

loop.n_epochs        number of epochs in current run
loop.n_batches       number of batches in current epoch
loop.n_optim_steps   number of optimization steps in current training run

loop.is_training     set by loop to True if training, False otherwise
loop.is_validating   set by loop to True if validating, False otherwise
loop.is_testing      set by loop to True if testing, False otherwise

loop.epoch_desc      set by loop to 'train', 'valid' or 'test'
loop.epoch_num       number of training epochs since instantiation of loop

loop.batch           object yielded by iteration of current dataset
loop.batch_num       batch number in current epoch

loop.optim_step_num  optimization step number in current training run
```

## Notes

The code is as simple as we could make it (e.g., we refrained from building in a fancier callback-handling mechanism for components) and as Pythonic as we could make it (e.g., when invoking a call method from a loop instance, we explicitly pass a reference to the loop object, instead of, say, dynamically binding callbacks to the loop object).
