# torch_train_test_loop

Composable training/testing of PyTorch deep learning models with minimal overhead:

```python
from torch_train_test_loop import TrainTestLoop, LoopComponent

class MainLoop(LoopComponent):

    def on_train_begin(self, loop):
        loop.loss_func = CrossEntropyLoss()
        loop.optimizer = MyOptimizer(loop.model.parameters(), lr=3e-4)
        loop.scheduler = MyScheduler(loop.optimizer, loop.n_optim_steps)

    def on_forward_pass(self, loop):
        model, batch = (loop.model, loop.batch)
        model.zero_grad()
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

loop = TrainTestLoop(my_model, [MainLoop(), SaveModel()], my_train_data, my_valid_data)
loop.train(n_epochs=10)
```
Output:
```
Done.
Saved.
```

## Why?

We were unable to find a simple, composable, standalone tool for manipulating training loops *without* the overhead and complexity of a full-blown framework.

If you regularly find yourself digging through code path dependencies to figure out how to try something new in your training loop, this tool is for you. It tries to do the bare minimum necessary for composing loops without getting in your way. The code is meant to be easy to understand and modify, filling just over two screens of a typical laptop display including comments.

## Overview

**torch_train_test_loop** consists of just two classes, `TrainTestLoop` and `LoopComponent`, that work together:

* `TrainTestLoop` contains barebones logic for running training/testing loops, keeping track of epochs and batches, setting a torch.no_grad() context for validating and testing phases, and managing other variables that control loop state. All other computations are performed by invoking callbacks of one or more `LoopComponent` instances, which access and modify loop state at predefined points on each iteration.

* `LoopComponent` contains callback methods that are invoked by a `TrainTestLoop` instance at predefined points on each iteration. For a list of predefined callback methods, see the [class definition](torch_train_test_loop.py). If a loop has multiple components, their callbacks are invoked in the following order:

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

Loop components are stored in a standard Python list, so you can dynamically insert, delete, replace, and reorder them at any time:

```python
# Create a train/test loop.
my_components = (MyInitializer(), MyBatchProcessor(), MyOptimManager(), MyStats())
loop = TrainTestLoop(my_model, my_components, train_data, valid_data)

# Insert a new component in the second position.
loop.components.insert(1, MyPreprocessing())

# Delete the last component.
del loop.components[-1]
```

The code is as simple as we could make it (e.g., we refrained from building in a fancier callback-handling mechanism) and as Pythonic as we could make it (e.g., when invoking a call method, we explicitly pass a reference to the loop object, instead of, say, dynamically binding callbacks to the loop object).  

By convention, components use "self" to refer to themselves and "loop" to refer to the calling loop.

## Installation

1. Clone this repository.

2. Copy the file `torch_train_test_loop.py` to your application directory.

3. There is no step #3.

## Requirements

The only requirement is a working installation of [PyTorch](https://pytorch.org/) on Python 3.

Note: Tested only with PyTorch versions 1.0 and 1.1, on Ubuntu Linux 18.04 and Python 3.6+.
