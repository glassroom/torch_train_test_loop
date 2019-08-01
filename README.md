# Composable training/testing loops for PyTorch

Composable training/testing of PyTorch deep learning models with minimal overhead:

```python
from torch_train_test_loop import TrainTestLoop, LoopComponent

# Create train/test loop with previously specified model, loop components, and data.
loop = TrainTestLoop(my_model, my_loop_components, train_data, valid_data)

# Define new kind of loop component.
class SaveBestSoFar(LoopComponent):

    def __init__(self, path):
        self.path = path
        self.value = 0.0

    def on_epoch_end(self, loop):
        if (loop.my_metric > self.value) and (loop.is_validating):
            self.value = loop.my_metric.item()
            torch.save(loop.model.state_dict(), self.path)

# Add new component to loop, train model, and show component state.
loop.components.append(SaveBestSoFar('./best_model.pth'))
loop.train(n_epochs=30)
print(loop.components[-1].value)
```

## Why?

We were unable to find a simple, composable, standalone tool for manipulating training loops *without* the overhead and complexity of a full-blown framework.

If you regularly find yourself digging through code dependencies to figure out how to try something new in your training loop, this tool is for you. It tries to do the bare minimum necessary for composing loops without getting in your way. The code is meant to be easy to understand and modify, filling just over two screens of a typical laptop display.

## Overview

**torch_train_test_loop** consists of just two classes, `TrainTestLoop` and `LoopComponent`, that work together:

* `TrainTestLoop` contains barebones logic for running training and testing loops, keeping track of number of epochs, number of batches, and other control-flow variables. All other computations in the loop are performed by invoking callbacks of one or more `LoopComponent` instances that can access and modify loop state at predefined points on each iteration.

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

Each additional component adds a layer of additional functionality across all predefined points in the loop.

Loop instances store components in a standard Python list, so they can be dynamically inserted, deleted, replaced, and reordered at any time:

```python
# Create a train/test loop.
components = (Initialize(), ManageBatch(), ManageOptim(), ManageStats())
loop = TrainTestLoop(model, components, train_data, valid_data)

# Insert a new component in the second position.
loop.components.insert(1, MixupBatches())

# Delete the last component.
del loop.components[-1]
```

The code is as simple as we could make it (e.g., we refrained from building in a fancier callback-handling mechanism).

## Installation

1. Clone this repository.

2. Copy the file `torch_train_test_loop.py` to your application directory.

3. There is no step #3.

## Requirements

The only requirement is a working installation of [PyTorch](https://pytorch.org/) on Python 3.

Note: Tested only with PyTorch versions 1.0 and 1.1, on Ubuntu Linux 18.04 and Python 3.6+.
