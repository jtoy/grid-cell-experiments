# grid-cell-experiments

### Running:

Install dependencies:

```sh
pip install -r requirements.txt
```

Run training loop:

```sh
python -m gridcells.main
```

Use tensorboard to monitor progrss:

```sh
tensorboard --logdir=tmp/tensorboard --host=0.0.0.0
```

---

Goal: Get a deeper understanding of using deepmind's grid cell paper. Can we reproduce the experiments and graphs? If we get a deep understanding of the system, we may be able to produce our own experiments.



Potential modifications:
* Can we implement our own trajectories with an artifical agent
* can we change out the model to use transformers
* can we use other forms of attention


Design doc:
training loop
Validation procedures
Visualization 

next steps:
* get the model to get as close of results to deepmind as possible
  * rmsprop is different: https://github.com/pytorch/pytorch/issues/32545
  * email to deepmind/lukas
  * scoring added to report
  * datasamples, use same way as deepmind , maybe not worth it?

* run small grid search across hyperparameters:
  * different optimizers
  * number of pointse to quantize 2d space (decrease/increase)
* change the data
  * try 3D 


# Questions

* What is the minimum architecture to make the grid cells appear
* what would an abstract space with cartesian coordinates look like
* why does the other paper get grid cells in different shapes
* are speed cells in the data, how would prove this
* does a regular lstm/rnn trained on simliar dataset get a similar loss
* testing out 3D, how would we generate the training data
* what is the loss representing


