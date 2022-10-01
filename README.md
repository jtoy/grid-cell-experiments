# grid-cell-experiments
The goal of this project is to get a deeper understanding of grid cells with code.

I've read dozens of papers on grid cells theory and now its time to implement it into computers.

This project reimplements DeepMind's paper: Vector-based navigation using grid-like representations in artificial agents https://www.deepmind.com/publications/vector-based-navigation-using-grid-like-representations-in-artificial-agents

This code is written in python3 and pytorch.

Of note, we were able to get grid like patterns to appear similar to the original paper, but only when using a batch size of 10, if we increased it, the number of grid like cells went down dramatically.
We also found that the better the path integration results were not that good and did not seem to be connected with grid cell patterns appearing.
In the images below you can see some of the results:


![path integration](https://github.com/jtoy/grid-cell-experiments/blob/main/images/paths.png?raw=true)
![sac scores](https://github.com/jtoy/grid-cell-experiments/blob/main/images/sac_scores.png?raw=true)



To learn more about grid cells, read [this](https://targetpattern.com/)

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

### Development

Pre-commit hooks with forced pytahon formatting ([black](https://github.com/psf/black), [flake8](https://flake8.pycqa.org/en/latest/), and [isort](https://pycqa.github.io/isort/)):

```sh
pip install pre-commit
pre-commit install
```

Whenever you execute `git commit` the files altered / added within the commit will be checked and corrected. `black` and `isort` can modify files locally - if that happens you have to `git add` them again.
You might also be prompted to introduce some fixes manually.

To run the hooks against all files without running `git commit`:

```sh
pre-commit run --all-files
```

---

Goal: Get a deeper understanding of using deepmind's grid cell paper. Can we reproduce the experiments and graphs? If we get a deep understanding of the system, we may be able to produce our own experiments.



Potential modifications:
* Can we implement our own trajectories with an artifical agent
* can we change out the model to use transformers
* can we use other forms of attention


# next steps:

* encoding from 16 to 32 - number of points to quantize 2d space (decrease/increase) (e)
* turn off dropout (e)
* run small grid search across hyperparameters:
  * hyper parameter numbers (e)
  * lstm number (e)
  * different optimizers (m)

* turn off HD cells for prediction (h)
* try 3D (h)

* get the model to get as close of results to deepmind as possible
  * rmsprop is different: https://github.com/pytorch/pytorch/issues/32545
  * scoring added to report
  * datasamples, use same way as deepmind , maybe not worth it?

* seed control


# Questions

* What is the minimum architecture to make the grid cells appear
* what would an abstract space with cartesian coordinates look like
* why do the other paper get grid cells in different shapes
* are speed cells in the data, how would prove this
* does a regular lstm/rnn trained on simliar dataset get a similar loss
* testing out 3D, how would we generate the training data
