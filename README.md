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


Technology: 
We will port it to Python 3 and pytorch.




Potential modifications:
* Can we implement our own trajectories with an artifical agent
* can we change out the model to use transformers
* can we use other forms of attention

Next steps:
1 private github repo of copy deepmind
2 Private Github repo new code
3 repurpose the data set
4 exploratory code to examine the dataset

Design doc:
training loop
Validation procedures
Visualization 


@tomek can you look at your schedule to see if we could meet twice a week?  I can do almost any days/times. 



