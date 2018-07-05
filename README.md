# MiniBOP

MiniBOP is a python package for Parallel and Scalable Bayesian Optimization.
The aim is to provide simple algorithms that can be run in a parallel computing
environment based on current (2018) literature.

We provide two similar algorithms for choosing points for evaluation:
* Bayesian Optimization with Poll steps (BOP)
* Function Barrier Variance Control (FuBar VC)

These, represents a heuristic synthesis of several recent ideas from the literature
in lieu of well developed, easy to use software implementations.
For a short review of Bayesian Optimization, references and algorithms description
see my [technical report](https://arxiv.org/abs/1807.00373).

I tried to design the code to be modular and hackable, so that individual parts
could be easily modified/replaced with (hopefully) no or little impact on other
parts.

## Credits

Some of the ideas and code were taken from the [Spearmint-lite](https://github.com/JasperSnoek/spearmint/)
and [George](https://github.com/dfm/george) projects.

## Algorithms' Features

* Suitable for a parallel environment in which many function evaluations are carried concurrently.
    - Bayesian inference is carried in parallel.
    - A variant of Thompson sampling is used to increase variability of the chosen points.
* Controlled predictive variance:
    - We do not select points for evaluation if the predictive variance at these
    points is below a certain threshold. This increases exploration once local
    function minima are found.
* The algorithm alternates between global Bayesian searches and local exploration
around estimated local minima (Poll steps).
* Boundary Avoidances:
    - In high dimension, Bayesian inference tends to choose many points on the
    edges of the parameter range. Since typically one does not expect to find
    the optimal parameter value at these location (this is how the range is
    usually chosen) the algorithms discard inferred points if they hit the
    parameters' boundary.

## Documentation

The miniBOP_2D notebook contains a simple working example.
At this point, further documentation can be found in my [technical report](https://arxiv.org/abs/1807.00373)
and inside the source code.

## Dependencies:

* Python3
* Numpy
* Scipy
* [George](https://github.com/dfm/george)
* [Ipyparallel](https://github.com/ipython/ipyparallel) (Optional)

## Support

Questions and comment are welcomed.
