# Modest MoPy Model Checker

Modest MoPy Model Checker is a model checker written in Python for Modest models exported to Python using Modest MoPy.

## Installation

1. Install the [Modest Toolset](https://www.modestchecker.net) and add the `modest` binary to your `$PATH`
2. Clone the github repository and navigate to the directory
```bash
$ git clone https://github.com/ethergeus/modest-mopy-model-checker.git
$ cd modest-mopy-model-checker
```

## Usage

Using the model checker requires exporting to a Python representation of the model first.
Say you have the file `mdp.modest` with the following contents:

```
// Description: Exercise 8.3 from Lecture Note.
action r, s;

bool successful= false;
bool failure = false;

property P1 = Pmax(<> (successful == true));
property P2 = Pmin(<> (successful == true));

transient int(0..8) reward = 0;
property R1 = Xmax(S(reward), successful == true);
property R2 = Xmin(S(reward), successful == true);

process Choose()
{
	alt {
	:: s {= reward = 1 =}; Safe()
	:: r {= reward = 8 =}; Risk()
	}
}

process Safe()
{
	tau palt {
	:0.1: {= successful = true =}; Success()
	:0.9: Choose()
	}
}

process Risk()
{
	alt {
		::s; Choose()
		::r palt {
		:0.5: {= successful = true =}; Success()
		:0.5: {= failure = true =}; Fail()
		}
	}
}

process Fail() { tau; Fail() }
process Success() { tau; Success() }

Choose()
```

We can export this file to Python using the following command:

```
$ modest export-to-python mdp.modest --output mdp.py
```

After exporting to Python we can start model checking:

```
./model-checker.py mdp.py [arguments]
```

Example:

```bash
$ ./model-checker.py mdp.py --value-iteration
Loading model from "modelPy/mdp-5.py"... done.
Exploring the state space... found a total of 5 states.
Performing value iteration... done. P1: p_max(eventually(ap(0))) = 0.9999912865330949
Performing value iteration... done. P2: p_min(eventually(ap(0))) = 0.0
Pre-computing Smin1... Pre-computing Smin0... Performing value iteration... done. R1: e_max_s(1, ap(0)) = inf
Pre-computing Smax1... Performing value iteration... done. R2: e_min_s(1, ap(0)) = 9.999916647515823
Done in 0.03 seconds.
```

Modest mcsta results:

```bash
$ modest mcsta mdp.modest
...
+ State space exploration
  State size:  4 bytes B
  States:      5 
  Transitions: 7 
  Branches:    9 
  Rate:        172 states/s 
  Time:        0.0 s

+ Property P1
  Probability: 0.9999921578797853 
  Bounds:      [0.9999921578797853, 1] 
  Time:        0.0 s

  + Value iteration
    Final error: 9.681714255750288E-07 
    Iterations:  106 
    Time:        0.0 s

+ Property P2
  Probability: 0 
  Bounds:      [0, 0] 
  Time:        0.0 s

  + Value iteration
    Final error: 0 
    Iterations:  2 
    Time:        0.0 s

+ Property R1
  Value:  Infinity 
  Bounds: [infinity, infinity] 
  Time:   0.0 s

  + Precomputations
    Min. prob. 0 states:          3 
    Time for min. prob. 0 states: 0.0 s

+ Property R2
  Value:  9.999924982764242 
  Bounds: [9.999924982764242, infinity) 
  Time:   0.1 s

  + Precomputations
    Max. prob. 1 states:          4 
    Time for max. prob. 1 states: 0.0 s

  + Value iteration
    Final error: 9.261464328371953E-07 
    Iterations:  112 
    Time:        0.0 s
```

### Arguments

The model checker supports a number of command line arguments, documentation can be accessed using `./model-checker.py --help`:

```
usage: model-checker.py [-h] [-p PROPERTIES [PROPERTIES ...]] [--value-iteration] [--relative-error RELATIVE_ERROR] [-k MAX_ITERATIONS] [--q-learning] [-e EPSILON] [-a ALPHA] [-g GAMMA] model

Model checker for MDPs.

positional arguments:
  model                 path to the model file

options:
  -h, --help            show this help message and exit
  -p PROPERTIES [PROPERTIES ...], --properties PROPERTIES [PROPERTIES ...]
                        list of properties to check (default: all)
  --value-iteration     use value iteration to evaluate properties
  --relative-error RELATIVE_ERROR
                        maximum relative error for value iteration (default: 1e-06)
  -k MAX_ITERATIONS, --max-iterations MAX_ITERATIONS
                        maximum number of iterations for value iteration, takes precedence over relative error
  --q-learning          use Q-learning to evaluate properties
  -e EPSILON, --epsilon EPSILON
                        epsilon (exploration probability) for Q-learning (default: 0.1)
  -a ALPHA, --alpha ALPHA
                        alpha (learning rate) for Q-learning (default: 0.1)
  -g GAMMA, --gamma GAMMA
                        gamma (discount factor) for Q-learning (default: 0.9)

by Andrey and Alex (Group 2)
```

Here `--gamma` (discount factor), `--alpha` (learning rate) and `epsilon` (exploration probability) are used in Qlearning, `--relative-error` in value iteration and `-k` in either. These are optional and the defaults are well-documented in the manual page above.
Supplying either `--value-iteration` or `--q-learning` is required.

One can supply the flag `-p` and a space, separated list of properties to check only the given properties, the model checker will (attempt to) check all properties otherwise.

### Q-learning

We implemented Q-learning as an additional feature to calculating P_opt and R_opt using value iteration.

Example:

```bash
$ ./model-checker.py modelPy/g7-dnd.py --q-learning --gamma 1 -p playerHPLeft fightDuration
Loading model from "modelPy/g7-dnd.py"... done.
playerHPLeft: e_max_s(1, ap(2)) = 5.739069850725511
fightDuration: e_min_s(3, ap(2)) = 2.244553864594371
Done in 6.89 seconds.
```

Modest modes results:

```bash
$ modest modes modelModest/g7-dnd.modest --learn Qlearning --props "playerHPLeft,fightDuration"
...
+ Property playerHPLeft
  Estimated value:     6.0173734046932115 
  Confidence interval: [6.007373410999746, 6.027373398386677] 
  Runs used:           218610 
  Run type:            MDP 
  Status:              Finished 

  + Sample data
    Mean:     6.0173734046932115 
    Variance: 5.690799627686754 
    Skewness: -0.8992818918602691 
    Kurtosis: 2.577749556169091 

  + Error bounds
    Statement: CI: 100(1 - α)% of intervals contain true value (CLT and large number of samples assumptions) 
    α:         0.050000000000000044 

+ Property fightDuration
  Estimated value:     1.4081215970961813 
  Confidence interval: [1.3981234627808081, 1.4181197314115546] 
  Runs used:           22040 
  Run type:            MDP 
  Status:              Finished 

  + Sample data
    Mean:     1.4081215970961813 
    Variance: 0.5735263049194487 
    Skewness: 2.6653059955230685 
    Kurtosis: 16.247459492414528 

  + Error bounds
    Statement: CI: 100(1 - α)% of intervals contain true value (CLT and large number of samples assumptions) 
    α:         0.050000000000000044
```

### Deep Q-learning

Deep Q learning is implemented in `model-checker.py` and can be used by supplying the `--deep-q-learning` flag. The implementation is based on the [PyTorch DQN tutorial](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html).

TODO:
- [✔️] Implement minimum and maximum probability queries
- [ ] Implement target network (double DQN)

```bash
(modest-dql) andrey@Andreys-Mac-mini ~/P/modest-mopy-model-checker> python model-checker.py modelPy/demo-mdp-2.py --deep-q-learning --gamma 0.9 --epsilon-start 0.1 --epsilon-min 0.1 -v
Loading model from "modelPy/demo-mdp-2.py"... done.
P1: p_max(eventually(ap(0))) = None
P2: p_max(eventually(ap(1))) = None
Exploring the state space... found a total of 5 states and 3 transitions.
P3: e_max_s(2, ap(3)) = 4.112936496734619 4204
P4: e_min_s(2, ap(0)) = 2.22222256660461434947
Done in 139.25 seconds.
(modest-dql) andrey@Andreys-Mac-mini ~/P/modest-mopy-model-checker> python model-checker.py modelPy/demo-mdp-2.py --q-learning --gamma 0.9 --epsilon 0.1
Loading model from "modelPy/demo-mdp-2.py"... done.
P1: p_max(eventually(ap(0))) = None
P2: p_max(eventually(ap(1))) = None
P3: e_max_s(2, ap(3)) = 3.7881236216617262
P4: e_min_s(2, ap(0)) = 0.0
Done in 1.27 seconds.
```

### Value iteration

Example:

```bash
$ ./model-checker.py modelPy/g7-dnd.py --value-iteration
Loading model from "modelPy/g7-dnd.py"... done.
Exploring the state space... found a total of 234 states.
Performing value iteration... done. playerWinsMax: p_max(eventually(ap(0))) = 0.9692100464929934
Performing value iteration... done. playerWinsMin: p_min(eventually(ap(0))) = 0.0
Pre-computing Smin1... Pre-computing Smin0... Performing value iteration... done. playerHPLeft: e_max_s(1, ap(2)) = 6.025813374951623
Pre-computing Smax1... Performing value iteration... done. fightDuration: e_min_s(3, ap(2)) = 1.4098434199501653
Done in 2.57 seconds.
```

Modest mcsta results:

```bash
$ modest mcsta modelModest/g7-dnd.modest
...
+ State space exploration
  State size:  4 bytes B
  States:      234 
  Transitions: 305 
  Branches:    1097 
  Rate:        7091 states/s 
  Time:        0.0 s

+ Property playerWinsMax
  Probability: 0.9945143447070296 
  Bounds:      [0.9945143447070296, 1] 
  Time:        0.0 s

  + Value iteration
    Final error: 3.4525892443546887E-07 
    Iterations:  17 
    Time:        0.0 s

+ Property playerWinsMin
  Probability: 0 
  Bounds:      [0, 0] 
  Time:        0.0 s

  + Value iteration
    Final error: 0 
    Iterations:  1 
    Time:        0.0 s

+ Property playerHPLeft
  Value:  6.025824690505463 
  Bounds: [6.025824690505463, infinity) 
  Time:   0.0 s

  + Precomputations
    Min. prob. 0 states:          0 
    Time for min. prob. 0 states: 0.0 s
    Min. prob. 1 states:          234 
    Time for min. prob. 1 states: 0.0 s

  + Value iteration
    Final error: 7.696623404101266E-07 
    Iterations:  54 
    Time:        0.0 s

+ Property fightDuration
  Value:  1.409845226044067 
  Bounds: [1.409845226044067, infinity) 
  Time:   0.0 s

  + Precomputations
    Max. prob. 1 states:          234 
    Time for max. prob. 1 states: 0.0 s

  + Value iteration
    Final error: 5.554297946190875E-07 
    Iterations:  19 
    Time:        0.0 s
```
