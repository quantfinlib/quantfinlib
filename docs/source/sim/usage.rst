Usage
=====

Generate 3 Brownian motion paths. All paths start at 1, and have 6 steps with
a stepsize of dt=1/4. The drift is -2 and volatility is 0.7.

The result is a 2d numpy array with the paths in columns

.. exec_code::

    from quantfinlib.sim import BrownianMotion

    bm = BrownianMotion(drift=-2, vol=0.7)
    paths = bm.path_sample(x0=1, dt=1/4, num_steps=6, num_paths=3)

    print(paths)
  

Add date time labels, and return a Pandas DataFrame with a DateTime index:

.. exec_code::

    from quantfinlib.sim import BrownianMotion

    bm = BrownianMotion(drift=-2, vol=0.7)

    paths = bm.path_sample(
        x0=1, dt=1/4, num_steps=6, num_paths=3, 
        label_start='2000-01-01', 
        label_freq='D'
    )

    print(paths)

Simulation of two correlated assets, the columns are organized in pair of
correlated stocks, and the columns hames have **_<scenario number>** appended.

.. exec_code::

    from quantfinlib.sim import BrownianMotion

    bm = BrownianMotion(
        drift=[0.05, 0.05], 
        vol=[0.5, 0.5], 
        cor=[[1, 0.4], [0.4, 1]]
        )
    
    paths = bm.path_sample(
        x0=[1, 2], dt=1/4, num_steps=6, num_paths=3, 
        label_start='2000-01-01', 
        label_freq='D'
        )

    print(paths)