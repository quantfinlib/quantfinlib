Usage
=====



Simple example
--------------

Generate 3 Brownian motion paths. All paths start at 1, and have 6 steps with
a stepsize of dt=1/4. The drift is -2 and volatility is 0.7.

The result is a 2d numpy array with the paths in columns

.. exec_code::

    from quantfinlib.sim import BrownianMotion

    bm = BrownianMotion(drift=-2, vol=0.7)
    paths = bm.path_sample(x0=1, dt=1/4, num_steps=6, num_paths=3)

    print(paths)
  

Adding Date-time indices
------------------------

Add date time labels, and return a Pandas DataFrame with a DateTime index:

In this example we add date-time index labels. The first row is labeled **2000-01-01**
and we specify we want daily (**D**) timesteps frquences. We also set **dt = 1/365**
so that the simulation timesteps align with the label timesteps.

.. exec_code::

    from quantfinlib.sim import BrownianMotion

    bm = BrownianMotion(drift=-2, vol=0.7)

    paths = bm.path_sample(
        x0=1, dt=1/365, num_steps=6, num_paths=3, 
        label_start='2000-01-01', 
        label_freq='D'
    )

    print(paths)

Other frequency string options are: 

* D: Daily
* B: Business days Modays till Friday, skipping Saturday and Sunday.
* W: Weekly
* MS: Month starts
* ME: Month ends
* QS: Quater starts
* QE: Quarter ends
* YS: Yearly starts
* YE: Yearly ends

These freqency string are compatible with the Pandas **date_range** function. More frequency strings can be founds in 
the Pandas documentation 
`Pandas Timeseries Offset Aliases <https://pandas.pydata.org/docs/user_guide/timeseries.html#timeseries-offset-aliases>`_ .


In the following example we say we want to start on Jan 12th 2000, and have all 
labels fall on month ends. The first label wil be the last day of them month Jan 2000, 
and consecutive labels on te next month ends.

.. exec_code::

    from quantfinlib.sim import BrownianMotion

    bm = BrownianMotion(drift=-2, vol=0.7)

    paths = bm.path_sample(
        x0=1, dt=1/12, num_steps=6, num_paths=3, 
        label_start='2000-01-12', 
        label_freq='ME'
    )

    print(paths)


Simulating correlated assets
----------------------------

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