Usage
=====



Simple example
--------------

Generate 3 Brownian motion paths. All paths start at 1, and have 6 steps with
a stepsize of dt=1/4. The drift is -2 and volatility is 0.7.

The result is a 2d numpy array with 3 columns for the 3 random paths.

.. exec_code::

    from quantfinlib.sim import BrownianMotion

    bm = BrownianMotion(drift=-2, vol=0.7)
    paths = bm.path_sample(x0=1, dt=1/4, num_steps=6, num_paths=3)

    print(paths)
  

Adding Date-time indices
------------------------

Add date time labels, and return a Pandas DataFrame with a DateTime index:

In this example we add date-time index labels by specifying **label_start** and **label_freq**. 
The first row is labeled **2000-01-01** and we specify we want daily (**D**) timesteps frquences. 
We also set **dt = 1/365** so that the simulation timesteps align with the label timesteps.

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

Start label
............

Many input types are supported for the start label, e.g.

* Strings like "Jul 31, 2009", "2009-12-31", "2009/12/31"
* Python datetime.date or datetime.datetime, like datetime(2023, 6, 30)
* Epoch timstamp like 1349720105


Label Frequency
...............

Freqency string are compatible with the Pandas **date_range** function. More frequency strings can be founds in 
the Pandas documentation 
`Pandas Timeseries Offset Aliases <https://pandas.pydata.org/docs/user_guide/timeseries.html#timeseries-offset-aliases>`_ .

The date time index labels are set independently from the path simulation routines.  
If you specifiy a **freq** in the **path_sim** function then you can ommit the simulation step size **dt**. The **dt** timstep 
will then default to the timestep listed in the following table.

====  =====================  ===========
Freq  Sim step-size (years)  Description
====  =====================  ===========
D     1 / 365                Daily
B     1 / 252                Business days Modays till Friday, skipping Saturday and Sunday.
W     1 / 52                 Weekly
M     1 / 12                 Monthly
MS    1 / 12                 Month starts
ME    1 / 12                 Month ends
Q     1 / 4                  Quarterly
QS    1 / 4                  Quater starts
QE    1 / 4                  Quarter ends
YS    1                      Yearly starts
YE    1                      Yearly ends
====  =====================  ===========



In the following example we say we want to start on Jan 12th 2000, and have all 
labels fall on month ends. The first label wil be the last day of them month Jan 2000, 
and consecutive labels on te next month ends. We don't specify **dt** which will make 
it default to 1 / 12.

.. exec_code::

    from quantfinlib.sim import BrownianMotion

    bm = BrownianMotion(drift=-2, vol=0.7)

    paths = bm.path_sample(
        x0=1,  num_steps=6, num_paths=3, 
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
        cor=[[1, 0.4],
             [0.4, 1]]
        )
    
    paths = bm.path_sample(
        x0=[1, 2], num_steps=6, num_paths=3, 
        label_start='2000-01-01', 
        label_freq='D'
        )

    print(paths)


Column Names
............

Optionally you can set the columns names with the **columns** argument. 

.. exec_code::

    from quantfinlib.sim import BrownianMotion

    bm = BrownianMotion(
        drift=[1.0, -0.1], 
        vol=[42, 0.1], 
        cor=[[1, 0.9], 
             [0.9, 1]]
        )
    
    paths = bm.path_sample(
        x0=[25.0, 0.011], num_steps=6, num_paths=3, 
        label_start='2000-01-01', 
        label_freq='D',
        columns=['GME', 'DOGE']
        )

    print(paths)