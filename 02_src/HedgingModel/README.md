# SabrSummerProject2022--Hedging module
 This is hedging module, done by Weitao Chen.

import dataframe must contain columns: time, code, type, greeks, signal

To use these functions, use pandas.dataframe.groupby('time').apply(*)

Before use functions in delta hedge module, refer to the python script to add corresponding commands. Functions in delta hedge module may not be running smoothly for now, since hedge_atm() requires the option pool remain consistent for all time.