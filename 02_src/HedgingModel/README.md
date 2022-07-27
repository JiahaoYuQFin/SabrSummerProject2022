# SabrSummerProject2022--Hedging module
 This is hedging module, done by Weitao Chen.

### Requirements

import dataframe must contain columns: time, code, type, greeks, signal

### Option Construct Part

Put-call parity: Put_Call_Parity_l(), Put_Call_Parity_s()

Straddle: Straddle()

Spread: Bull_Call_Spread()

### Position Control Part

Open position according to previous volume and hold until signal revert or vanish: Option_Position_Hold()

### Hedge Strategy Part

Use underlying spot to hedge: Hedge_Spot()

Use ATM option to hedge: Hedge_ATM()

### Note

Note that before entering position control and hedge part, plz use Hedge_Transform() to do data preparation