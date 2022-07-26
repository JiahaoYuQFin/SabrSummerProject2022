# SabrSummerProject2022
## Instruction
This is a heding strategies project using sabr model shared by Weitao Chen, Dongyu Wei, and Jiahao Yu.
## OptionModel
from OptionModel.sabr import SabrHagan2002
# 3 options for the calibration; 
# `is_vol` is whether the input variable is price of implied vol; 
# `setval` is whether we set the calibrated params as the model params.
sabrmodel.calibrate3(price_or_vol3=opt_price, strike3=strike, spot=spot, texp=texp, is_vol=False, setval=True)
>>>{'sigma': 0.31446090777041097,
 'vov': 1.5805873590513595,
 'rho': -0.2536946574367376}
 sabrmodel.price(strike=np.array([3.2, 3.1, 2.95, 2.85, 2.75]), spot=spot, texp=texp, cp=1)
 >>>array([0.01729237, 0.03168252, 0.07599214, 0.12615207, 0.192496  ])
 sabrmodel.delta_numeric(strike=strike, spot=spot, texp=texp, cp=1)
 >>>array([0.68602924, 0.52392876, 0.34993297])
