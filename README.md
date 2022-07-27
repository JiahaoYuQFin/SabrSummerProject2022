# SabrSummerProject2022 Instruction
This is a heding strategies project using sabr model shared by Weitao Chen, Dongyu Wei, and Jiahao Yu.
## OptionModel

```python
Examples:
>>> from OptionModel.sabr import SabrHagan2002

# market info
>>> opt_price = np.array([0.1575, 0.0989, 0.0574])
>>> strike = np.array([2.8, 2.9, 3.0])
>>> spot=2.8870
>>> texp=68/365.25

# init model
>>> sabrmodel = SabrHagan2002(sigma=0.2, vov=0.6, rho=0.3, beta=0.6)

# 3 options for the calibration; 
# `is_vol` is whether the input variable is price or implied vol; 
# `setval` is whether we set the calibrated params as the model params.
>>> sabrmodel.calibrate3(price_or_vol3=opt_price, strike3=strike, spot=spot, texp=texp, is_vol=False, setval=True)
{'sigma': 0.31446090777041097,
 'vov': 1.5805873590513595,
 'rho': -0.2536946574367376}
>>> sabrmodel.price(strike=np.array([3.2, 3.1, 2.95, 2.85, 2.75]), spot=spot, texp=texp, cp=1)
array([0.01729237, 0.03168252, 0.07599214, 0.12615207, 0.192496  ])
>>> sabrmodel.delta_numeric(strike=strike, spot=spot, texp=texp, cp=1)
array([0.68602924, 0.52392876, 0.34993297])
```
## HedgingModel

Plz refer to .$\backslasb$HedgingModel$\backslasb$README.md and .$\backslasb$HedgingModel$\backslasb$analysis_demo.py