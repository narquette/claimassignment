import pandas as pd
import numpy as np

def GetProviderCount(data, providerid):
    paid_count = data.loc[(data['Provider.ID'] == providerid)
                          & (data['PaidClaim'] == 1)]
    unpaid_count = data.loc[(data['Provider.ID'] == providerid)
                          & (data['UnpaidClaim'] == 1)]
    return providerid, len(paid_count), len(unpaid_count)