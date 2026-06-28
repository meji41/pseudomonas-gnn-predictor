import pandas as pd

class Bit(object):
    
    def __init__(self, onebitdict={'idx': None,
                                   'name': None,
                                   'on': None,
                                   'patt': None,
                                   'smarts':None,
                                   'info': None,
                                   'descr': None,
                                   'atomInUse': None,
                                   'bondInUse': None
                                   }):
        self.__dict__.update(onebitdict)

    
    
class FP(object):

    def bits2df(self, bits):
        infos = []
        for bit in bits:
            onebitdict={'idx': bit.idx,
                       'name': bit.name,
                       'on': bit.on,
                       'patt': bit.patt,
                       'smarts':bit.smarts,
                       'info': bit.info,
                       'descr': bit.descr,
                       'atomInUse': bit.atomInUse,
                       'bondInUse': bit.bondInUse}    
            infos.append(onebitdict)
        dfbit = pd.DataFrame(infos)
        return dfbit