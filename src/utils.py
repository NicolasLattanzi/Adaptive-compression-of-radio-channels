import numpy as np

classes = ['OOK','4ASK','8ASK','BPSK','QPSK','8PSK','16PSK','32PSK',
           '16APSK','32APSK','64APSK','128APSK','16QAM','32QAM','64QAM',
           '128QAM','256QAM','AM-SSB-WC','AM-SSB-SC','AM-DSB-WC','AM-DSB-SC',
           'FM','GMSK','OQPSK']


# modulation is one-hot encoded in Y. argmax returns the index of the 1
def get_modulation(encoded_mod):
    return classes[ np.argmax(encoded_mod) ]
    