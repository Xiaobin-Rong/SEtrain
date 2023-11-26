import numpy as np

def sisnr(esti, tagt):
    """ for single wav """
    esti = esti - esti.mean()
    tagt = tagt - tagt.mean()

    a = np.sum(esti * tagt) / np.sum(tagt**2 + 1e-8)
    e_tagt = a * tagt
    e_res = esti - e_tagt

    return 10*np.log10((np.sum(e_tagt**2)+1e-8) / (np.sum(e_res**2)+1e-8))




if __name__ == "__main__":
    x = np.random.randn(100)
    s = np.random.randn(100)
    print(sisnr(x, s))