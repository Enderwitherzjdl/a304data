# a304data/qbanalyzer/qbtheory.py
import numpy as np
import matplotlib.pyplot as plt

class QBTheory:
    def gs_ho_ef(aq=0.02, S=0.45, gamma=5, weg=3, w0=1, cutoff=10, p=1,w=np.linspace(-20,20,1001)):
        """
        基于等频率(ef)谐振子(ho)计算基态(gs)量子拍强度谱。
        
        Args:
            aq (float): α*q0, α = dμ/dq, q0 = sqrt{hbar/(m_eff * ω0)}.
            S (float): Huang-Rhys 因子.
            gamma (float): 退相干速率/激发态寿命倒数 γ。
            weg (float): 跃迁频率 ω_eg.
            w0 (float): 振动频率 ω0, ω_nm = ω_eg + (n-m) * ω0.
            cutoff (int): 求和的截断，运算涉及阶乘不要设太大。
            p (int): 基频 p=1, 泛频 p=2。
            w (np.NDArray): 波长范围。
        
        Returns:
            M (np.NDArray): 计算的量子拍谱。
        """

        # 计算 A_{n,m}(S)
        
        fac = np.ones(cutoff)
        for i in range(1,cutoff):
            fac[i] = fac[i-1]*i
        A = np.zeros((cutoff,cutoff))
        for n in range(cutoff):
            for m in range(cutoff):
                for j in range(min(n,m)+1):
                    A[n,m] += (-1)**j * S**(-j) / (fac[j] * fac[n-j] * fac[m-j])

        # 计算 c_{n,m}
        C = np.zeros((cutoff,cutoff))
        for n in range(cutoff):
            for m in range(cutoff):
                C[n,m] = A[n,m] * (-1)**n * np.exp(-S/2) * np.sqrt(fac[n]*fac[m]*(S**(n+m)))
                
        D = np.zeros((cutoff,cutoff))
        for n in range(cutoff):
            for m in range(cutoff):
                D[n,m] = S * A[n,m]
                if m > 0:
                    D[n,m] += A[n,m-1]
                if n > 0:
                    D[n,m] -= A[n-1,m]
                D[n,m] *= aq * (-1)**n * np.exp(-S/2) * np.sqrt(fac[n]*fac[m]*(S**(n+m-1))/2)
                # 这里的 D[n,m] = alpha*d_{n,m}，合并了alpha和q0

        M = np.zeros(len(w))
        coeff = 0
        for n in range(cutoff):
            coeff += C[n,p]*D[n,0] + D[n,p]*C[n,0]
            w1 = weg + n*w0
            w2 = weg + (n-p)*w0
            M += (C[n,0]+D[n,0]) * (C[n,p]+D[n,p]) * (1/((w-w1)**2+(gamma**2)/4) + 1/((w-w2)**2+(gamma**2)/4))
        M *= coeff * (-gamma/2)
        return M
    
    # 生成等频率harmonic的ES-QB谱
    def es_ho_ef(S=0.45, gamma=5, weg=3, w0=1, cutoff=10, p=1,w=np.linspace(-20,20,1001)):
        """
        基于等频率(ef)谐振子(ho)计算激发态(es)量子拍强度谱。
        
        Args:
            S (float): Huang-Rhys 因子.
            gamma (float): 退相干速率/激发态寿命倒数 γ。
            weg (float): 跃迁频率 ω_eg.
            w0 (float): 振动频率 ω0, ω_nm = ω_eg + (n-m) * ω0.
            cutoff (int): 求和的截断，运算涉及阶乘不要设太大。
            p (int): 基频 p=1, 泛频 p=2。
            w (np.NDArray): 波长范围。
        
        Returns:
            M (np.NDArray): 计算的量子拍谱。
        """
        fac = np.ones(cutoff)
        for i in range(1,cutoff):
            fac[i] = fac[i-1]*i
        A = np.zeros((cutoff,cutoff))
        for n in range(cutoff):
            for m in range(cutoff):
                for j in range(min(n,m)+1):
                    A[n,m] += (-1)**j * S**(-j) / (fac[j] * fac[n-j] * fac[m-j])

        M = np.zeros(len(w))
        for n in range(cutoff):
            for m in range(cutoff): 
                if n+p < cutoff:
                    w1 = weg + (n+p-m)*w0
                    w2 = weg + (n-m)*w0
                    M += fac[m] * S**(2*n + m + p) * A[n,m] * A[n+p,m] * (1/((w-w1)**2+(gamma**2)/4) + 1/((w-w2)**2+(gamma**2)/4))
        M *= np.exp(-2*S) * (-gamma/2)
        return M