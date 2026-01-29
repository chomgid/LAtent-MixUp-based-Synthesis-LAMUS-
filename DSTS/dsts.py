import numpy as np
from DSTS.calibration2 import *
from DSTS.mixup import *
from sklearn.decomposition import PCA

class dsts:
    def __init__(self, sort, centering, pca_mixup):
        self.sort = sort
        self.center = centering
        self.pca_mixup = pca_mixup

    def fit(self, data):            
        try:
            data = np.array(data)
        except:
            raise ValueError("Data cannot be converted to numpy ndarray")
        if data.ndim != 3:
            print("Data dimension: ", data.ndim)
            raise ValueError("Data dimension needs to be 3 (n, l, c).")
        
        # Check for nans
        self.data = self.__test(data)
        c = data.shape[2]
        

        # Centering
        if self.center == 'sample_wise':
            self.sample_means = []
            sample_means = np.mean(self.data, axis=1, keepdims=True)  # shape: (n, 1, c)
            data_centered = self.data - sample_means

        elif self.center == 'feature_wise':
            self.feature_means = []
            feature_means = np.mean(self.data, axis=0, keepdims=True)  # shape: (1, l, c)
            data_centered = self.data - feature_means

        elif self.center == 'double':
            self.sample_means = []
            self.feature_means = []
            sample_means = np.mean(self.data, axis=1, keepdims=True)
            data_c_centered = self.data - sample_means
            feature_means = np.mean(data_c_centered, axis=0, keepdims=True)
            data_centered = data_c_centered - feature_means

        self.data_c = data_centered

        self.Zqs = []
        self.Wqs = []
        self.qs = []      

        for i in range(c):
            X = data_centered[:, :, i]  # shape (n, l)
            pca = PCA()
            pca.fit(X)

            # find q: number of components to explain ≥90%
            cum_var = np.cumsum(pca.explained_variance_ratio_)
            q = max(np.searchsorted(cum_var, 0.99) + 1, 2)

            print(f"Channel {i+1}: Number of q: {q}, Explained variance: {cum_var[q-1]:.4f}")

            # get top-q eigenvectors and eigenvalues
            Wq = pca.components_[:q, :]             # shape (q, l)
            Zq = pca.transform(X)[:, :q]            # shape (n, q)

            # Save data
            self.Wqs.append(Wq)
            self.Zqs.append(Zq)
            self.qs.append(q)

            if self.center == 'sample_wise':
                self.sample_means.append(sample_means[:,:,i])       # (n, 1)
            elif self.center == 'feature_wise':
                self.feature_means.append(feature_means[:,:,i])     # (1, l)
            elif self.center == 'double':
                self.sample_means.append(sample_means[:,:,i])
                self.feature_means.append(feature_means[:,:,i])
        
    def generate(self, tot_iter=5, aug=5) -> np.ndarray:
        k=5
            
        if self.pca_mixup:
            print("Start PCA space mixup")
            if self.center in ['sample_wise', 'double']:
                mixup_Z, mixup_mean = prin_mixup(self.Zqs, self.sample_means, k, alpha=0.5, beta=0.5)
            else: 
                mixup_Z, _ = prin_mixup(self.Zqs, None, k, alpha=0.5, beta=0.5)

            c = len(self.Zqs)

            recon_channels = []
            for i in range(c):
                Wq = self.Wqs[i]                               # (q_c, l)
                mixZ = mixup_Z[i]                              # (n*k, q_c)
                X_recon = mixZ.dot(Wq)                         # (n*k, l)
                # sample wise mean recovery
                if self.center in ('sample_wise', 'double'):
                    mixm = mixup_mean[i]                       # (n*k, 1)
                    X_recon = X_recon + mixm
                # feature wise mean recovery
                if self.center in ('feature_wise', 'double'):
                    mean_f = self.feature_means[i]             # (1, l)
                    X_recon = X_recon + mean_f
                recon_channels.append(X_recon)

            synth = np.stack(recon_channels, axis=2)

        else:
            print("Start neighbor finding in PCA space")
            mixup_data = mixup(self.data, self.Zqs, k, alpha=0.5, beta=0.5)    
            synth = mixup_data

        print("Start Calibration")
        calib_data = calibration(self.data, synth, tot_iter, aug)
        final_data = calib_data

        return final_data
       

    def __test(self, data):
        # Check if data contains any NaN values
        if np.isnan(data).any():
            raise ValueError("Your data must not contain any NaN values.", flush=True)

        return data
        
