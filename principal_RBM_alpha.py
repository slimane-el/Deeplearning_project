import numpy as np
import pandas as pd
import scipy.io
import random
from math import *
import matplotlib.pyplot as plt
import torch
from torch import from_numpy as torchnp
from torch import bernoulli

global device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

dict_az = {
    '0': 0, '1': 1, '2': 2, '3': 3, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
    'A': 10, 'B': 11, 'C': 12, 'D': 13, 'E': 14, 'F': 15, 'G': 16, 'H': 17, 'I': 18,
    'J': 19, 'K': 20, 'L': 21, 'M': 22, 'N': 23, 'O': 24, 'P': 25, 'Q': 26, 'R': 27,
    'S': 28, 'T': 29, 'U': 30, 'V': 31, 'W': 32, 'X': 33, 'Y': 34, 'Z': 35
}

path_data = "data/binaryalphadigs.mat"


def lire_alpha_digit(caractere_list, path_data):
    caractere_list = [dict_az[el] for el in caractere_list]
    print(caractere_list)
    data = scipy.io.loadmat(path_data)
    data = list(data['dat'][i] for i in caractere_list)
    final_data = []
    # For each letter
    for i in range(len(caractere_list)):
        data_tmp = [data[i][j].flatten() for j in range(39)]
        final_data.append(np.vstack(data_tmp))
        if np.vstack(data_tmp).shape[0] != 39:
            raise ValueError

    p = final_data[0].shape[1]
    final_data = np.vstack(final_data)
    final_data = np.resize(
        final_data, (final_data.shape[0], 1, final_data.shape[1]))

    return final_data, p


class RBM:
    def __init__(self):
        pass

    def init_RBM(self, p, q):
        self.p = p  # Nombre de neurones d'entrée => doit être le même que taille des données, à vérifier lors de l'entraînement
        self.q = q  # hyperparameters
        self.b = torchnp(np.zeros((1, self.q))).to(device)
        self.a = torchnp(np.zeros((1, self.p))).to(device)
        self.W = torchnp(np.random.normal(
            0, 0.01, size=(self.p, self.q))).to(device)

    def sig(self, x):
        return 1/(1 + torch.exp(-x))

    def entree_sortie_RBM(self, donnees_entree, sig=True):
        if donnees_entree.shape[1] != 1:
            raise ValueError(
                "Erreur : 'donnees_entree' n'est pas de la bonne dimension (n, 1, p).")
        if sig is True:
            res = self.sig(donnees_entree @ self.W + self.b)
        else:
            res = donnees_entree @ self.W + self.b
        return res

    def sortie_entree_RBM(self, donnees_sortie):
        return self.sig(donnees_sortie @ torch.transpose(self.W, 0, 1) + self.a)

    def generer_image_RBM(self, iterations_gibbs, nb_images, show=False):
        # initialisation donnees v
        v = torchnp(np.random.random_sample((nb_images, 1, self.p))).to(device)
        v = torch.round(v)
        for i in range(iterations_gibbs):
            tirage_h = self.entree_sortie_RBM(v)  # (n, 1, q)
            h = bernoulli(tirage_h)
            # Tirage de v (taille p x 1) dans loi p(v|h^0)
            tirage_v = self.sortie_entree_RBM(h)  # (n, 1, p)
            v = bernoulli(tirage_v)
        list_img = []
        for img in range(nb_images):
            X = np.reshape(v[img].cpu().flatten(), (20, 16))
            if show is True:
                plt.figure()
                im = plt.imshow(X, cmap='Greys')
                plt.show()
            list_img.append(X)

        return list_img

    def train_RBM(self, epochs, lr, batch_fraction, x, plot_error=False):
        # vérifier type de x et le mette en torch
        if type(x) == np.ndarray:
            x = torchnp(x).to(device)
            x = x.double()
        x = x.to(device)
        print(f"Taille des données ====> {x.shape}")
        assert 0 < batch_fraction <= 1
        batch_size = max(1, int(np.floor(batch_fraction * x.shape[0])))
        print(f"Batch size ====> {batch_size}")
        # ////!\\\\ x doit être de la taille (n, 1, p)
        if self.p != x.shape[2]:
            raise ValueError(
                "p doit être égal à la taille d'une donnée d'entrée (nombre de pixels).")
        # à répéter pour chaque epoch
        eqm_list = []
        for e in range(epochs):
            # données d'entrees  ////!\\\\ (n, 1, p) n en première dimension !!!
            v_0 = x
            # Batch size
            shuffled_indices = np.arange(v_0.shape[0])
            np.random.shuffle(shuffled_indices)
            v_0_shuffled = v_0[shuffled_indices]
            num_batches = np.ceil(v_0.shape[0] / batch_size).astype(int)
            list_v_0 = [
                v_0_shuffled[i * batch_size:(i + 1) * batch_size] for i in range(num_batches)]
            for v_0 in list_v_0:
                # Forward v0 -> h0
                tirage_h = self.entree_sortie_RBM(v_0)  # (n, 1, q)
                # Tirage de h0 (taille q x 1) dans p(h|v), tirer q nombres entre 0 et 1 et comparer à h
                h = bernoulli(tirage_h)
                # Backward h0 -> v1
                tirage_v = self.sortie_entree_RBM(h)  # (n, 1, p)
                # Tirage de v1 (taille p x 1) dans loi p(v|h^0)
                v_1 = bernoulli(tirage_v)
                # Forward v1 -> h1
                tirage_h_1 = self.entree_sortie_RBM(v_1)  # (n, 1, q)

                # (p, 1, n) @ (1, q, n) => (p, q, n)
                delta_W = torch.transpose(
                    v_0, 1, 2) @ tirage_h - torch.transpose(v_1, 1, 2) @ tirage_h_1
                delta_a = v_0 - v_1  # (n, 1, p) #v_1
                delta_b = tirage_h - tirage_h_1  # (n ,1, q)

                delta_W = delta_W.mean(axis=0)
                delta_a = delta_a.mean(axis=0)
                delta_b = delta_b.mean(axis=0)

                self.W += lr * delta_W
                self.a += lr * delta_a
                self.b += lr * delta_b

            eqm = torch.abs((v_0 - v_1)).mean()
            eqm_list.append(eqm.cpu())
            if plot_error is True:
                print(f"============= Epoch n°{e} =============\n MAE = {eqm}")
        if plot_error is True:
            plt.figure()
            plt.plot(eqm_list)
            plt.ylabel('MAE')
            plt.xlabel('Epoch')
            plt.grid()
            plt.show()

# if __name__ == '__main__':
#     path_data = "data/binaryalphadigs.mat"
#     data, nb_pixels = lire_alpha_digit(["A"], path_data)
#     rbm = RBM()
#     rbm.init_RBM(
#         p = nb_pixels,
#         q = 200)
#     rbm.train_RBM(epochs=300, lr=0.1, batch_fraction=0.2, x=data, plot_error=True)
#     img = rbm.generer_image_RBM(iterations_gibbs=500, nb_images=1, show=True)
