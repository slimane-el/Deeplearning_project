from principal_RBM_alpha import *


class DBN(RBM):
    def __init__(self):
        super().__init__()

    def init_DBN(self, Q):
        # liste des nombres de neurones par couche ex: [100, 50, 20, 10, 5, 2]
        self.Q = Q
        self.nb_couche = len(self.Q)  # nombre de couches
        self.list_RBM = []
        for i in range(self.nb_couche - 1):
            rbm = RBM()
            # construction rbm de nb neurones sortie Q[i]
            rbm.init_RBM(
                p=Q[i],
                q=Q[i+1],
            )
            self.list_RBM.append(rbm)

    def train_DBN(self, epochs, lr, batch_fraction, x, layers=None, plot_error=False):
        if layers is None:
            layers = self.nb_couche
        if layers > self.nb_couche:
            raise ValueError(
                "Nombre de couches à entrainer supérieur au nombre de couches du réseau.")
        i = 0
        if type(x) == np.ndarray:
            x = torchnp(x).double().to(device)
        if len(epochs) != self.nb_couche - 1:
            epochs = [epochs[0]] * (self.nb_couche - 1)
        for rbm in self.list_RBM[0:layers]:
            i += 1
            print(f"====================== RBM n°{i} ======================\n")
            rbm.train_RBM(
                epochs=epochs[i-1], lr=lr, batch_fraction=batch_fraction, x=x, plot_error=plot_error)
            x = rbm.entree_sortie_RBM(donnees_entree=x)

    def generer_image_DBN(self, iterations_gibbs, nb_images, show=False):
        # initialisation donnees v
        v = torchnp(np.random.random_sample(
            (nb_images, 1, self.Q[0]))).to(device)
        v = torch.round(v)
        for i in range(iterations_gibbs):
            for rbm in self.list_RBM:
                tirage_h = rbm.entree_sortie_RBM(v)  # (n, 1, q)
                h = bernoulli(tirage_h)
                v = h
            for rbm in reversed(self.list_RBM):
                # Tirage de v (taille p x 1) dans loi p(v|h^0)
                tirage_v = rbm.sortie_entree_RBM(v)  # (n, 1, p)
                v = bernoulli(tirage_v)
            v = bernoulli(v)
        list_img = []
        for img in range(nb_images):
            X = np.reshape(v[img].cpu().flatten(), (20, 16))
            list_img.append(X)
            if show is True:
                plt.figure()
                im = plt.imshow(X, cmap='Greys')
                plt.show()

        return list_img


# if __name__ == '__main__':
#     path_data = "data/binaryalphadigs.mat"
#     data, nb_pixels = lire_alpha_digit(["A"], path_data)
#     dbn = DBN()
#     Q = [nb_pixels, 200, 200]
#     dbn.init_DBN(
#         Q = Q
#         )
#     dbn.train_DBN(epochs=[300], lr=0.1, batch_fraction=0.2, x=data, plot_error=True)
#     img = dbn.generer_image_DBN(500, 1)
