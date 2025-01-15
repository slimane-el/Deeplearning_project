from principal_DBN_alpha import *


class DNN(DBN):
    def __init__(self):
        super().__init__()

    def init_DNN(self, Q, nb_classes):
        self.init_DBN(Q=Q)
        # couche de classification supplémentaire
        self.list_RBM.append(RBM())
        # q est le nombre de classes
        self.list_RBM[-1].init_RBM(p=Q[-1], q=nb_classes)

    def pretrain_DNN(self, epochs, lr, batch_fraction, x, plot_error=False):
        # //!\\ pretrain_DBN ne doit train que les n-1 premières couches
        self.train_DBN(epochs=epochs, lr=lr, batch_fraction=batch_fraction, x=x,
                       layers=-1, plot_error=plot_error)

    def calcul_softmax(self, z):
        return torch.exp(z) / (torch.exp(z).sum(axis=2, keepdims=True))

    def entree_sortie_reseau(self, donnees_entree):
        sortie = []
        if type(donnees_entree) == np.ndarray:
            donnees_entree = torchnp(donnees_entree).double().to(device)
        for rbm in self.list_RBM[:-1]:
            donnees_entree = rbm.entree_sortie_RBM(donnees_entree)
            sortie.append(donnees_entree)
            donnees_entree = bernoulli(donnees_entree)
        rbm = self.list_RBM[-1]
        donnees_entree = rbm.entree_sortie_RBM(donnees_entree, sig=False)
        proba = self.calcul_softmax(donnees_entree)
        sortie.append(proba)
        return sortie

    def retropropagation(self, epochs, lr, batch_fraction, X, Y, plot_error=False):
        if type(X) == np.ndarray:
            X = torchnp(X).double().to(device)
        if type(Y) == np.ndarray:
            Y = torchnp(Y).double().to(device)
        print(f"Taille des données ====> {X.shape}")
        assert 0 < batch_fraction <= 1
        batch_size = max(1, int(np.floor(batch_fraction * X.shape[0])))
        print(f"Batch size ====> {batch_size}")
        # initialisation des poids de la dernière couche
        couche = self.list_RBM[-1]
        couche.W = np.random.randn(
            int(couche.W.shape[0]), int(couche.W.shape[1]))*0.01
        couche.W = torchnp(couche.W).double().to(device)
        couche.b = np.zeros((int(couche.b.shape[0]), int(couche.b.shape[1])))
        couche.b = torchnp(couche.b).double().to(device)
        loss_list = []
        for e in range(epochs):
            if e % 10 == 0 and e != 0:
                print(
                    f"============= Epoch n°{e} =============\n loss = {loss}")
            # Mini-batches
            shuffled_indices = np.arange(X.shape[0])
            np.random.shuffle(shuffled_indices)
            X_shuffled = X[shuffled_indices]
            Y_shuffled = Y[shuffled_indices]
            num_batches = np.ceil(X.shape[0] / batch_size).astype(int)
            X_batches = [
                X_shuffled[i * batch_size:(i + 1) * batch_size] for i in range(num_batches)]
            Y_batches = [
                Y_shuffled[i * batch_size:(i + 1) * batch_size] for i in range(num_batches)]
            for x, y in zip(X_batches, Y_batches):
                # forward
                sortie = self.entree_sortie_reseau(x)
                proba = sortie[-1]
                loss = -(y * torch.log(proba)).sum()  # /(x.shape[0])
                loss_list.append(loss.cpu())
                # backpropagation sur TOUTES les couches du réseau
                for i in reversed(range(self.nb_couche)):
                    couche = self.list_RBM[i]
                    # calcul du gradient
                    if i == self.nb_couche - 1:
                        delta_W = torch.transpose(
                            sortie[i-1], 1, 2) @ (proba - y)
                        delta_b = proba - y
                    elif i != 0:
                        delta_b = (
                            delta_b @ self.list_RBM[i+1].W.T) * (sortie[i] * (1 - sortie[i]))
                        delta_W = torch.transpose(sortie[i-1], 1, 2) @ delta_b
                    else:
                        delta_b = (
                            delta_b @ self.list_RBM[i+1].W.T) * (sortie[i] * (1 - sortie[i]))
                        delta_W = torch.transpose(x, 1, 2) @ delta_b
                    delta_W_mean = delta_W.mean(axis=0)
                    delta_b_mean = delta_b.mean(axis=0)
                    # mise à jour des poids
                    couche.W -= lr * delta_W_mean
                    couche.b -= lr * delta_b_mean
        if plot_error is True:
            plt.figure()
            plt.plot(loss_list)
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.show()
        return 0

    def test_DNN(self, x, y):
        sortie = self.entree_sortie_reseau(x)
        proba = sortie[-1].cpu()
        y_pred = torch.argmax(proba, axis=2).cpu()
        y_true = torch.argmax(y, axis=2).cpu()
        taux_erreur = (y_pred != y_true).sum()/len(y_true)
        print(f"Taux d'erreur = {taux_erreur}")
        return taux_erreur, proba


# if __name__ == '__main__':
#     # Test DNN sur 2 lettres de Binary Alpha Digits
#     path_data = "data/binaryalphadigs.mat"
#     data, nb_pixels = lire_alpha_digit(["A", "B"], path_data)
#     dnn = DNN()
#     Q = [nb_pixels, 200, 200]
#     dnn.init_DNN(
#         Q=Q,
#         nb_classes=2
#     )
#     dnn.pretrain_DNN(epochs=[100], lr=0.1,
#                      batch_fraction=0.2, x=data, plot_error=True)
#     # y shape shoulde be (n, 1, 2) where n is the number of images
#     # test y shape
#     y1 = np.array([[[1, 0]]]*int(data.shape[0]/2))
#     y2 = np.array([[[0, 1]]]*int(data.shape[0]/2))
#     y = np.concatenate((y1, y2), axis=0)
#     y = torchnp(y).double().to(device)
#     if y.shape != (data.shape[0], 1, 2):
#         raise ValueError(
#             "y shape should be (n, 1, 2) where n is the number of images")
#     sortie = dnn.retropropagation(epochs=300, lr=0.1, batch_fraction=0.2, X=data,
#                                   Y=y, plot_error=True)
#     # Données de test
#     tau, proba = dnn.test_DNN(x=data, y=y)
#     print(f"Taux d'erreur = {tau}")
#     print(proba)
