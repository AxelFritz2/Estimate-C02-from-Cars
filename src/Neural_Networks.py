############################################# Imports #################################################

## Neural Networks' Packages
import torch
from torch import nn

## Data Preparation
import data_preparation as Clean
import time

############################################# Model #################################################

class Model_NN(nn.Module):
    def __init__(self, train, test):
        super().__init__()
        self.train = train
        self.test = test

        self.dataprep = Clean.DataPreparation(self.train, self.test, neural_networks=True, target = 'Ewltp_(g/km)')
        self.X_train, self.y_train, self.X_val, self.y_val = self.dataprep.prepare_data()

        self.input_dim = self.X_train.shape[1]

        self.h1_dim = 128
        self.h2_dim = 256
        self.h3_dim = 256
        self.h4_dim = 128
        self.output = 1

        #layers
        self.tab_2hid1 = nn.Linear(self.input_dim, self.h1_dim)
        self.hid1_2hid2 = nn.Linear(self.h1_dim, self.h2_dim)
        self.hid2_2hid3 = nn.Linear(self.h2_dim, self.h3_dim)
        self.hid3_2hid4 = nn.Linear(self.h3_dim, self.h4_dim)
        self.hid4_2output = nn.Linear(self.h4_dim, self.output)

        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Run the whole model.

        Args :
            - x (torch) : data to synthesize

        Return :
            tuple :
                - x_reconstructed : synthesized data
                - mu and sigma : distribution parameter

        """

        h = self.relu(self.tab_2hid1(x))
        h = self.relu(self.hid1_2hid2(h))
        h = self.relu(self.hid2_2hid3(h))
        h = self.relu(self.hid3_2hid4(h))
        output_predicted = self.relu(self.hid4_2output(h))
        return output_predicted

    def fit(self, num_epochs=300, LR_rate=1e-3, display=False):
        """
        This function trains the model.

        Args :
            - num_epochs (int) : number of times the model will be trained.
            - LR_rate (float64) : Learning rate.
            - display (bool) : display or not the information during the training process.
        """
        print("Entraînement débuté")
        start_time = time.time()
        device = torch.device('cpu')

        optimizer = torch.optim.AdamW(self.parameters(), lr=LR_rate)
        MAE_loss = nn.L1Loss()

        for epoch in range(num_epochs):
            for x, y in zip(self.dataprep.train_dataloader, self.dataprep.target_dataloader) :
                x.to(device), y.to(device)
                output = self.forward(x)

                # loss
                loss = MAE_loss(output, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if display and epoch % 5 == 0:
                    print(f"Epoch {epoch} terminée, loss = {loss}")

        print("Entraînement terminé")
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Temps total de l'entraînement : {round(elapsed_time,2)} secondes")



