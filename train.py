import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import math
import argparse
import xlsxwriter

torch.set_default_dtype(torch.float64)

class DEDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, scale_data=False):
        if not torch.is_tensor(X) and not torch.is_tensor(y):
            #Apply scaling if necessary
            if scale_data:
                X = StandardScaler().fit_transform(X)
            self.X = torch.from_numpy(X)
            self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(25, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.layers(x)

  
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_number", default=1, type=int,
                        help="exp number")
    parser.add_argument("--val_number", default=1, type=int,
                        help="val number")
    parser.add_argument("--val_count", default=1, type=int,
                        help="val sample count number")
    parser.add_argument("--sample_name", type=str, default='Sample1')
    parser.add_argument("--train", action="store_true", 
                        help="run training")
    parser.add_argument("--test", action="store_true", 
                        help="run testing")

    args = parser.parse_args()

    # Initialize the MLP
    mlp = MLP()

    if args.train:
        torch.manual_seed(42)
    
        # Load dataset
        DE_list = ['DE0', 'DE50', 'DE100']
        xls = pd.ExcelFile('./train_' + str(args.exp_number)+ '/train.xlsx')

        X = []
        y = []

        for de in DE_list:
            azimuth = []
            df1 = pd.read_excel(xls, de)

            tilt_angle_1 = [i*15 for i in range(12)]
            tilt_angel_2 = [i*(-15) for i in range(1, 13)]

            tilt_angle = tilt_angle_1 + tilt_angel_2

            for i in tilt_angle:
                azimuth.append(df1[i].values)

            tilt_angel_2 = [(i*(-15) + 360) for i in range(1, 13)]
            tilt_angle = tilt_angle_1 + tilt_angel_2

            for i in range(len(tilt_angle)):
                tilt_angle[i] = [((tilt_angle[i] * math.pi) / 180.0), float(de[2:]) / 100.0]

            X = X + azimuth
            y = y + tilt_angle


        X = np.array(X)
        y = np.array(y)
        y = y.astype(float)
    
        # Prepare dataset
        dataset = DEDataset(X, y)
        trainloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
        # Define the loss function and optimizer
        loss_function = nn.L1Loss()
        optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-3)

        best_loss = 100.0
        # Run the training loop
        for epoch in range(0, 15000): # 5 epochs at maximum
        
            # Set current loss value
            current_loss = 0.0
        
            # Iterate over the DataLoader for training data
            for i, data in enumerate(trainloader):
        
            # Get and prepare inputs
                inputs, targets = data
                inputs, targets = inputs.double(), targets.double()
                targets = targets.reshape((targets.shape[0], 2))
        
                # Perform forward pass
                outputs = mlp(inputs)
        
                # Compute loss
                loss = loss_function(outputs, targets)
        
                # Perform backward pass
                loss.backward()
        
                # Perform optimization
                optimizer.step()

                # Zero the gradients
                optimizer.zero_grad()

                current_loss += loss.item()
                
            if epoch % 100 == 0:
                cur_loss = (current_loss / (100 * len(trainloader)))
                print('Loss after epoch %5d: %.12f' % (epoch, cur_loss))
                if best_loss > cur_loss:
                    best_loss = cur_loss
                    torch.save(mlp.state_dict(), './train_' + str(args.exp_number)+ '/best_checkpoint.bin')

            current_loss = 0.0

        # Process is complete.
        print('Training process has finished.')

    if args.test:
        xls = pd.ExcelFile('./train_' + str(args.exp_number)+ '/val' + str(args.val_number)+ '.xlsx')
        df2 = pd.read_excel(xls, args.sample_name)

        sample_count = args.val_count
        y = [0 for i in range(sample_count)]
        X = []

        for i in range(sample_count):
            X.append(df2[i].values)

        X = np.array(X)
        y = np.array(y)
        y = y.astype(float)

        dataset = DEDataset(X, y)
        testloader = DataLoader(dataset, batch_size=args.val_count, shuffle=False)

        mlp.load_state_dict(torch.load('./train_' + str(args.exp_number)+ '/best_checkpoint.bin'))
        mlp.eval()

        workbook = xlsxwriter.Workbook('train_' + str(args.exp_number) + '_' + args.sample_name + '.xlsx')
        worksheet = workbook.add_worksheet(args.sample_name)
        row = 0
        col = 0

        for i, data in enumerate(testloader):
            # Get and prepare inputs
            inputs, targets = data
            inputs, targets = inputs.double(), targets.double()
            targets = targets.reshape((targets.shape[0], 1))
        
            # Perform forward pass
            outputs = mlp(inputs)
            print(inputs)
            angles = (outputs[:, :1] * 180.0) / math.pi # angle
            des = (outputs[:, 1:2]) * 100.0             # DE

            for i, tup in enumerate(zip(angles, des)):
                print(tup[0].item(), tup[1].item())
                worksheet.write(row, col, tup[0].item())
                worksheet.write(row, col + 1, tup[1].item())
                row += 1

        workbook.close()
