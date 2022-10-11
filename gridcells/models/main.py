import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F


class WorstModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(203, 100)
        self.fc2 = nn.Linear(100, 100)
        self.out_place_cells = nn.Linear(100, 200)
        self.out_head_cells = nn.Linear(100, 100)

    def forward(self, init_pos: Tensor, init_hd: Tensor, ego_vel: Tensor) -> tuple[Tensor, Tensor]:
        # Concatenate everything into a single vector
        head_rotation_speed = torch.arctan2(ego_vel[..., 1], ego_vel[..., 2])
        forward_speed = ego_vel[..., 0]
        x = torch.hstack([head_rotation_speed, forward_speed, init_hd, init_pos])

        # Run it through a MLP
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # Decode the output into two sets of "cells"
        # Expected shape for position
        place_cells = self.out_place_cells(x)
        place_cells = place_cells.view(-1, 100, 2)

        # Expected shape for head direction
        head_cells = self.out_head_cells(x)
        head_cells = head_cells.view(-1, 100, 1)
        return place_cells, head_cells


class DeepMindModel(nn.Module):
    def __init__(
        self,
        lstm_hidden_size: int = 128,
        position_encoding_size: int = 256,
        head_encoding_size: int = 12,
        bottleneck_size: int = 256,
        use_dropout: bool = True,
        weight_decay: float = 1e-5,
    ):
        super().__init__()

        self.weight_decay = weight_decay
        self.use_dropout = use_dropout
        self.head_encoding_size = head_encoding_size
        self.position_encoding_size = position_encoding_size

        inputs_size = position_encoding_size + head_encoding_size
        self.l1 = nn.Linear(inputs_size, lstm_hidden_size)
        self.l2 = nn.Linear(inputs_size, lstm_hidden_size)

        self.rnn = nn.LSTMCell(input_size=3, hidden_size=lstm_hidden_size)
        self.bottleneck_layer = nn.Linear(lstm_hidden_size, bottleneck_size, bias=False)
        self.pc_logits = nn.Linear(bottleneck_size, position_encoding_size)
        self.hd_logits = nn.Linear(bottleneck_size, head_encoding_size)

    @property
    def number_of_parameters(self) -> int:
        how_many = sum(p.numel() for p in self.parameters())
        return how_many

    def forward(self, concat_init, ego_vel):
        init_lstm_cell = self.l1(concat_init)
        init_lstm_state = self.l2(concat_init)
        (hx, cx) = (init_lstm_state, init_lstm_cell)

        lstm_inputs = ego_vel.transpose(0, 1)

        bottlenecks = []
        predicted_positions = []
        predicted_hd = []
        for lstm_input in lstm_inputs:
            hx, cx = self.rnn(lstm_input, (hx, cx))

            bottleneck = self.bottleneck_layer(hx)
            if self.use_dropout:
                bottleneck = nn.functional.dropout(bottleneck, 0.5)
            bottlenecks.append(bottleneck)

            predicted_position = self.pc_logits(bottleneck)
            predicted_positions.append(predicted_position)

            predicted_head_direction = self.hd_logits(bottleneck)
            predicted_hd.append(predicted_head_direction)

        bottlenecks = torch.stack(bottlenecks, dim=1)
        predicted_positions = torch.stack(predicted_positions, dim=1)
        predicted_hd = torch.stack(predicted_hd, dim=1)

        return predicted_positions, predicted_hd, bottlenecks

    @property
    def l2_bottleneck(self):
        loss = self.bottleneck_layer.weight.norm(2) + self.pc_logits.weight.norm(2) + self.hd_logits.weight.norm(2)
        return loss

    def regularization(self) -> torch.Tensor:
        loss = self.weight_decay * self.l2_bottleneck
        return loss
    
    

    
class GanguliRNN(torch.nn.Module):
    def __init__(self, options, place_cells):
        super(RNN, self).__init__()
        self.Ng = 4096
        self.Np = 512
        self.sequence_length = 20
        self.weight_decay = 1e-4
        self.place_cells = place_cells

        # Input weights
        self.encoder = torch.nn.Linear(self.Np, self.Ng, bias=False)
        self.RNN = torch.nn.RNN(input_size=2,
                                hidden_size=self.Ng,
                                nonlinearity=options.activation,
                                bias=False)
        # Linear read-out weights
        self.decoder = torch.nn.Linear(self.Ng, self.Np, bias=False)
        
        self.softmax = torch.nn.Softmax(dim=-1)

    def g(self, inputs):
        '''
        Compute grid cell activations.
        Args:
            inputs: Batch of 2d velocity inputs with shape [batch_size, sequence_length, 2].

        Returns: 
            g: Batch of grid cell activations with shape [batch_size, sequence_length, Ng].
        '''
        v, p0 = inputs
        init_state = self.encoder(p0)[None]
        g,_ = self.RNN(v, init_state)
        return g
    

    def predict(self, inputs):
        '''
        Predict place cell code.
        Args:
            inputs: Batch of 2d velocity inputs with shape [batch_size, sequence_length, 2].

        Returns: 
            place_preds: Predicted place cell activations with shape 
                [batch_size, sequence_length, Np].
        '''
        place_preds = self.decoder(self.g(inputs))
        
        return place_preds


    def compute_loss(self, inputs, pc_outputs, pos):
        '''
        Compute avg. loss and decoding error.
        Args:
            inputs: Batch of 2d velocity inputs with shape [batch_size, sequence_length, 2].
            pc_outputs: Ground truth place cell activations with shape 
                [batch_size, sequence_length, Np].
            pos: Ground truth 2d position with shape [batch_size, sequence_length, 2].

        Returns:
            loss: Avg. loss for this training batch.
            err: Avg. decoded position error in cm.
        '''
        y = pc_outputs
        preds = self.predict(inputs)
        yhat = self.softmax(self.predict(inputs))
        loss = -(y*torch.log(yhat)).sum(-1).mean()

        # Weight regularization 
        loss += self.weight_decay * (self.RNN.weight_hh_l0**2).sum()

        # Compute decoding error
        pred_pos = self.place_cells.get_nearest_cell_pos(preds)
        err = torch.sqrt(((pos - pred_pos)**2).sum(-1)).mean()

        return loss, err