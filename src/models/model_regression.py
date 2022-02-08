import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel


class BertForMultiDimensionRegression(BertPreTrainedModel):

    def __init__(self,
                 config,
                 loss_func=None,
                 target_dim=3,
                 hidden_layers_count=1,
                 hidden_layer_dim=400,
                 **kwargs):
        super().__init__(config)
        self.target_dim = target_dim
        self.hidden_layers_count = hidden_layers_count
        self.hidden_layers_dim = hidden_layer_dim

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.act_func = nn.Sigmoid  # final activation function
        self.final_act_func = nn.Sigmoid()  # final activation function

        if hidden_layers_count == 1:
            self.output_layer = nn.Linear(config.hidden_size, self.target_dim)
        else:
            # stack the hidden layers
            layers_lst = []
            layers_lst += [nn.Linear(config.hidden_size, self.hidden_layers_dim), self.act_func()]
            for _ in range(self.hidden_layers_count - 2):
                layers_lst += [nn.Linear(self.hidden_layers_dim, self.hidden_layers_dim), self.act_func()]
            layers_lst += [nn.Linear(self.hidden_layers_dim, self.target_dim)]
            self.output_layer = nn.Sequential(*layers_lst)

        # Sigmoid layer + Binary Cross Entropy Loss (will be for every VAD dimension)
        # https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html
        self.loss_func = nn.BCEWithLogitsLoss()

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            output_targets=None,
            vad=None,
            **kwargs
    ):
        """
            output_targets - torch.Tensor of shape (# input sentences, self.target_dim) (same as logits)
            vad - if vad argument is present, is overrides outputs_target
        """
        output_targets = output_targets if vad is None else vad

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        # outputs[0] : the whole last-hidden-layer of BERT (batch_size, sequence_length, hidden_size)
        # outputs[1] : [CLS] token in last hidden layer
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.output_layer(pooled_output)
        # logits = self.final_act_func(logits) # we give up on the activation function (dealt by the loss)

        outputs = (logits,) + outputs[2:]  # adds hidden states and attention if they are here

        if output_targets is not None:
            loss = self.loss_func(logits, output_targets)
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)
