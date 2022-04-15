import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel


class BertForMultiDimensionRegressionPenalty(BertPreTrainedModel):

    def __init__(self,
                 config,
                 loss_func=None,
                 target_dim=3,
                 hidden_layers_count=1,
                 hidden_layer_dim=400,
                 pool_mode='cls',
                 **kwargs):
        super().__init__(config)
        self.target_dim = target_dim
        self.hidden_layers_count = hidden_layers_count
        self.hidden_layers_dim = hidden_layer_dim
        self.pool_mode = pool_mode
        self.emotions_vads_lst = None  # will be fulfilled

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

        # Loss choices (comment-out unused losses before experimenting):
        losses = {
            'MSE': nn.MSELoss(),
            'MAE': nn.L1Loss(),
            'EXP1': lambda input, target: torch.logsumexp((input - target).abs(), dim=-1).mean(),
            'EXP2': lambda input, target: (input - target).abs().exp().sum(dim=-1).mean(),
        }

        # Choose your loss here:
        self.loss_func = losses['MAE'] if (loss_func is None) else losses[loss_func]
        self.ce = torch.nn.CrossEntropyLoss()  # penalty
        self.lambda_param = kwargs.get('lambda_param', 0.1)  # weight of the CE

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
        one_hot_labels = kwargs.get('one_hot_labels')

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        # outputs.last_hidden_state : the whole last-hidden-layer of BERT (batch_size, sequence_length, hidden_size)
        # outputs.pooler_output : [CLS] token in last hidden layer
        pooled_output = None

        if self.pool_mode == 'cls':
            pooled_output = outputs.pooler_output
        elif self.pool_mode == 'last':
            pooled_output = outputs.last_hidden_state[:, -1, :]
        elif self.pool_mode == 'sum':
            pooled_output = outputs.last_hidden_state.sum(axis=1)
        elif self.pool_mode == 'mean':
            pooled_output = outputs.last_hidden_state.mean(axis=1)
        else:
            raise ValueError(f"Given pool_mode `{self.pool_mode}` is invalid.")

        pooled_output = self.dropout(pooled_output)
        logits = self.output_layer(pooled_output)
        logits = self.final_act_func(logits)

        outputs = (logits,) + outputs[2:]  # adds hidden states and attention if they are here

        # getting VAD-space-distance from predictions to target-labels
        regr_targets = self.emotions_vads_lst.unsqueeze(dim=0).to(
            logits.device)  # VAD targets [28,3] --> [1,28,3]
        regr_logits = logits.unsqueeze(dim=1)  # logits [N,3] --> [N,1,3]
        dist = torch.sub(regr_logits, regr_targets)  # broadcasted substraction
        class_logits = torch.linalg.vector_norm(dist.float(), dim=-1,ord=1) * -1  # [N, 28] # we change the sign, because the smaller the value the better much it is, the higher score # manhattan distane

        # ---
        class_targets = one_hot_labels.argmax(dim=-1)

        if output_targets is not None:
            loss = self.loss_func(logits, output_targets)
            loss += self.ce(class_logits, class_targets) * self.lambda_param

            outputs = (loss,) + outputs


        return outputs  # (loss), logits, (hidden_states), (attentions)
