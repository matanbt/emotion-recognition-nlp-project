import torch
import torch.nn as nn
from attrdict import AttrDict
from transformers import BertPreTrainedModel, BertModel


class BertForMultiDimensionRegressionAndClassification(BertPreTrainedModel):

    def __init__(self,
                 config,
                 loss_func: str = None,  # for regression
                 target_dim=3,
                 hidden_layers_count=1,
                 hidden_layer_dim=400,
                 pool_mode='mean',
                 lambda_param: float = 0.5,
                 args: AttrDict = None,
                 **kwargs):
        super().__init__(config)
        self.target_dim = target_dim
        self.hidden_layers_count = hidden_layers_count
        self.hidden_layers_dim = hidden_layer_dim
        self.pool_mode = pool_mode if pool_mode is not None else 'mean'

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.act_func = nn.Sigmoid  # hidden activation function
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
            'RMSE': lambda preds, targets : torch.sqrt(self._MSE(preds, targets)),
            'MAE': nn.L1Loss(),
            'EXP1': lambda input, target: torch.logsumexp((input - target).abs(), dim=-1).mean(),
            'EXP2': lambda input, target: (input - target).abs().exp().sum(dim=-1).mean(),
        }

        # Choose your loss here:
        self.loss_func_regr = losses['MAE'] if (loss_func is None) else losses[loss_func]

        labels_count = 28
        self.classifier = nn.Linear(config.hidden_size + self.target_dim, labels_count)
        # TODO-DUAL - add hidden layers?
        self.loss_func_classifier = nn.BCEWithLogitsLoss()
        self.lambda_param = lambda_param # regression weight in loss

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
            one_hot_labels=None,
            vad=None,
            **kwargs
    ):
        """
            output_targets - torch.Tensor of shape (# input sentences, self.target_dim) (same as logits)
            vad - if vad argument is present, is overrides outputs_target
        """
        output_targets = output_targets if vad is None else vad
        global_step = kwargs.get('global_step')

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

        # Regression Phase
        pooled_output = self.dropout(pooled_output)
        logits_regr = self.output_layer(pooled_output)
        logits_regr = self.final_act_func(logits_regr)

        # Classification Phase (fed with the regression)
        class_logits = self.classifier(torch.cat([pooled_output, logits_regr], dim=-1))

        outputs = (class_logits,) + outputs[2:]  # adds hidden states and attention if they are here

        if output_targets is not None:
            loss_regr = self.loss_func_regr(logits_regr, output_targets)
            loss_class = self.loss_func_classifier(class_logits, one_hot_labels)
            if global_step is not None and global_step <= 18000:
                 loss = loss_regr
            else:
                 # self.final_act_func(logits_regr).requires_grad_(False)
                 loss = loss_class * 0.9 + loss_regr * 0.1
            # TODO-DUAL - it's possible that scaling is needed here:
            #loss = loss_regr * self.lambda_param + loss_class * (1 - self.lambda_param)
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)
