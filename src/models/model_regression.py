import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel

class BertForMultiDimensionRegression(BertPreTrainedModel):
    def __init__(self, config, loss_func = nn.MSELoss()):
        super().__init__(config)
        self.num_dim = config.num_dim
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.output_layer = nn.Linear(config.hidden_size, self.num_dim)
        self.loss_func = loss_func

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
            **kwargs
    ):
        """
            output_targets - torch.Tensor of shape (# input sentences, self.num_dim) (same as logits)
        """
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.output_layer(pooled_output)

        outputs = (logits,) + outputs[2:]  # adds hidden states and attention if they are here

        if output_targets is not None:
            loss = self.loss_func(logits, output_targets)
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)
