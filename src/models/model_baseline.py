import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel


class BertForMultiLabelClassification(BertPreTrainedModel):
    def __init__(self,
                 config,
                 pool_mode='cls',
                 **kwargs):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.pool_mode = pool_mode

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)
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
            one_hot_labels=None,
            labels=None,
            **kwargs
    ):
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
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # adds hidden states and attention if they are here

        if one_hot_labels is not None:
            loss = self.loss_func(logits, one_hot_labels)
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)

