import torch
from loguru import logger
from torchcrf import CRF
from typing import Optional
from transformers import AutoConfig
from transformers import AutoModel, AutoModelForTokenClassification


class TransformersBiLstmCrf(torch.nn.Module):
    def __init__(self, config: AutoConfig):
        super(TransformersBiLstmCrf, self).__init__()

        self.transformers = AutoModel.from_config(config=config)
        self.num_labels = config.num_labels
        self.bi_lstm = torch.nn.LSTM(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )

        self.crf = CRF(config.num_labels, batch_first=True)

    def forward(self,
                input_ids: Optional[torch.LongTensor] = None,
                token_type_ids: Optional[torch.LongTensor] = None,
                attention_mask: Optional[torch.FloatTensor] = None,
                labels: Optional[torch.LongTensor] = None,
                valid_ids: Optional[torch.LongTensor] = None,
                label_masks: Optional[torch.LongTensor] = None):

        seq_output = self.transformers(input_ids=input_ids,
                                       token_type_ids=token_type_ids,
                                       attention_mask=attention_mask,
                                       head_mask=None)[0]

        seq_output, _ = self.bi_lstm(seq_output)

        batch_size, max_len, feat_dim = seq_output.shape
        valid_output = torch.zeros(batch_size, max_len, feat_dim, dtype=torch.float32, device=seq_output.device)
        for i in range(batch_size):
            jj = -1
            for j in range(max_len):
                if valid_ids[i][j].item() == 1:
                    jj += 1
                    valid_output[i][jj] = seq_output[i][j]

        sequence_output = self.dropout(valid_output)
        logits = self.classifier(sequence_output)
        seq_tags = self.crf.decode(logits, mask=label_masks != 0)
        if labels is not None:
            log_likelihood = self.crf(logits, labels, mask=label_masks.type(torch.uint8))
            return -1.0 * log_likelihood, seq_tags
        else:
            return seq_tags


if __name__ == "__main__":
    conf = AutoConfig.from_pretrained("vinai/phobert-base")
    net = TransformersBiLstmCrf(config=conf)
    print(net)
    input_ids = torch.randint(0, 100, [2, 20], dtype=torch.long)
    mask = torch.ones([2, 20], dtype=torch.long)
    output = net(input_ids=input_ids, attention_mask=mask)
