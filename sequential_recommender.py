import torch.nn as nn
import torch


class SequentialRecommender(nn.Module):
    """
    This is a abstract sequential recommender. All the sequential model should implement This class.
    """

    def __init__(self, config):
        super(SequentialRecommender, self).__init__()

        # load parameters info
        self.device = config['device']
        self.n_batch = 0
        self.n_epoch = 0

    def gather_indexes(self, output, gather_index):
        """Gathers the vectors at the specific positions over a minibatch"""
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)

    def get_attention_mask(self, item_seq, bidirectional=False):
        """Generate left-to-right uni-directional or bidirectional attention mask for multi-head attention."""
        attention_mask = (item_seq != 0)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.bool
        if not bidirectional:
            extended_attention_mask = torch.tril(extended_attention_mask.expand((-1, -1, item_seq.size(-1), -1)))
        extended_attention_mask = torch.where(extended_attention_mask, 0., -10000.)
        return extended_attention_mask

    def batch_step(self):
        pass

    def epoch_step(self):
        pass
