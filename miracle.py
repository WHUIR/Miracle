import torch.nn.functional as F
from Miracle.sequential_recommender import SequentialRecommender
from recbole.model.layers import TransformerEncoder
from torch_scatter import scatter_add
import torch
from torch import nn


class PWLayer(nn.Module):
    def __init__(self, input_size, output_size, dropout=0.0):
        super(PWLayer, self).__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.bias = nn.Parameter(torch.zeros(input_size), requires_grad=True)
        self.lin = nn.Linear(input_size, output_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, x):
        return self.lin(self.dropout(x) - self.bias)


class MoEAdaptorLayer(nn.Module):
    def __init__(self, n_exps, layers, dropout=0.0, noise=True):
        super(MoEAdaptorLayer, self).__init__()

        self.n_exps = n_exps
        self.noisy_gating = noise

        self.experts = nn.ModuleList([PWLayer(layers[0], layers[1], dropout) for i in range(n_exps)])
        self.w_gate = nn.Parameter(torch.zeros(layers[0], n_exps), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(layers[0], n_exps), requires_grad=True)

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        clean_logits = x @ self.w_gate
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = ((F.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits).to(x.device) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        gates = F.softmax(logits, dim=-1)
        return gates

    def forward(self, x):
        gates = self.noisy_top_k_gating(x, self.training)  # (B, n_E)
        expert_outputs = [self.experts[i](x).unsqueeze(-2) for i in range(self.n_exps)]  # [(B, 1, D)]
        expert_outputs = torch.cat(expert_outputs, dim=-2)
        multiple_outputs = gates.unsqueeze(-1) * expert_outputs
        return multiple_outputs.sum(dim=-2)


class TimeAwareAttention(nn.Module):
    def __init__(self, config):
        super(TimeAwareAttention, self).__init__()
        self.hidden_size = config['hidden_size']  # same as embedding_size
        self.max_seq_length = config['max_seq_len']
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.attn_linear = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Dropout(p=config['hidden_dropout_prob']),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, 1)
        )
        self.tau = 1

    def forward(self, seq_output, item_seq):
        position_ids = torch.arange(seq_output.size(1), dtype=torch.long, device=seq_output.device)
        position_ids = position_ids.view(1, -1, 1)
        position_embedding = self.position_embedding(position_ids)
        tma_inputs = position_embedding + seq_output
        tma_weight = self.attn_linear(tma_inputs).squeeze(-1) / self.tau

        tma_weight = torch.masked_fill(tma_weight, (item_seq == 0), -1e9)
        tma_weight = F.softmax(tma_weight.view(seq_output.size(0), -1), dim=-1)
        return tma_weight.view(seq_output.size(0), seq_output.size(1), seq_output.size(2))


class MultiInterestExtractor(nn.Module):
    def __init__(self, config):
        super(MultiInterestExtractor, self).__init__()

        self.hidden_size = config['hidden_size']
        self.initializer_range = config['initializer_range']
        self.linear = nn.Linear(self.hidden_size, self.hidden_size)
        self.aspect_embs = nn.Embedding(config['aspects'], config['hidden_size'])

        self.time_aware_attn = TimeAwareAttention(config)

        self.aspects = config['aspects']
        self.tau = 1
        self.noise_scale = config['noise_scale']
        self.device = config['device']
        self.caps_layers = config['caps_layers']
        self.moe_dropout = nn.Dropout(p=config['moe_dropout'])
        self.ln = nn.LayerNorm(self.hidden_size, eps=1e-12)

    def forward_sequence(self, item_emb, item_seq, aspect_mask=None):
        batch_size, seq_len = item_emb.size()[0], item_emb.size()[1]
        tma_weight = self.time_aware_attn(item_emb.unsqueeze(-2), item_seq.unsqueeze(-1))

        gates = self.generate_gates(item_emb)

        if aspect_mask is None:
            topk_gates, topk_gates_idx = torch.topk(gates, dim=-1, k=1)
            src = torch.ones([batch_size, seq_len], device=self.device)
            src = torch.masked_fill(src, item_seq.unsqueeze(-1).view(batch_size, -1) == 0,
                                    0)
            aspect_mask = scatter_add(src,
                                      topk_gates_idx.view(batch_size, -1),
                                      out=torch.zeros([batch_size, self.aspects], device=self.device))
            aspect_mask = aspect_mask == 0
        else:
            aspect_mask = aspect_mask
        item_moe_emb = F.tanh(self.linear(item_emb)) + item_emb
        item_moe_emb = self.ln(self.moe_dropout(item_moe_emb))
        item_moe_emb = item_moe_emb.unsqueeze(2)

        bij = gates
        interest_capsule = self.aspect_embs.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        for i in range(self.caps_layers):
            seq_mask = item_seq == 0
            cij = torch.masked_fill(bij, aspect_mask.unsqueeze(1), -1e9)
            cij = torch.softmax(cij / self.tau, dim=-1)
            cij = torch.masked_fill(cij, seq_mask.unsqueeze(-1), 0)
            interest_capsule = torch.sum(
                cij.unsqueeze(-1) * item_moe_emb * tma_weight.unsqueeze(-1),
                dim=1)
            cap_norm = torch.sum(torch.pow(interest_capsule, 2), dim=-1, keepdim=True)
            scalar_factor = cap_norm / (1 + cap_norm) / torch.sqrt(cap_norm + 1e-9)
            interest_capsule = scalar_factor * interest_capsule

            # Squash
            delta_weight = (item_moe_emb * interest_capsule.unsqueeze(1)).sum(dim=-1)
            bij = bij + delta_weight

        return interest_capsule, F.softmax(gates / self.tau, dim=-1), aspect_mask

    def generate_gates(self, item_emb, noise_epsilon=1e-2):
        clean_gates = item_emb @ self.aspect_embs.weight.t()
        if self.training:
            noise_stddev = clean_gates.detach() * self.noise_scale + noise_epsilon
            noisy_gates = clean_gates + (torch.randn_like(clean_gates).to(item_emb.device) * noise_stddev)
            gates = noisy_gates
        else:
            gates = clean_gates
        return gates

    def forward_item(self, item_emb):
        if len(item_emb.size()) == 2:
            item_emb = item_emb.unsqueeze(1)
        gates = self.generate_gates(item_emb)
        return item_emb.unsqueeze(-2), F.softmax(gates / self.tau, dim=-1)

    def forward_item_all(self):
        pass


class Miracle(SequentialRecommender):
    def __init__(self, config):
        super(Miracle, self).__init__(config)

        # load parameters info
        self.hidden_size = config['hidden_size']  # same as embedding_size
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.layer_norm_eps = config['layer_norm_eps']
        self.n_items = config['item_count']
        self.max_seq_length = config['max_seq_len']
        self.stage = config['stage']

        self.initializer_range = config['initializer_range']
        self.loss_type = config['loss_type']

        # define layers and loss
        if self.stage == 'trans':
            self.item_embedding = nn.Embedding(self.n_items, self.hidden_size)
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.plm_embedding = config['text_emb']

        self.trm_encoder = TransformerEncoder(
            n_layers=config['n_layers'],
            n_heads=config['n_heads'],
            hidden_size=self.hidden_size,
            inner_size=config['inner_size'],
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=config['attn_dropout_prob'],
            hidden_act=config['hidden_act'],
            layer_norm_eps=self.layer_norm_eps
        )

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        self.act = nn.LeakyReLU()

        self.mi_extractor = MultiInterestExtractor(config)
        self.pad_mode = config['pad_mode']
        self.neg_count = config['neg_count']
        self.item_embs = None
        self.balance_alpha = config['balance_alpha']
        self.aspects = config['aspects']
        self.time_aware_attn = TimeAwareAttention(config)
        self.moe_dropout = nn.Dropout(p=config['moe_dropout'])
        self.aspect_cons_tau = config['aspect_cons_tau']
        self.aspect_alpha = config['aspect_alpha']

        self.moe_adaptor = MoEAdaptorLayer(
            config['n_exps'],
            config['adaptor_layers'],
            config['adaptor_dropout_prob']
        )
        self.mask_idx = -1
        self.mask_param = nn.Parameter(torch.zeros(config['hidden_size']).normal_(0, self.initializer_range),
                                       requires_grad=True)

        self.seq_cons_alpha = config['seq_cons_alpha']
        self.seq_cons_tau = config['seq_cons_tau']

        # parameters initialization
        self.apply(self._init_weights)
        self.config = config
        self.item_drop_ratio = 0.2

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def seq_encode(self, item_seq, item_emb, item_seq_len):
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq, bidirectional=True)
        trm_output = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)
        return trm_output[-1]

    def forward(self, item_seq, item_seq_len):
        item_emb = self.moe_adaptor(self.plm_embedding[item_seq].to(self.device))
        if self.stage == 'trans':
            item_emb = item_emb + self.item_embedding(item_seq)
        # Sequential
        seq_output = self.seq_encode(item_seq, item_emb, item_seq_len)
        # MoE
        interests, gates, aspect_mask = self.mi_extractor.forward_sequence(seq_output,
                                                                           item_seq)

        return interests, gates, aspect_mask

    def calculate_loss(self, interaction):
        if self.stage == 'pretrain':
            return self.calculate_pretrain_loss(interaction)
        else:
            return self.calculate_downstream_loss(interaction)

    def calculate_pretrain_loss(self, interaction):
        batch_size = interaction['item_seqs'].size()[0]

        item_seq = interaction['item_seqs']
        item_seq_len = interaction['lengths']
        interests, moe_gates, aspect_mask = self.forward(item_seq, item_seq_len)

        pos_items = interaction['labels']
        total_embs, item_gates = self.mi_extractor.forward_item(
            self.moe_adaptor(self.plm_embedding[pos_items].to(self.device)))
        total_embs = total_embs.squeeze(1)

        total_score = interests.transpose(0, 1) @ total_embs.permute(1, 2, 0)
        total_score = torch.masked_fill(total_score, aspect_mask.t().unsqueeze(-1), -100)
        loss = F.cross_entropy(total_score.max(dim=0)[0],
                               torch.arange(total_embs.size()[0], device=self.device))
        balance_loss = self.cal_balance_loss(moe_gates, item_seq) + self.cal_balance_loss(item_gates,
                                                                                          torch.ones(size=[batch_size],
                                                                                                     device=self.device))
        aspect_contrastive_loss = self.cal_aspects_contrastive_loss()
        mi_cs_loss = self.cal_mI_contrastive_loss(interaction, interests, aspect_mask, moe_gates)

        return [
            loss + self.balance_alpha * balance_loss + self.aspect_alpha * aspect_contrastive_loss + self.seq_cons_alpha * mi_cs_loss]

    def calculate_downstream_loss(self, interaction):
        item_seq = interaction['item_seqs']
        item_seq_len = interaction['lengths']
        interests, moe_gates, aspect_mask = self.forward(item_seq, item_seq_len)
        total_embs = self.moe_adaptor(self.plm_embedding.to(self.device))
        if self.stage == 'trans':
            total_embs = total_embs + self.item_embedding.weight
        total_embs, item_gates = self.mi_extractor.forward_item(total_embs)
        total_embs = total_embs.squeeze(1)

        total_score = interests.transpose(0, 1) @ total_embs.permute(1, 2, 0)
        total_score = total_score.permute(1, 2, 0)
        total_score = torch.masked_fill(total_score, aspect_mask.unsqueeze(1), -100)
        total_score = torch.max(total_score, dim=-1)[0]
        loss = F.cross_entropy(total_score,
                               interaction['labels'])
        balance_loss = self.cal_balance_loss(moe_gates, item_seq) + self.cal_balance_loss(item_gates, torch.ones(
            size=[item_gates.size()[0]], device=self.device))
        aspect_contrastive_loss = self.cal_aspects_contrastive_loss()

        return [loss + self.balance_alpha * balance_loss + self.aspect_alpha * aspect_contrastive_loss]

    def predict(self, interaction):
        item_seq = interaction['item_seqs']
        item_seq_len = interaction['lengths']
        interests, moe_gates, aspect_mask = self.forward(item_seq, item_seq_len)

        pos_items = interaction['labels']
        neg_items = interaction['neg_items']

        total_items = torch.cat([pos_items.unsqueeze(-1), neg_items], dim=1)
        total_embs, item_gates = self.mi_extractor.forward_item(
            self.moe_adaptor(self.plm_embedding[total_items].to(self.device)))

        total_score = (interests.unsqueeze(1) * total_embs).sum(dim=-1)
        mi_logits = torch.masked_fill(total_score, aspect_mask.unsqueeze(1), -100)

        logits = torch.max(mi_logits, dim=-1)[0]
        return logits, torch.zeros(interaction['labels'].size()[0])

    def full_sort_predict(self, interaction):
        item_seq = interaction['item_seqs']
        item_seq_len = interaction['lengths']
        interests, moe_gates, aspect_mask = self.forward(item_seq, item_seq_len)

        total_embs = self.moe_adaptor(self.plm_embedding.to(self.device))
        if self.stage == 'trans':
            total_embs = total_embs + self.item_embedding.weight
        total_embs, item_gates = self.mi_extractor.forward_item(total_embs)
        total_embs = total_embs.squeeze(1)
        total_score = interests.transpose(0, 1) @ total_embs.permute(1, 2, 0)
        total_score = total_score.permute(1, 2, 0)
        total_score = torch.masked_fill(total_score, aspect_mask.unsqueeze(1), -100)
        total_score = torch.max(total_score, dim=-1)[0]
        return total_score, interaction['labels']

    def cal_balance_loss(self, gates, item_seq):
        gates = gates.view(-1, gates.size()[-1])
        gates = gates[item_seq.view(-1) != 0]

        _, idx = gates.max(dim=-1)

        p = gates.mean(dim=0)
        f = scatter_add(torch.ones(size=idx.size(), device=self.device), idx,
                        out=torch.zeros(size=p.size(), device=self.device)) / gates.size()[0]
        return (f * p).sum()

    def cal_aspects_contrastive_loss(self):
        embs = self.mi_extractor.aspect_embs.weight
        embs = F.normalize(embs, dim=-1)
        sim = embs @ embs.t()
        sim = sim / self.aspect_cons_tau
        loss = F.cross_entropy(sim, torch.arange(embs.size()[0], device=self.device))
        return loss

    def cal_mI_contrastive_loss(self, interaction, interests, aspect_mask, gates):
        item_seq, item_seq_len = interaction['item_seqs'], interaction['lengths']
        item_seq_aug, item_seq_len_aug, seq_aug_mask = self.seq_aug(item_seq, item_seq_len)
        item_emb_aug = self.moe_adaptor(
            self.plm_embedding[item_seq_aug].to(self.device))
        item_emb_aug[item_seq_aug == self.mask_idx] = self.mask_param.data
        seq_output_aug = self.seq_encode(item_seq_aug, item_emb_aug,
                                         item_seq_len_aug)
        interests_aug, gates_aug, aspect_mask_aug = self.mi_extractor.forward_sequence(seq_output_aug,
                                                                                       item_seq,
                                                                                       aspect_mask)

        mask_item_gates = gates[item_seq_aug == self.mask_idx]
        _, mask_item_gates_max_idx = mask_item_gates.max(dim=-1)

        row_idx = (item_seq_aug == self.mask_idx).nonzero()[:, 0]
        interests = interests[row_idx, mask_item_gates_max_idx]
        interests_aug = interests_aug[row_idx, mask_item_gates_max_idx]
        interests_sim = interests @ interests_aug.t()
        mi_cs_loss = F.cross_entropy(interests_sim, torch.arange(interests_sim.size()[0], device=self.device))
        return mi_cs_loss

    def downstream_freeze_parameter(self):
        for _ in self.position_embedding.parameters():
            _.requires_grad = False
        for _ in self.trm_encoder.parameters():
            _.requires_grad = False

    def batch_step(self):
        self.n_batch += 1
        self.mi_extractor.tau = 1

    def seq_aug(self, item_seq, item_seq_len):
        item_seq = item_seq.cpu()
        item_seq_len = item_seq_len.cpu()
        mask_p = torch.full_like(item_seq, self.item_drop_ratio, dtype=torch.float)
        mask = torch.bernoulli(mask_p).to(torch.bool)
        mask[:, -1] = False
        mask = torch.masked_fill(mask, item_seq == 0, False)
        mask[item_seq_len < 5] = False
        item_seq_aug = torch.masked_fill(item_seq, mask, self.mask_idx)  # -1 represents [mask]
        item_seq_len_aug = (item_seq_aug != 0).sum(dim=-1)
        return item_seq_aug.to(self.device), item_seq_len_aug.to(self.device), mask

