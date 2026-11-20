import torch
import torch.nn as nn
import torch.nn.functional as F


def row_normalized_sparse(adjacency):
    adjacency = adjacency.coalesce()
    indices = adjacency.indices()
    values = adjacency.values()
    degree = torch.zeros(
        adjacency.size(0), dtype=values.dtype, device=values.device
    )
    degree.scatter_add_(0, indices[0], values)
    normalized = values / torch.clamp(degree[indices[0]], min=1.0)
    return torch.sparse_coo_tensor(
        indices,
        normalized,
        adjacency.size(),
        dtype=values.dtype,
        device=values.device,
    ).coalesce()


def centered_neighbor_delta(adjacency, h):

    adjacency = adjacency.coalesce()
    message = torch.sparse.mm(row_normalized_sparse(adjacency), h)
    active = torch.zeros(adjacency.size(0), dtype=h.dtype, device=h.device)
    if adjacency._nnz():
        active[adjacency.indices()[0].unique()] = 1.0
    return (message - h) * active.unsqueeze(1)


class GENELink(nn.Module):

    def __init__(self, input_dim, hidden1_dim, hidden2_dim, hidden3_dim,
                 hidden4_dim, output_dim, num_head1, num_head2, num_head3,
                 alpha, device, type, reduction):
        super(GENELink, self).__init__()
        self.num_head1 = num_head1
        self.num_head2 = num_head2
        self.num_head3 = num_head3
        self.device = device
        self.alpha = alpha
        self.type = type
        self.reduction = reduction

        if self.reduction == 'mean':
            self.hidden1_dim = hidden1_dim
            self.hidden2_dim = hidden2_dim
            self.hidden3_dim = hidden3_dim
        elif self.reduction == 'concate':
            self.hidden1_dim = num_head1 * hidden1_dim
            self.hidden2_dim = num_head2 * hidden2_dim
            self.hidden3_dim = num_head3 * hidden3_dim

        self.ConvLayer1 = [
            AttentionLayer(input_dim, hidden1_dim, alpha)
            for _ in range(num_head1)
        ]
        for i, attention in enumerate(self.ConvLayer1):
            self.add_module('ConvLayer1_AttentionHead{}'.format(i), attention)
        self.ConvLayer2 = [
            AttentionLayer(self.hidden1_dim, hidden2_dim, alpha)
            for _ in range(num_head2)
        ]
        for i, attention in enumerate(self.ConvLayer2):
            self.add_module('ConvLayer2_AttentionHead{}'.format(i), attention)
        self.ConvLayer3 = [
            AttentionLayer(self.hidden2_dim, hidden3_dim, alpha)
            for _ in range(num_head3)
        ]
        for i, attention in enumerate(self.ConvLayer3):
            self.add_module('ConvLayer3_AttentionHead{}'.format(i), attention)

        self.tf_linear1 = nn.Linear(hidden3_dim, hidden4_dim)
        self.target_linear1 = nn.Linear(hidden3_dim, hidden4_dim)
        self.tf_linear2 = nn.Linear(hidden4_dim, output_dim)
        self.target_linear2 = nn.Linear(hidden4_dim, output_dim)

        if self.type == 'MLP':
            self.linear1 = nn.Linear(2 * output_dim, 16)
            self.linear2 = nn.Linear(16, 1)
        elif self.type == 'b_dot':
            self.bias_weight = nn.Parameter(
                torch.FloatTensor(output_dim, output_dim)
            )
            nn.init.xavier_uniform_(self.bias_weight.data, gain=1.414)

        self.reset_parameters()

    def reset_parameters(self):
        for attention in self.ConvLayer1:
            attention.reset_parameters()
        for attention in self.ConvLayer2:
            attention.reset_parameters()
        nn.init.xavier_uniform_(self.tf_linear1.weight, gain=1.414)
        nn.init.xavier_uniform_(self.target_linear1.weight, gain=1.414)
        nn.init.xavier_uniform_(self.tf_linear2.weight, gain=1.414)
        nn.init.xavier_uniform_(self.target_linear2.weight, gain=1.414)

    def encode(self, x, adj):
        if self.reduction == 'concate':
            x = torch.cat([att(x, adj) for att in self.ConvLayer1], dim=1)
            x = F.elu(x)
        elif self.reduction == 'mean':
            x = torch.mean(
                torch.stack([att(x, adj) for att in self.ConvLayer1]), dim=0
            )
            x = F.elu(x)
        else:
            raise TypeError

        if self.reduction == 'concate':
            x = torch.cat([att(x, adj) for att in self.ConvLayer2], dim=1)
            x = F.elu(x)
        elif self.reduction == 'mean':
            x = torch.mean(
                torch.stack([att(x, adj) for att in self.ConvLayer2]), dim=0
            )
            x = F.elu(x)
        else:
            raise TypeError

        return torch.mean(
            torch.stack([att(x, adj) for att in self.ConvLayer3]), dim=0
        )

    def decode(self, tf_embed, target_embed):
        if self.type == 'dot':
            prob = torch.mul(tf_embed, target_embed)
            return torch.sum(prob, dim=1).view(-1, 1)
        if self.type == 'cosine':
            return torch.cosine_similarity(
                tf_embed, target_embed, dim=1
            ).view(-1, 1)
        if self.type == 'MLP':
            h = torch.cat([tf_embed, target_embed], dim=1)
            prob = F.elu(self.linear1(h))
            return F.elu(self.linear2(prob))
        if self.type == 'b_dot':
            return torch.sum(
                torch.matmul(tf_embed, self.bias_weight) * target_embed,
                dim=1,
            ).view(-1, 1)
        raise TypeError(r'{} is not available'.format(self.type))

    def forward(self, x, adj, train_sample, return_logits=True):
        embed = self.encode(x, adj)

        tf_embed = self.tf_linear1(embed)
        tf_embed = F.leaky_relu(tf_embed)
        tf_embed = F.dropout(tf_embed, p=0.01, training=self.training)
        tf_embed = self.tf_linear2(tf_embed)
        tf_embed = F.leaky_relu(tf_embed)
        tf_embed = F.dropout(tf_embed, p=0.01, training=self.training)

        target_embed = self.target_linear1(embed)
        target_embed = F.leaky_relu(target_embed)
        target_embed = F.dropout(target_embed, p=0.01, training=self.training)
        target_embed = self.target_linear2(target_embed)
        target_embed = F.leaky_relu(target_embed)
        target_embed = F.dropout(
            target_embed, p=0.01, training=self.training
        )

        self.tf_output = tf_embed
        self.target_output = target_embed
        train_tf = tf_embed[train_sample[:, 0].long()]
        train_target = target_embed[train_sample[:, 1].long()]
        logits = self.decode(train_tf, train_target)

        if return_logits:
            return logits
        if self.type == 'MLP' and logits.size(1) == 2:
            return torch.softmax(logits, dim=1)
        return torch.sigmoid(logits)

    def get_embedding(self):
        return self.tf_output, self.target_output


class CUIMAIN(GENELink):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hidden2 = int(args[2] if len(args) > 2 else kwargs['hidden2_dim'])
        hidden3 = int(args[3] if len(args) > 3 else kwargs['hidden3_dim'])
        heads2 = int(args[7] if len(args) > 7 else kwargs['num_head2'])
        if self.reduction != 'concate' or hidden2 != hidden3:
            raise ValueError(
                'Frozen ILRR requires concate and equal H2/H3 head widths'
            )

        self.motif_film_scale = nn.Parameter(
            torch.zeros(hidden3, dtype=torch.float32)
        )
        self.evidence_film_scale = nn.Parameter(
            torch.zeros(hidden3, dtype=torch.float32)
        )
        self.ilrr_gamma2 = nn.Parameter(torch.zeros((), dtype=torch.float32))
        self.h2_head_width = hidden2
        self.h2_head_count = heads2
        self.last_h_chip = None
        self.last_motif_message = None
        self.last_evidence_message = None
        self.last_delta = None
        self.last_h2_post = None
        self.last_h2_mean = None
        self.last_ilrr_delta = None

    def adapter_parameters(self):
        return [
            self.motif_film_scale,
            self.evidence_film_scale,
            self.ilrr_gamma2,
        ]

    def _encode_main(self, x, adjacency):
        h1 = torch.cat(
            [head(x, adjacency) for head in self.ConvLayer1], dim=1
        )
        h1 = F.elu(h1)
        h2 = torch.cat(
            [head(h1, adjacency) for head in self.ConvLayer2], dim=1
        )
        h2 = F.elu(h2)
        self.last_h2_post = h2
        self.last_h2_mean = h2.reshape(
            h2.size(0), self.h2_head_count, self.h2_head_width
        ).mean(dim=1)
        return torch.mean(
            torch.stack(
                [head(h2, adjacency) for head in self.ConvLayer3]
            ),
            dim=0,
        )

    def encode(self, x, adjacencies):
        if not isinstance(adjacencies, (tuple, list)) or len(adjacencies) != 3:
            raise TypeError('Expected (A_chip, A_motif_comp, B_support)')
        a_chip, a_motif, b_support = adjacencies
        h_chip = self._encode_main(x, a_chip)
        motif_message = centered_neighbor_delta(a_motif, h_chip)
        evidence_message = centered_neighbor_delta(b_support, h_chip)
        delta = (
            motif_message * torch.tanh(self.motif_film_scale)
            + evidence_message * torch.tanh(self.evidence_film_scale)
        )
        self.last_h_chip = h_chip
        self.last_motif_message = motif_message
        self.last_evidence_message = evidence_message
        self.last_delta = delta
        h_c = h_chip + delta
        self.last_ilrr_delta = self.ilrr_gamma2 * self.last_h2_mean
        return h_c + self.last_ilrr_delta

    def contribution_stats(self):
        values = {
            'ilrr_parameter_count': 1.0,
            'ilrr_gamma2': float(self.ilrr_gamma2.detach().item()),
        }
        for name, parameter in (
            ('motif', self.motif_film_scale),
            ('evidence', self.evidence_film_scale),
        ):
            scale = torch.tanh(parameter.detach())
            values[name + '_scale_mean'] = float(scale.mean().item())
            values[name + '_scale_abs_mean'] = float(scale.abs().mean().item())
            values[name + '_scale_abs_max'] = float(scale.abs().max().item())
        if self.last_h_chip is not None:
            chip = float(self.last_h_chip.detach().norm().item())
            delta = float(self.last_delta.detach().norm().item())
            ilrr = float(self.last_ilrr_delta.detach().norm().item())
            hc = float(
                (self.last_h_chip + self.last_delta).detach().norm().item()
            )
            values.update(
                h_chip_norm=chip,
                adapter_delta_norm=delta,
                adapter_delta_to_h_chip_ratio=delta / (chip + 1e-12),
                motif_message_norm=float(
                    self.last_motif_message.detach().norm().item()
                ),
                evidence_message_norm=float(
                    self.last_evidence_message.detach().norm().item()
                ),
                h2_post_norm=float(self.last_h2_post.detach().norm().item()),
                h2_mean_norm=float(self.last_h2_mean.detach().norm().item()),
                ilrr_delta_norm=ilrr,
                ilrr_delta_to_hc_ratio=ilrr / (hc + 1e-12),
            )
        return values


class AttentionLayer(nn.Module):
    def __init__(self, input_dim, output_dim, alpha=0.2, bias=True,
                 add_residual=True):
        super(AttentionLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.alpha = alpha
        self.weight = nn.Parameter(
            torch.FloatTensor(self.input_dim, self.output_dim)
        )
        self.weight_interact = nn.Parameter(
            torch.FloatTensor(self.input_dim, self.output_dim)
        )
        self.a = nn.Parameter(torch.zeros(size=(2 * self.output_dim, 1)))
        self.add_residual = add_residual
        if add_residual and input_dim != output_dim:
            self.residual_linear = nn.Linear(input_dim, output_dim)
        else:
            self.residual_linear = None
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(self.output_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight.data, gain=1.414)
        nn.init.xavier_uniform_(self.weight_interact.data, gain=1.414)
        if self.bias is not None:
            self.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, x, adj):
        h = torch.matmul(x, self.weight)
        h = F.leaky_relu(h, self.alpha)
        e = self._prepare_attentional_mechanism_input(h)
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj.to_dense() > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, p=0.5, training=self.training)
        h_prime = torch.matmul(attention, h)
        h_prime = F.normalize(h_prime, p=2, dim=1)
        if self.bias is not None:
            h_prime += self.bias
        if self.add_residual:
            if self.residual_linear is not None:
                residual = self.residual_linear(x)
            else:
                residual = x
            h_prime += residual
        return h_prime

    def _prepare_attentional_mechanism_input(self, wh):
        wh1 = torch.matmul(wh, self.a[:self.output_dim, :])
        wh2 = torch.matmul(wh, self.a[self.output_dim:, :])
        return F.leaky_relu(wh1 + wh2.T, negative_slope=self.alpha)
