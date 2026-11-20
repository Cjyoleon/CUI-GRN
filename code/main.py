from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from scGNN5 import CUIMAIN, GENELink
from utils import (scRNADataset, load_data, adj2saprse_tensor, Evaluation,
                   Network_Statistic, adjacency_from_directed,
                   support_mask_from_directed, pair_set,
                   sample_uniform_training_sets, MEMBER_SEEDS)
import pandas as pd
import numpy as np
import random
import os
import time
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=3e-3, help='Initial learning rate.')
parser.add_argument('--epochs', type=int, default=30, help='Number of epoch.')
parser.add_argument('--num_head', type=list, default=[3, 3, 3], help='Number of head attentions.')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--hidden_dim', type=int, default=[128, 64, 64, 32], help='The dimension of hidden layer')
parser.add_argument('--output_dim', type=int, default=16, help='The dimension of latent layer')
parser.add_argument('--batch_size', type=int, default=256, help='The size of each batch')
parser.add_argument('--loop', type=bool, default=False, help='whether to add self-loop in adjacent matrix')
parser.add_argument('--seed', type=int, default=8, help='Random seed')
parser.add_argument('--Type', type=str, default='b_dot', help='score metric')
parser.add_argument('--flag', type=bool, default=False, help='the identifier whether to conduct causal inference')
parser.add_argument('--reduction', type=str, default='concate', help='how to integrate multihead attention')
parser.add_argument('--net', type=str, default='Non-Specific')
parser.add_argument('--num', type=int, default=1000)
parser.add_argument('--data', type=str, default='mHSC-GM')
args = parser.parse_args()


data_type = args.data
num = args.num
net_type = args.net


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def embed2file(tf_embed, tg_embed, gene_file, tf_path, target_path):
    gene_set = pd.read_csv(gene_file, index_col=0)
    pd.DataFrame(
        tf_embed.cpu().detach().numpy(), index=gene_set['Gene'].values
    ).to_csv(tf_path)
    pd.DataFrame(
        tg_embed.cpu().detach().numpy(), index=gene_set['Gene'].values
    ).to_csv(target_path)


def read_edges(path):
    frame = pd.read_csv(path)
    return frame[['tf_index', 'target_index']].astype(np.int64).to_numpy()


def read_labeled_frame(path):
    """Read exactly the three columns used by the frozen formal runner."""
    return (
        pd.read_csv(path, index_col=0)
        .loc[:, ['TF', 'Target', 'Label']]
        .astype(np.int64)
        .reset_index(drop=True)
    )


def read_pair_ids(path):
    return pd.read_csv(path, usecols=['TF', 'Target']).astype(np.int64)


def build_model(input_dim, device):
    return CUIMAIN(
        input_dim=input_dim,
        hidden1_dim=args.hidden_dim[0],
        hidden2_dim=args.hidden_dim[1],
        hidden3_dim=args.hidden_dim[2],
        hidden4_dim=args.hidden_dim[3],
        output_dim=args.output_dim,
        num_head1=args.num_head[0],
        num_head2=args.num_head[1],
        num_head3=args.num_head[2],
        alpha=args.alpha,
        device=device,
        type=args.Type,
        reduction=args.reduction,
    ).to(device)


def build_primary_model(input_dim, device):
    return GENELink(
        input_dim=input_dim,
        hidden1_dim=args.hidden_dim[0],
        hidden2_dim=args.hidden_dim[1],
        hidden3_dim=args.hidden_dim[2],
        hidden4_dim=args.hidden_dim[3],
        output_dim=args.output_dim,
        num_head1=args.num_head[0],
        num_head2=args.num_head[1],
        num_head3=args.num_head[2],
        alpha=args.alpha,
        device=device,
        type=args.Type,
        reduction=args.reduction,
    ).to(device)


def predict(model, feature, adjacencies, samples):
    model.eval()
    with torch.no_grad():
        logits = model(feature, adjacencies, samples.to(feature.device))
        return torch.softmax(logits, dim=1) if args.flag else torch.sigmoid(logits)


set_seed(args.seed)
Network_Statistic(data_type, num, net_type)

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dataset_dir = os.path.join(base_dir, 'Dataset', 'Benchmark Dataset', net_type + ' Dataset', data_type, 'TFs+{}'.format(num))
split_dir = os.path.join(base_dir, net_type, data_type + ' ' + str(num))
result_dir = os.path.join(base_dir, 'Result', data_type + ' ' + str(num))
model_dir = os.path.join(base_dir, 'model')
for directory in [result_dir, model_dir]:
    if not os.path.exists(directory):
        os.makedirs(directory)

exp_file = os.path.join(dataset_dir, 'BL--ExpressionData.csv')
tf_file = os.path.join(dataset_dir, 'TF.csv')
target_file = os.path.join(dataset_dir, 'Target.csv')
label_file = os.path.join(dataset_dir, 'Label.csv')
motif_file = os.path.join(dataset_dir, 'A_M_COMP.csv')
support_file = os.path.join(dataset_dir, 'B_REAL.csv')
train_file = os.path.join(split_dir, 'Train_set.csv')
val_file = os.path.join(split_dir, 'Validation_set.csv')
test_file = os.path.join(split_dir, 'Test_set.csv')

data_input = pd.read_csv(exp_file, index_col=0)
feature = torch.from_numpy(load_data(data_input).exp_data())
tf = pd.read_csv(tf_file, index_col=0)['index'].values.astype(np.int64)
targets = pd.read_csv(target_file, index_col=0)['index'].values.astype(np.int64)
train_frame = read_labeled_frame(train_file)
validation_frame = read_labeled_frame(val_file)
test_pairs = read_pair_ids(test_file)


original_train_array = train_frame[
    ['TF', 'Target', 'Label']
].to_numpy(np.int64)
train_graph = scRNADataset(
    original_train_array, feature.shape[0], flag=args.flag
)
adj_chip = train_graph.Adj_Generate(torch.from_numpy(tf), loop=args.loop)
motif_edges = read_edges(motif_file)
support_edges = read_edges(support_file)
if not set(map(tuple, support_edges)) <= set(map(tuple, motif_edges)):
    raise AssertionError('B_REAL contains edges outside A_M_COMP')
adj_motif = adjacency_from_directed(motif_edges, feature.shape[0])
adj_support = support_mask_from_directed(support_edges, feature.shape[0])

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
data_feature = feature.to(device)
adjacencies = tuple(
    adj2saprse_tensor(adjacency).coalesce().to(device)
    for adjacency in (adj_chip, adj_motif, adj_support)
)
validation_data = torch.from_numpy(validation_frame.to_numpy()).to(device)


if net_type == 'Specific':
    train_arrays = [original_train_array]
    member_seeds = (args.seed,)
else:
    labels = pd.read_csv(label_file, index_col=0)
    evaluation_pairs = pair_set(validation_frame) | pair_set(test_pairs)
    known_positives = pair_set(labels)
    train_arrays = sample_uniform_training_sets(
        train_frame, evaluation_pairs, known_positives, targets
    )
    member_seeds = MEMBER_SEEDS

models = []
start_time = time.time()
for member, (member_seed, train_array) in enumerate(
        zip(member_seeds, train_arrays), start=1):
    set_seed(member_seed)
    model = build_model(feature.size()[1], device)


    set_seed(member_seed)
    primary_model = build_primary_model(feature.size()[1], device)
    primary_state = {
        name: tensor.detach().clone()
        for name, tensor in primary_model.state_dict().items()
    }
    del primary_model
    state = model.state_dict()
    for name, tensor in primary_state.items():
        state[name] = tensor.clone().to(state[name].device)
    model.load_state_dict(state, strict=True)


    set_seed(member_seed)
    train_load = scRNADataset(train_array, feature.shape[0], flag=False)
    generator = torch.Generator().manual_seed(member_seed)
    loader = DataLoader(
        train_load, batch_size=args.batch_size, shuffle=True,
        generator=generator
    )

    special = model.adapter_parameters()
    special_ids = {id(parameter) for parameter in special}
    ordinary = [parameter for parameter in model.parameters()
                if id(parameter) not in special_ids]
    optimizer = Adam([
        {'params': ordinary, 'weight_decay': 0.015},
        {'params': special, 'weight_decay': 0.0},
    ], lr=args.lr)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.999)

    for epoch in range(args.epochs):
        running_loss = 0.0
        for train_x, train_y in loader:
            model.train()
            optimizer.zero_grad(set_to_none=True)
            train_x = train_x.to(device)
            train_y = train_y.to(device)
            if not args.flag:
                train_y = train_y.float().view(-1)
            logits = model(data_feature, adjacencies, train_x)
            prediction = (torch.softmax(logits, dim=1) if args.flag
                          else torch.sigmoid(logits.view(-1)))
            loss = F.binary_cross_entropy(prediction, train_y)
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite loss')
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()

        score = predict(model, data_feature, adjacencies, validation_data)
        AUC, AUPR, _ = Evaluation(
            y_pred=score, y_true=validation_data[:, -1], flag=args.flag
        )
        print('Member:{}'.format(member),
              'Epoch:{}'.format(epoch + 1),
              'train loss:{}'.format(running_loss),
              'AUC:{:.3F}'.format(AUC),
              'AUPR:{:.3F}'.format(AUPR))

    suffix = '' if len(member_seeds) == 1 else '_member{}'.format(member)
    torch.save(
        model.state_dict(),
        os.path.join(model_dir, data_type + ' ' + str(num) + suffix + '.pkl')
    )
    tf_embed, target_embed = model.get_embedding()
    embed2file(
        tf_embed, target_embed, target_file,
        os.path.join(result_dir, 'Channel1' + suffix + '.csv'),
        os.path.join(result_dir, 'Channel2' + suffix + '.csv'),
    )
    models.append(model)

print('Total training time: {:.2f} seconds'.format(time.time() - start_time))


test_frame = read_labeled_frame(test_file)
test_data = torch.from_numpy(test_frame.to_numpy()).to(device)
member_scores = [predict(model, data_feature, adjacencies, test_data)
                 for model in models]
score = torch.stack(member_scores, dim=0).mean(dim=0)
AUC, AUPR, _ = Evaluation(
    y_pred=score, y_true=test_data[:, -1], flag=args.flag
)
print('AUC:{}'.format(AUC), 'AUPRC:{}'.format(AUPR))
